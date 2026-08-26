unit Prism.Vulkan;

{ Vulkan compute backend for Prism's quantized MatVec.

  This is the unit that finally puts GGUF models on the GPU. The CPU kernels
  in Prism.Vector stay as the reference implementation and as the fallback:
  every single decision here can degrade to "return False, let the CPU do
  it", and nothing in Prism breaks if Vulkan is missing, broken, or out of
  memory.

  How it works
  ------------
  * Weights stay QUANTIZED in VRAM. A tensor's bytes are uploaded verbatim,
    exactly as they sit in the GGUF file, and the shader dequantizes inside
    the kernel. Dequantizing on the host would throw away the factor-4
    bandwidth advantage that made quantization worth having.
  * Weights are RESIDENT. Uploaded once on first use, then reused for every
    subsequent token. This is the whole point -- the old OpenCL path
    re-uploaded through a 256 MB cache and therefore never got ahead of the
    CPU.
  * VRAM is filled first-come. Because Prism.Llama walks layers in order,
    that naturally yields llama.cpp's n_gpu_layers behaviour: the first N
    layers land on the GPU, the remainder stays on the CPU, and a mixed
    model simply runs at a mixed speed.

  Known limitation of this stage
  ------------------------------
  One submit + one fence per MatVec. The activation round-trip is small
  (a few KB each way through a mapped staging buffer), but the fence costs
  roughly 20-50 us of driver latency, and a token needs 7 MatVecs per layer.
  That caps the ceiling somewhere in the low hundreds of tokens/s, which is
  far above what we get today but is the next thing to fix: recording a whole
  token as one command buffer and keeping the KV cache in VRAM removes it.

  Correctness
  -----------
  The bit layouts in shaders/matvec.comp were transcribed by hand from
  Prism.Vector.pas. That is exactly the kind of code that is wrong in a way
  which merely *looks* plausible, so Init runs SelfTest: synthetic tensors of
  every supported type are evaluated on both paths and compared. If they
  disagree, the backend refuses to activate and Prism stays on the CPU. }

{$POINTERMATH ON}

interface

uses
  System.SysUtils, System.Classes, System.SyncObjs,
  System.Generics.Collections, System.Math, System.StrUtils,
  Prism.Vector, Prism.Vulkan.Api;

type
  TVulkanStats = record
    DeviceName: string;
    Discrete: Boolean;
    VramBytes: Int64;
    BudgetBytes: Int64;
    ResidentBytes: Int64;
    ResidentTensors: Integer;
    Dispatches: Int64;
    CpuFallbacks: Int64;
    BudgetRejects: Int64;
    Broken: Boolean;
    function Describe: string;
  end;

{ Brings the backend up. BudgetMB <= 0 means "decide from the device's VRAM".
  Returns False with a human-readable Info when Vulkan is unavailable or the
  self-test fails -- both are non-fatal, Prism then runs on the CPU. }
function VulkanInit(BudgetMB: Integer; const Log: TProc<string>;
  out Info: string): Boolean;
procedure VulkanShutdown;
function VulkanReady: Boolean;
function VulkanStats: TVulkanStats;

{ Runs the CPU-vs-GPU layout comparison on demand (Init already does it). }
function VulkanSelfTest(out Report: string): Boolean;

{ The hook Prism.Vector calls. True = computed on the GPU, False = caller
  must run the CPU kernel. }
function VulkanMatVec(const T: TQTensor; Y, X: PSingle): Boolean;

{ Same thing for Prism.Inference's plain F32 weight blocks, which live inside
  one big parameter array rather than in a TQTensor. WKey/WKeyOff identify the
  owning block so the caller can invalidate on eviction; the residency key is
  the resulting weight address. }
function VulkanMatVecF32(Y, W, X: PSingle; Rows, Cols: Integer;
  Bias: PSingle; Owner: Pointer): Boolean;

{ Drops a tensor's VRAM copy. Prism.Llama must call this when it evicts a
  streamed layer: the TBytes behind the tensor is freed then, and a later
  allocation could hand the same address to a different tensor. }
procedure VulkanEvict(const T: TQTensor);

{ Drops every VRAM copy that came out of one weight block -- what
  Prism.Streaming needs when it evicts a layer/expert cluster. }
procedure VulkanEvictOwner(Owner: Pointer);
procedure VulkanEvictAll;

implementation

{$I Prism.Vulkan.Shaders.inc}

const
  { Rows per dispatch. The Vulkan spec guarantees at least 65535 workgroups
    in X, so we chunk rather than query -- Llama 3's 128256-row output tensor
    exceeds it. }
  MAX_GROUPS_X = 65535;

  { Default activation/result capacities, in floats. Covers every model we
    care about (largest FFN dim ~28k, largest vocab ~256k); grown on demand. }
  DEF_X_FLOATS = 64 * 1024;
  DEF_Y_FLOATS = 256 * 1024;

  STAGE_BYTES = 32 * 1024 * 1024;   // weight-upload staging chunk

  { Leave room for the display, the compositor and other processes. }
  VRAM_HEADROOM  = Int64(384) * 1024 * 1024;
  VRAM_FRACTION  = 0.90;

  FENCE_TIMEOUT_NS = UInt64(10) * 1000 * 1000 * 1000;  // 10 s

  MAX_RESIDENT_TENSORS = 4096;

  { The shader reads weights four bytes at a time, and quant block strides
    (18/20/34/210 B) put groups on 2-byte boundaries, so the last fetch of a
    tensor can reach up to 6 bytes past its final byte. Padding every tensor
    buffer keeps that read in bounds without having to turn on
    robustBufferAccess, which costs performance on every access. }
  BUF_SLACK = 16;

type
  TMatVecPush = record
    Rows, Cols, RowBytes, RowBase, XOff, YOff: UInt32;
  end;

  TResident = record
    Buf: TVkBuffer;
    Mem: TVkDeviceMemory;
    DescSet: TVkDescriptorSet;
    Bytes: Int64;
    Typ: TGgmlType;
    Rows, Cols: Integer;
    { Whoever owns the host memory this was copied from. For a GGUF tensor
      that is the tensor itself; for Prism.Inference it is the weight block
      the row lives inside, so Prism.Streaming can invalidate a whole
      evicted cluster in one call. }
    Owner: Pointer;
  end;

  TPrismVulkan = class
  private
    FApi: TVulkanApi;
    FInstance: TVkInstance;
    FPhysical: TVkPhysicalDevice;
    FDevice: TVkDevice;
    FQueue: TVkQueue;
    FQueueFamily: UInt32;
    FProps: TVkPhysicalDeviceProperties;
    FMemProps: TVkPhysicalDeviceMemoryProperties;

    FModules: array [0 .. 14] of TVkShaderModule;
    FPipelines: array [0 .. 14] of TVkPipeline;
    FSetLayoutW, FSetLayoutXY: TVkDescriptorSetLayout;
    FPipeLayout: TVkPipelineLayout;
    FDescPool: TVkDescriptorPool;
    FXYSet: TVkDescriptorSet;

    FCmdPool: TVkCommandPool;
    FCmd: TVkCommandBuffer;
    FFence: TVkFence;

    FDevX, FDevY: TVkBuffer;
    FDevXMem, FDevYMem: TVkDeviceMemory;
    FHostX, FHostY: TVkBuffer;
    FHostXMem, FHostYMem: TVkDeviceMemory;
    FHostXPtr, FHostYPtr: PSingle;
    FXFloats, FYFloats: Integer;

    FStage: TVkBuffer;
    FStageMem: TVkDeviceMemory;
    FStagePtr: PByte;

    FResident: TDictionary<Pointer, TResident>;
    FLock: TCriticalSection;

    FReady, FBroken: Boolean;
    FBudget, FUsed, FVram: Int64;
    FMaxBufferBytes: Int64;
    FDeviceName: string;
    FDiscrete: Boolean;
    FDispatches, FFallbacks, FRejects: Int64;
    FLog: TProc<string>;

    procedure Note(const S: string);
    function FindMemoryType(TypeBits: UInt32; Want: TVkFlags;
      out Index: UInt32): Boolean;
    function MakeBuffer(Bytes: Int64; Usage, MemWant: TVkFlags;
      out Buf: TVkBuffer; out Mem: TVkDeviceMemory): Boolean;
    procedure DropBuffer(var Buf: TVkBuffer; var Mem: TVkDeviceMemory);

    function CreateInstanceObj(out Why: string): Boolean;
    function PickDevice(out Why: string): Boolean;
    function CreateDeviceObj(out Why: string): Boolean;
    function CreatePipelines(out Why: string): Boolean;
    function CreatePools(out Why: string): Boolean;
    function CreateIoBuffers(XFloats, YFloats: Integer;
      out Why: string): Boolean;
    procedure WriteXYSet;
    function EnsureIoCapacity(Cols, Rows: Integer): Boolean;

    function SubmitAndWait: Boolean;
    function UploadRaw(Src: PByte; Bytes: Int64; Typ: TGgmlType;
      Rows, Cols: Integer; Owner: Pointer; out R: TResident): Boolean;
    procedure FreeResident(const R: TResident);
  public
    constructor Create;
    destructor Destroy; override;
    function Init(BudgetMB: Integer; const ALog: TProc<string>;
      out Info: string): Boolean;
    { The one real entry point. Src is the first weight byte and doubles as
      the residency key, so a GGUF TQTensor and a raw F32 weight block from
      Prism.Inference share the same cache without knowing about each other. }
    function MatVecRaw(Src: PByte; Typ: TGgmlType; Rows, Cols: Integer;
      RowBytes: Int64; Owner: Pointer; Y, X: PSingle): Boolean;
    function MatVec(const T: TQTensor; Y, X: PSingle): Boolean;
    function SelfTest(out Report: string): Boolean;
    procedure Evict(Key: Pointer);
    procedure EvictByOwner(Owner: Pointer);
    procedure EvictAll;
    function Stats: TVulkanStats;
  end;

var
  GVk: TPrismVulkan = nil;

{ ---------- helpers ---------- }

function FloatToHalf(V: Single): Word;
var
  U, S, M: UInt32;
  E: Integer;
begin
  U := PUInt32(@V)^;
  S := (U shr 31) and 1;
  E := Integer((U shr 23) and $FF) - 127 + 15;
  M := U and $7FFFFF;
  if E <= 0 then
    Result := Word(S shl 15)                    // underflow -> signed zero
  else if E >= 31 then
    Result := Word((S shl 15) or (UInt32($1F) shl 10))
  else
    Result := Word((S shl 15) or (UInt32(E) shl 10) or (M shr 13));
end;

function TVulkanStats.Describe: string;
begin
  if DeviceName = '' then
    Exit('no Vulkan device');
  Result := DeviceName;
  if Discrete then
    Result := Result + ' (discrete'
  else
    Result := Result + ' (integrated';
  Result := Result + Format(', %.1f GB VRAM)', [VramBytes / (1024 * 1024 * 1024)]);
  Result := Result + Format('  resident %d tensors / %.0f MB of %.0f MB budget',
    [ResidentTensors, ResidentBytes / (1024 * 1024),
     BudgetBytes / (1024 * 1024)]);
  if Dispatches > 0 then
    Result := Result + Format('  dispatches %d, CPU fallbacks %d',
      [Dispatches, CpuFallbacks]);
  if Broken then
    Result := Result + '  [DISABLED after a device error]';
end;

{ ---------- TPrismVulkan ---------- }

constructor TPrismVulkan.Create;
begin
  inherited Create;
  FLock := TCriticalSection.Create;
  FResident := TDictionary<Pointer, TResident>.Create;
end;

destructor TPrismVulkan.Destroy;
var
  I: Integer;
  R: TResident;
begin
  if Assigned(FApi.DeviceWaitIdle) and (FDevice <> nil) then
    FApi.DeviceWaitIdle(FDevice);

  if (FDevice <> nil) and Assigned(FApi.DestroyBuffer) then
  begin
    for R in FResident.Values do
      FreeResident(R);
    FResident.Clear;

    if FHostXPtr <> nil then FApi.UnmapMemory(FDevice, FHostXMem);
    if FHostYPtr <> nil then FApi.UnmapMemory(FDevice, FHostYMem);
    if FStagePtr <> nil then FApi.UnmapMemory(FDevice, FStageMem);
    DropBuffer(FHostX, FHostXMem);
    DropBuffer(FHostY, FHostYMem);
    DropBuffer(FDevX, FDevXMem);
    DropBuffer(FDevY, FDevYMem);
    DropBuffer(FStage, FStageMem);

    if FFence <> 0 then FApi.DestroyFence(FDevice, FFence, nil);
    if FCmdPool <> 0 then FApi.DestroyCommandPool(FDevice, FCmdPool, nil);
    if FDescPool <> 0 then FApi.DestroyDescriptorPool(FDevice, FDescPool, nil);

    for I := Low(FPipelines) to High(FPipelines) do
    begin
      if FPipelines[I] <> 0 then FApi.DestroyPipeline(FDevice, FPipelines[I], nil);
      if FModules[I] <> 0 then FApi.DestroyShaderModule(FDevice, FModules[I], nil);
    end;
    if FPipeLayout <> 0 then FApi.DestroyPipelineLayout(FDevice, FPipeLayout, nil);
    if FSetLayoutW <> 0 then
      FApi.DestroyDescriptorSetLayout(FDevice, FSetLayoutW, nil);
    if FSetLayoutXY <> 0 then
      FApi.DestroyDescriptorSetLayout(FDevice, FSetLayoutXY, nil);

    FApi.DestroyDevice(FDevice, nil);
    FDevice := nil;
  end;

  if (FInstance <> nil) and Assigned(FApi.DestroyInstance) then
    FApi.DestroyInstance(FInstance, nil);
  FInstance := nil;

  FResident.Free;
  FLock.Free;
  inherited;
end;

procedure TPrismVulkan.Note(const S: string);
begin
  if Assigned(FLog) then
    FLog(S);
end;

function TPrismVulkan.FindMemoryType(TypeBits: UInt32; Want: TVkFlags;
  out Index: UInt32): Boolean;
var
  I: UInt32;
begin
  for I := 0 to FMemProps.memoryTypeCount - 1 do
    if ((TypeBits and (UInt32(1) shl I)) <> 0) and
       ((FMemProps.memoryTypes[I].propertyFlags and Want) = Want) then
    begin
      Index := I;
      Exit(True);
    end;
  Index := 0;
  Result := False;
end;

function TPrismVulkan.MakeBuffer(Bytes: Int64; Usage, MemWant: TVkFlags;
  out Buf: TVkBuffer; out Mem: TVkDeviceMemory): Boolean;
var
  BCI: TVkBufferCreateInfo;
  Req: TVkMemoryRequirements;
  MAI: TVkMemoryAllocateInfo;
  MemIdx: UInt32;
begin
  Buf := 0;
  Mem := 0;
  Result := False;

  FillChar(BCI, SizeOf(BCI), 0);
  BCI.sType := VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  BCI.size := Bytes;
  BCI.usage := Usage;
  BCI.sharingMode := VK_SHARING_MODE_EXCLUSIVE;
  if FApi.CreateBuffer(FDevice, BCI, nil, @Buf) <> VK_SUCCESS then
    Exit;

  FillChar(Req, SizeOf(Req), 0);
  FApi.GetBufferMemoryRequirements(FDevice, Buf, @Req);
  if not FindMemoryType(Req.memoryTypeBits, MemWant, MemIdx) then
  begin
    FApi.DestroyBuffer(FDevice, Buf, nil);
    Buf := 0;
    Exit;
  end;

  FillChar(MAI, SizeOf(MAI), 0);
  MAI.sType := VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  MAI.allocationSize := Req.size;
  MAI.memoryTypeIndex := MemIdx;
  if FApi.AllocateMemory(FDevice, MAI, nil, @Mem) <> VK_SUCCESS then
  begin
    FApi.DestroyBuffer(FDevice, Buf, nil);
    Buf := 0;
    Mem := 0;
    Exit;
  end;

  if FApi.BindBufferMemory(FDevice, Buf, Mem, 0) <> VK_SUCCESS then
  begin
    FApi.FreeMemory(FDevice, Mem, nil);
    FApi.DestroyBuffer(FDevice, Buf, nil);
    Buf := 0;
    Mem := 0;
    Exit;
  end;
  Result := True;
end;

procedure TPrismVulkan.DropBuffer(var Buf: TVkBuffer; var Mem: TVkDeviceMemory);
begin
  if Buf <> 0 then
  begin
    FApi.DestroyBuffer(FDevice, Buf, nil);
    Buf := 0;
  end;
  if Mem <> 0 then
  begin
    FApi.FreeMemory(FDevice, Mem, nil);
    Mem := 0;
  end;
end;

function TPrismVulkan.CreateInstanceObj(out Why: string): Boolean;
var
  AI: TVkApplicationInfo;
  ICI: TVkInstanceCreateInfo;
  Res: TVkResult;
  ExtName: PAnsiChar;
begin
  FillChar(AI, SizeOf(AI), 0);
  AI.sType := VK_STRUCTURE_TYPE_APPLICATION_INFO;
  AI.pApplicationName := 'Prism';
  AI.applicationVersion := VK_MAKE_VERSION(1, 0, 0);
  AI.pEngineName := 'Prism';
  AI.engineVersion := VK_MAKE_VERSION(1, 0, 0);
  AI.apiVersion := VK_MAKE_VERSION(1, 0, 0);

  FillChar(ICI, SizeOf(ICI), 0);
  ICI.sType := VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ICI.pApplicationInfo := @AI;

  Res := FApi.CreateInstance(ICI, nil, @FInstance);

  { On macOS/iOS the loader hides MoltenVK unless portability enumeration is
    explicitly asked for. Retry once with the flag before giving up. }
  if Res = VK_ERROR_INCOMPATIBLE_DRIVER then
  begin
    ExtName := 'VK_KHR_portability_enumeration';
    ICI.flags := VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    ICI.enabledExtensionCount := 1;
    ICI.ppEnabledExtensionNames := @ExtName;
    Res := FApi.CreateInstance(ICI, nil, @FInstance);
  end;

  if Res <> VK_SUCCESS then
  begin
    Why := 'vkCreateInstance: ' + VkResultStr(Res);
    Exit(False);
  end;
  Result := VkLoadInstanceProcs(FApi, FInstance, Why);
end;

function TPrismVulkan.PickDevice(out Why: string): Boolean;
var
  Count, I, J: UInt32;
  Devices: TArray<TVkPhysicalDevice>;
  P: TVkPhysicalDeviceProperties;
  MP: TVkPhysicalDeviceMemoryProperties;
  QCount: UInt32;
  Fams: TArray<TVkQueueFamilyProperties>;
  Score, BestScore: Int64;
  BestIdx, BestFam: Integer;
  Vram: Int64;
  Want: string;
  NameStr: string;
begin
  Result := False;
  Count := 0;
  if (FApi.EnumeratePhysicalDevices(FInstance, @Count, nil) <> VK_SUCCESS) or
     (Count = 0) then
  begin
    Why := 'no Vulkan physical device';
    Exit;
  end;
  SetLength(Devices, Count);
  if FApi.EnumeratePhysicalDevices(FInstance, @Count, @Devices[0]) <> VK_SUCCESS then
  begin
    Why := 'vkEnumeratePhysicalDevices failed';
    Exit;
  end;

  Want := Trim(GetEnvironmentVariable('PRISM_VK_DEVICE'));
  BestScore := -1;
  BestIdx := -1;
  BestFam := -1;

  for I := 0 to Count - 1 do
  begin
    FillChar(P, SizeOf(P), 0);
    FApi.GetPhysicalDeviceProperties(Devices[I], @P);
    NameStr := string(AnsiString(PAnsiChar(@P.deviceName[0])));

    { a compute-capable queue family is mandatory }
    QCount := 0;
    FApi.GetPhysicalDeviceQueueFamilyProperties(Devices[I], @QCount, nil);
    if QCount = 0 then
      Continue;
    SetLength(Fams, QCount);
    FApi.GetPhysicalDeviceQueueFamilyProperties(Devices[I], @QCount, @Fams[0]);
    J := 0;
    while (J < QCount) and
          ((Fams[J].queueFlags and VK_QUEUE_COMPUTE_BIT) = 0) do
      Inc(J);
    if J >= QCount then
      Continue;

    FillChar(MP, SizeOf(MP), 0);
    FApi.GetPhysicalDeviceMemoryProperties(Devices[I], @MP);
    Vram := 0;
    for var H := 0 to Integer(MP.memoryHeapCount) - 1 do
      if (MP.memoryHeaps[H].flags and VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) <> 0 then
        if Int64(MP.memoryHeaps[H].size) > Vram then
          Vram := Int64(MP.memoryHeaps[H].size);

    { discrete beats integrated beats anything else; VRAM breaks ties }
    case P.deviceType of
      VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   Score := 4000;
      VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: Score := 2000;
      VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    Score := 1000;
      VK_PHYSICAL_DEVICE_TYPE_CPU:            Score := 10;
    else
      Score := 500;
    end;
    Inc(Score, Vram div (1024 * 1024 * 1024));

    if (Want <> '') and
       (SameText(Want, IntToStr(I)) or
        ContainsText(NameStr, Want)) then
      Inc(Score, 1000000);   // explicit user choice wins outright

    Note(Format('  Vulkan device %d: %s (type %d, %.1f GB) score %d',
      [I, NameStr, P.deviceType, Vram / (1024 * 1024 * 1024), Score]));

    if Score > BestScore then
    begin
      BestScore := Score;
      BestIdx := Integer(I);
      BestFam := Integer(J);
    end;
  end;

  if BestIdx < 0 then
  begin
    Why := 'no device with a compute queue';
    Exit;
  end;

  FPhysical := Devices[BestIdx];
  FQueueFamily := UInt32(BestFam);
  FillChar(FProps, SizeOf(FProps), 0);
  FApi.GetPhysicalDeviceProperties(FPhysical, @FProps);
  FillChar(FMemProps, SizeOf(FMemProps), 0);
  FApi.GetPhysicalDeviceMemoryProperties(FPhysical, @FMemProps);

  FDeviceName := string(AnsiString(PAnsiChar(@FProps.deviceName[0])));
  FDiscrete := FProps.deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
  FMaxBufferBytes := Int64(FProps.limits.maxStorageBufferRange);
  FVram := 0;
  for I := 0 to FMemProps.memoryHeapCount - 1 do
    if (FMemProps.memoryHeaps[I].flags and VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) <> 0 then
      if Int64(FMemProps.memoryHeaps[I].size) > FVram then
        FVram := Int64(FMemProps.memoryHeaps[I].size);
  Result := True;
end;

function TPrismVulkan.CreateDeviceObj(out Why: string): Boolean;
var
  QCI: TVkDeviceQueueCreateInfo;
  DCI: TVkDeviceCreateInfo;
  Prio: Single;
  Res: TVkResult;
  ExtCount: UInt32;
  Exts: TArray<TVkExtensionProperties>;
  I: Integer;
  PortName: PAnsiChar;
  WantPortability: Boolean;
begin
  Result := False;
  Prio := 1.0;

  FillChar(QCI, SizeOf(QCI), 0);
  QCI.sType := VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  QCI.queueFamilyIndex := FQueueFamily;
  QCI.queueCount := 1;
  QCI.pQueuePriorities := @Prio;

  { VK_KHR_portability_subset must be enabled if the driver advertises it
    (MoltenVK); on every other driver it is simply absent. }
  WantPortability := False;
  ExtCount := 0;
  if FApi.EnumerateDeviceExtensionProperties(FPhysical, nil, @ExtCount, nil)
     = VK_SUCCESS then
    if ExtCount > 0 then
    begin
      SetLength(Exts, ExtCount);
      if FApi.EnumerateDeviceExtensionProperties(FPhysical, nil, @ExtCount,
        @Exts[0]) = VK_SUCCESS then
        for I := 0 to Integer(ExtCount) - 1 do
          if string(AnsiString(PAnsiChar(@Exts[I].extensionName[0])))
             = 'VK_KHR_portability_subset' then
            WantPortability := True;
    end;

  FillChar(DCI, SizeOf(DCI), 0);
  DCI.sType := VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  DCI.queueCreateInfoCount := 1;
  DCI.pQueueCreateInfos := @QCI;
  if WantPortability then
  begin
    PortName := 'VK_KHR_portability_subset';
    DCI.enabledExtensionCount := 1;
    DCI.ppEnabledExtensionNames := @PortName;
  end;

  Res := FApi.CreateDevice(FPhysical, DCI, nil, @FDevice);
  if Res <> VK_SUCCESS then
  begin
    Why := 'vkCreateDevice: ' + VkResultStr(Res);
    Exit;
  end;
  if not VkLoadDeviceProcs(FApi, FDevice, Why) then
    Exit;
  FApi.GetDeviceQueue(FDevice, FQueueFamily, 0, @FQueue);
  Result := FQueue <> nil;
  if not Result then
    Why := 'vkGetDeviceQueue returned nil';
end;

function TPrismVulkan.CreatePipelines(out Why: string): Boolean;
var
  BindW: TVkDescriptorSetLayoutBinding;
  BindXY: array [0 .. 1] of TVkDescriptorSetLayoutBinding;
  DSLCI: TVkDescriptorSetLayoutCreateInfo;
  PCR: TVkPushConstantRange;
  PLCI: TVkPipelineLayoutCreateInfo;
  Layouts: array [0 .. 1] of TVkDescriptorSetLayout;
  I: Integer;

  function MakePipe(Idx: Integer; Code: PUInt32; Words: Integer): Boolean;
  var
    SMCI: TVkShaderModuleCreateInfo;
    CPCI: TVkComputePipelineCreateInfo;
  begin
    FillChar(SMCI, SizeOf(SMCI), 0);
    SMCI.sType := VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    SMCI.codeSize := NativeUInt(Words) * SizeOf(UInt32);
    SMCI.pCode := Code;
    if FApi.CreateShaderModule(FDevice, SMCI, nil, @FModules[Idx])
       <> VK_SUCCESS then
      Exit(False);

    FillChar(CPCI, SizeOf(CPCI), 0);
    CPCI.sType := VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    CPCI.stage.sType := VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    CPCI.stage.stage := VK_SHADER_STAGE_COMPUTE_BIT;
    CPCI.stage.module := FModules[Idx];
    CPCI.stage.pName := 'main';
    CPCI.layout := FPipeLayout;
    Result := FApi.CreateComputePipelines(FDevice, 0, 1, @CPCI, nil,
      @FPipelines[Idx]) = VK_SUCCESS;
  end;

begin
  Result := False;

  FillChar(BindW, SizeOf(BindW), 0);
  BindW.binding := 0;
  BindW.descriptorType := VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  BindW.descriptorCount := 1;
  BindW.stageFlags := VK_SHADER_STAGE_COMPUTE_BIT;

  FillChar(DSLCI, SizeOf(DSLCI), 0);
  DSLCI.sType := VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  DSLCI.bindingCount := 1;
  DSLCI.pBindings := @BindW;
  if FApi.CreateDescriptorSetLayout(FDevice, DSLCI, nil, @FSetLayoutW)
     <> VK_SUCCESS then
  begin
    Why := 'descriptor set layout (weights) failed';
    Exit;
  end;

  FillChar(BindXY, SizeOf(BindXY), 0);
  for I := 0 to 1 do
  begin
    BindXY[I].binding := UInt32(I);
    BindXY[I].descriptorType := VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    BindXY[I].descriptorCount := 1;
    BindXY[I].stageFlags := VK_SHADER_STAGE_COMPUTE_BIT;
  end;
  DSLCI.bindingCount := 2;
  DSLCI.pBindings := @BindXY[0];
  if FApi.CreateDescriptorSetLayout(FDevice, DSLCI, nil, @FSetLayoutXY)
     <> VK_SUCCESS then
  begin
    Why := 'descriptor set layout (x/y) failed';
    Exit;
  end;

  PCR.stageFlags := VK_SHADER_STAGE_COMPUTE_BIT;
  PCR.offset := 0;
  PCR.size := SizeOf(TMatVecPush);
  if PCR.size > FProps.limits.maxPushConstantsSize then
  begin
    Why := 'push constant block too large for this device';
    Exit;
  end;

  Layouts[0] := FSetLayoutW;
  Layouts[1] := FSetLayoutXY;
  FillChar(PLCI, SizeOf(PLCI), 0);
  PLCI.sType := VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  PLCI.setLayoutCount := 2;
  PLCI.pSetLayouts := @Layouts[0];
  PLCI.pushConstantRangeCount := 1;
  PLCI.pPushConstantRanges := @PCR;
  if FApi.CreatePipelineLayout(FDevice, PLCI, nil, @FPipeLayout)
     <> VK_SUCCESS then
  begin
    Why := 'pipeline layout failed';
    Exit;
  end;

  if not (MakePipe(Ord(gtF32),  @SPV_MATVEC_F32[0],  Length(SPV_MATVEC_F32))  and
          MakePipe(Ord(gtF16),  @SPV_MATVEC_F16[0],  Length(SPV_MATVEC_F16))  and
          MakePipe(Ord(gtQ4_0), @SPV_MATVEC_Q4_0[0], Length(SPV_MATVEC_Q4_0)) and
          MakePipe(Ord(gtQ4_1), @SPV_MATVEC_Q4_1[0], Length(SPV_MATVEC_Q4_1)) and
          MakePipe(Ord(gtQ8_0), @SPV_MATVEC_Q8_0[0], Length(SPV_MATVEC_Q8_0)) and
          MakePipe(Ord(gtQ4_K), @SPV_MATVEC_Q4_K[0], Length(SPV_MATVEC_Q4_K)) and
          MakePipe(Ord(gtQ5_K), @SPV_MATVEC_Q5_K[0], Length(SPV_MATVEC_Q5_K)) and
          MakePipe(Ord(gtQ6_K), @SPV_MATVEC_Q6_K[0], Length(SPV_MATVEC_Q6_K))) then
  begin
    Why := 'compute pipeline creation failed';
    Exit;
  end;
  Result := True;
end;

function TPrismVulkan.CreatePools(out Why: string): Boolean;
var
  PS: array [0 .. 0] of TVkDescriptorPoolSize;
  DPCI: TVkDescriptorPoolCreateInfo;
  DSAI: TVkDescriptorSetAllocateInfo;
  CPCI: TVkCommandPoolCreateInfo;
  CBAI: TVkCommandBufferAllocateInfo;
  FCI: TVkFenceCreateInfo;
  Lay: TVkDescriptorSetLayout;
begin
  Result := False;

  PS[0].typ := VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  PS[0].descriptorCount := MAX_RESIDENT_TENSORS + 2;

  FillChar(DPCI, SizeOf(DPCI), 0);
  DPCI.sType := VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  DPCI.flags := VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  DPCI.maxSets := MAX_RESIDENT_TENSORS + 1;
  DPCI.poolSizeCount := 1;
  DPCI.pPoolSizes := @PS[0];
  if FApi.CreateDescriptorPool(FDevice, DPCI, nil, @FDescPool) <> VK_SUCCESS then
  begin
    Why := 'descriptor pool failed';
    Exit;
  end;

  Lay := FSetLayoutXY;
  FillChar(DSAI, SizeOf(DSAI), 0);
  DSAI.sType := VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  DSAI.descriptorPool := FDescPool;
  DSAI.descriptorSetCount := 1;
  DSAI.pSetLayouts := @Lay;
  if FApi.AllocateDescriptorSets(FDevice, DSAI, @FXYSet) <> VK_SUCCESS then
  begin
    Why := 'x/y descriptor set allocation failed';
    Exit;
  end;

  FillChar(CPCI, SizeOf(CPCI), 0);
  CPCI.sType := VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  CPCI.flags := VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  CPCI.queueFamilyIndex := FQueueFamily;
  if FApi.CreateCommandPool(FDevice, CPCI, nil, @FCmdPool) <> VK_SUCCESS then
  begin
    Why := 'command pool failed';
    Exit;
  end;

  FillChar(CBAI, SizeOf(CBAI), 0);
  CBAI.sType := VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  CBAI.commandPool := FCmdPool;
  CBAI.level := VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  CBAI.commandBufferCount := 1;
  if FApi.AllocateCommandBuffers(FDevice, CBAI, @FCmd) <> VK_SUCCESS then
  begin
    Why := 'command buffer allocation failed';
    Exit;
  end;

  FillChar(FCI, SizeOf(FCI), 0);
  FCI.sType := VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  if FApi.CreateFence(FDevice, FCI, nil, @FFence) <> VK_SUCCESS then
  begin
    Why := 'fence creation failed';
    Exit;
  end;
  Result := True;
end;

function TPrismVulkan.CreateIoBuffers(XFloats, YFloats: Integer;
  out Why: string): Boolean;
var
  XB, YB: Int64;
begin
  Result := False;
  XB := Int64(XFloats) * SizeOf(Single);
  YB := Int64(YFloats) * SizeOf(Single);

  { Activations live in DEVICE_LOCAL memory: every workgroup re-reads the
    whole x vector, and on a discrete GPU a host-visible x would drag those
    reads across PCIe. The small staged copy is much cheaper. }
  if not MakeBuffer(XB,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, FDevX, FDevXMem) then
  begin
    Why := 'device x buffer allocation failed';
    Exit;
  end;
  if not MakeBuffer(YB,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, FDevY, FDevYMem) then
  begin
    Why := 'device y buffer allocation failed';
    Exit;
  end;
  if not MakeBuffer(XB, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    FHostX, FHostXMem) then
  begin
    Why := 'host x buffer allocation failed';
    Exit;
  end;
  if not MakeBuffer(YB, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    FHostY, FHostYMem) then
  begin
    Why := 'host y buffer allocation failed';
    Exit;
  end;

  if FApi.MapMemory(FDevice, FHostXMem, 0, VK_WHOLE_SIZE, 0,
    @FHostXPtr) <> VK_SUCCESS then
  begin
    Why := 'mapping host x failed';
    Exit;
  end;
  if FApi.MapMemory(FDevice, FHostYMem, 0, VK_WHOLE_SIZE, 0,
    @FHostYPtr) <> VK_SUCCESS then
  begin
    Why := 'mapping host y failed';
    Exit;
  end;

  FXFloats := XFloats;
  FYFloats := YFloats;
  WriteXYSet;
  Result := True;
end;

procedure TPrismVulkan.WriteXYSet;
var
  BI: array [0 .. 1] of TVkDescriptorBufferInfo;
  W: array [0 .. 1] of TVkWriteDescriptorSet;
  I: Integer;
begin
  BI[0].buffer := FDevX;
  BI[0].offset := 0;
  BI[0].range := VK_WHOLE_SIZE;
  BI[1].buffer := FDevY;
  BI[1].offset := 0;
  BI[1].range := VK_WHOLE_SIZE;

  FillChar(W, SizeOf(W), 0);
  for I := 0 to 1 do
  begin
    W[I].sType := VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    W[I].dstSet := FXYSet;
    W[I].dstBinding := UInt32(I);
    W[I].descriptorCount := 1;
    W[I].descriptorType := VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    W[I].pBufferInfo := @BI[I];
  end;
  FApi.UpdateDescriptorSets(FDevice, 2, @W[0], 0, nil);
end;

function TPrismVulkan.EnsureIoCapacity(Cols, Rows: Integer): Boolean;
var
  NewX, NewY: Integer;
  Why: string;
begin
  if (Cols <= FXFloats) and (Rows <= FYFloats) then
    Exit(True);

  NewX := Max(FXFloats, Cols);
  NewY := Max(FYFloats, Rows);

  FApi.DeviceWaitIdle(FDevice);
  if FHostXPtr <> nil then
  begin
    FApi.UnmapMemory(FDevice, FHostXMem);
    FHostXPtr := nil;
  end;
  if FHostYPtr <> nil then
  begin
    FApi.UnmapMemory(FDevice, FHostYMem);
    FHostYPtr := nil;
  end;
  DropBuffer(FHostX, FHostXMem);
  DropBuffer(FHostY, FHostYMem);
  DropBuffer(FDevX, FDevXMem);
  DropBuffer(FDevY, FDevYMem);

  Result := CreateIoBuffers(NewX, NewY, Why);
  if not Result then
  begin
    Note('Vulkan: growing the x/y buffers failed (' + Why +
      ') - switching to CPU');
    FBroken := True;
  end;
end;

function TPrismVulkan.SubmitAndWait: Boolean;
var
  SI: TVkSubmitInfo;
  Res: TVkResult;
begin
  FillChar(SI, SizeOf(SI), 0);
  SI.sType := VK_STRUCTURE_TYPE_SUBMIT_INFO;
  SI.commandBufferCount := 1;
  SI.pCommandBuffers := @FCmd;

  FApi.ResetFences(FDevice, 1, @FFence);
  Res := FApi.QueueSubmit(FQueue, 1, @SI, FFence);
  if Res <> VK_SUCCESS then
  begin
    Note('Vulkan: vkQueueSubmit failed (' + VkResultStr(Res) +
      ') - switching to CPU');
    FBroken := True;
    Exit(False);
  end;

  Res := FApi.WaitForFences(FDevice, 1, @FFence, 1, FENCE_TIMEOUT_NS);
  if Res <> VK_SUCCESS then
  begin
    { A timeout here means the device is wedged or the driver died. Never
      retry -- fall back to the CPU for the rest of the process. }
    Note('Vulkan: fence wait failed (' + VkResultStr(Res) +
      ') - switching to CPU');
    FBroken := True;
    Exit(False);
  end;
  Result := True;
end;

function TPrismVulkan.UploadRaw(Src: PByte; Bytes: Int64; Typ: TGgmlType;
  Rows, Cols: Integer; Owner: Pointer; out R: TResident): Boolean;
var
  Done, Chunk: Int64;
  DSAI: TVkDescriptorSetAllocateInfo;
  Lay: TVkDescriptorSetLayout;
  BI: TVkDescriptorBufferInfo;
  W: TVkWriteDescriptorSet;
  BBI: TVkCommandBufferBeginInfo;
  Copy: TVkBufferCopy;
begin
  Result := False;
  FillChar(R, SizeOf(R), 0);

  if Bytes + BUF_SLACK > FMaxBufferBytes then
  begin
    Inc(FRejects);
    Exit;   // single tensor exceeds maxStorageBufferRange -> stays on CPU
  end;
  if FUsed + Bytes + BUF_SLACK > FBudget then
  begin
    Inc(FRejects);
    Exit;   // VRAM budget spent -> this and later tensors stay on CPU
  end;
  if FResident.Count >= MAX_RESIDENT_TENSORS then
  begin
    Inc(FRejects);
    Exit;
  end;

  if not MakeBuffer(Bytes + BUF_SLACK,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, R.Buf, R.Mem) then
  begin
    Inc(FRejects);
    Exit;
  end;

  { staged upload, one STAGE_BYTES chunk per submit }
  Done := 0;
  while Done < Bytes do
  begin
    Chunk := Min(Int64(STAGE_BYTES), Bytes - Done);
    Move(Src[Done], FStagePtr^, Chunk);

    FApi.ResetCommandBuffer(FCmd, 0);
    FillChar(BBI, SizeOf(BBI), 0);
    BBI.sType := VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    BBI.flags := VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    FApi.BeginCommandBuffer(FCmd, BBI);
    Copy.srcOffset := 0;
    Copy.dstOffset := Done;
    Copy.size := Chunk;
    FApi.CmdCopyBuffer(FCmd, FStage, R.Buf, 1, @Copy);
    FApi.EndCommandBuffer(FCmd);
    if not SubmitAndWait then
    begin
      DropBuffer(R.Buf, R.Mem);
      Exit;
    end;
    Inc(Done, Chunk);
  end;

  Lay := FSetLayoutW;
  FillChar(DSAI, SizeOf(DSAI), 0);
  DSAI.sType := VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  DSAI.descriptorPool := FDescPool;
  DSAI.descriptorSetCount := 1;
  DSAI.pSetLayouts := @Lay;
  if FApi.AllocateDescriptorSets(FDevice, DSAI, @R.DescSet) <> VK_SUCCESS then
  begin
    DropBuffer(R.Buf, R.Mem);
    Inc(FRejects);
    Exit;
  end;

  BI.buffer := R.Buf;
  BI.offset := 0;
  BI.range := VK_WHOLE_SIZE;
  FillChar(W, SizeOf(W), 0);
  W.sType := VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  W.dstSet := R.DescSet;
  W.dstBinding := 0;
  W.descriptorCount := 1;
  W.descriptorType := VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  W.pBufferInfo := @BI;
  FApi.UpdateDescriptorSets(FDevice, 1, @W, 0, nil);

  R.Bytes := Bytes;
  R.Typ := Typ;
  R.Rows := Rows;
  R.Cols := Cols;
  R.Owner := Owner;
  Inc(FUsed, Bytes);
  Result := True;
end;

procedure TPrismVulkan.FreeResident(const R: TResident);
var
  S: TVkDescriptorSet;
begin
  if R.DescSet <> 0 then
  begin
    S := R.DescSet;
    FApi.FreeDescriptorSets(FDevice, FDescPool, 1, @S);
  end;
  if R.Buf <> 0 then
    FApi.DestroyBuffer(FDevice, R.Buf, nil);
  if R.Mem <> 0 then
    FApi.FreeMemory(FDevice, R.Mem, nil);
end;

procedure TPrismVulkan.Evict(Key: Pointer);
var
  R: TResident;
begin
  if not FReady then
    Exit;
  FLock.Enter;
  try
    if FResident.TryGetValue(Key, R) then
    begin
      FApi.DeviceWaitIdle(FDevice);
      FreeResident(R);
      Dec(FUsed, R.Bytes);
      FResident.Remove(Key);
    end;
  finally
    FLock.Leave;
  end;
end;

procedure TPrismVulkan.EvictByOwner(Owner: Pointer);
var
  Doomed: TArray<Pointer>;
  Pair: TPair<Pointer, TResident>;
  K: Pointer;
  R: TResident;
begin
  if not FReady then
    Exit;
  FLock.Enter;
  try
    Doomed := nil;
    for Pair in FResident do
      if Pair.Value.Owner = Owner then
        Doomed := Doomed + [Pair.Key];
    if Length(Doomed) = 0 then
      Exit;
    FApi.DeviceWaitIdle(FDevice);
    for K in Doomed do
      if FResident.TryGetValue(K, R) then
      begin
        FreeResident(R);
        Dec(FUsed, R.Bytes);
        FResident.Remove(K);
      end;
  finally
    FLock.Leave;
  end;
end;

procedure TPrismVulkan.EvictAll;
var
  R: TResident;
begin
  if not FReady then
    Exit;
  FLock.Enter;
  try
    FApi.DeviceWaitIdle(FDevice);
    for R in FResident.Values do
      FreeResident(R);
    FResident.Clear;
    FUsed := 0;
  finally
    FLock.Leave;
  end;
end;

function TPrismVulkan.MatVecRaw(Src: PByte; Typ: TGgmlType;
  Rows, Cols: Integer; RowBytes: Int64; Owner: Pointer;
  Y, X: PSingle): Boolean;
var
  R: TResident;
  Key: Pointer;
  Push: TMatVecPush;
  BBI: TVkCommandBufferBeginInfo;
  Copy: TVkBufferCopy;
  Bar: TVkBufferMemoryBarrier;
  Sets: array [0 .. 1] of TVkDescriptorSet;
  Base, Groups: UInt32;
  Gran: Integer;
  Bytes: Int64;
begin
  Result := False;
  if (not FReady) or FBroken or (Src = nil) or (Rows <= 0) or (Cols <= 0) then
    Exit;
  if (Ord(Typ) > High(FPipelines)) or (FPipelines[Ord(Typ)] = 0) then
    Exit;

  { The shader walks each row in 32-column units; K-quants additionally need
    whole 256-value super-blocks. Anything else is left to the CPU. }
  if Typ in [gtQ4_K, gtQ5_K, gtQ6_K] then
    Gran := QK_K
  else
    Gran := QK;
  if Cols mod Gran <> 0 then
    Exit;

  { A row offset is a uint in the shader, so the whole tensor must stay
    inside 4 GB -- true for every model we target, checked anyway. }
  Bytes := RowBytes * Rows;
  if (Bytes <= 0) or (Bytes > Int64($FFFFFFFF)) then
    Exit;

  Key := Src;

  FLock.Enter;
  try
    if FBroken then
      Exit(False);
    if not EnsureIoCapacity(Cols, Rows) then
      Exit(False);

    if FResident.TryGetValue(Key, R) then
    begin
      { Freed memory can hand its address to a different tensor later on.
        A shape mismatch is the tell: drop the stale copy and re-upload. }
      if (R.Typ <> Typ) or (R.Rows <> Rows) or (R.Cols <> Cols) or
         (R.Bytes <> Bytes) then
      begin
        FApi.DeviceWaitIdle(FDevice);
        FreeResident(R);
        Dec(FUsed, R.Bytes);
        FResident.Remove(Key);
        if not UploadRaw(Src, Bytes, Typ, Rows, Cols, Owner, R) then
        begin
          Inc(FFallbacks);
          Exit(False);
        end;
        FResident.Add(Key, R);
      end;
    end
    else
    begin
      if not UploadRaw(Src, Bytes, Typ, Rows, Cols, Owner, R) then
      begin
        Inc(FFallbacks);
        Exit(False);
      end;
      FResident.Add(Key, R);
    end;

    Move(X^, FHostXPtr^, Int64(Cols) * SizeOf(Single));

    FApi.ResetCommandBuffer(FCmd, 0);
    FillChar(BBI, SizeOf(BBI), 0);
    BBI.sType := VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    BBI.flags := VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    FApi.BeginCommandBuffer(FCmd, BBI);

    Copy.srcOffset := 0;
    Copy.dstOffset := 0;
    Copy.size := UInt64(Cols) * SizeOf(Single);
    FApi.CmdCopyBuffer(FCmd, FHostX, FDevX, 1, @Copy);

    FillChar(Bar, SizeOf(Bar), 0);
    Bar.sType := VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    Bar.srcAccessMask := VK_ACCESS_TRANSFER_WRITE_BIT;
    Bar.dstAccessMask := VK_ACCESS_SHADER_READ_BIT;
    Bar.srcQueueFamilyIndex := VK_QUEUE_FAMILY_IGNORED;
    Bar.dstQueueFamilyIndex := VK_QUEUE_FAMILY_IGNORED;
    Bar.buffer := FDevX;
    Bar.offset := 0;
    Bar.size := VK_WHOLE_SIZE;
    FApi.CmdPipelineBarrier(FCmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nil, 1, @Bar, 0, nil);

    FApi.CmdBindPipeline(FCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
      FPipelines[Ord(Typ)]);
    Sets[0] := R.DescSet;
    Sets[1] := FXYSet;
    FApi.CmdBindDescriptorSets(FCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
      FPipeLayout, 0, 2, @Sets[0], 0, nil);

    Push.Rows := UInt32(Rows);
    Push.Cols := UInt32(Cols);
    Push.RowBytes := UInt32(RowBytes);
    Push.XOff := 0;
    Push.YOff := 0;

    { >65535 rows (Llama 3's output tensor) needs more than one dispatch }
    Base := 0;
    while Base < UInt32(Rows) do
    begin
      Push.RowBase := Base;
      Groups := UInt32(Rows) - Base;
      if Groups > MAX_GROUPS_X then
        Groups := MAX_GROUPS_X;
      FApi.CmdPushConstants(FCmd, FPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, SizeOf(Push), @Push);
      FApi.CmdDispatch(FCmd, Groups, 1, 1);
      Inc(Base, Groups);
    end;

    Bar.srcAccessMask := VK_ACCESS_SHADER_WRITE_BIT;
    Bar.dstAccessMask := VK_ACCESS_TRANSFER_READ_BIT;
    Bar.buffer := FDevY;
    FApi.CmdPipelineBarrier(FCmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nil, 1, @Bar, 0, nil);

    Copy.srcOffset := 0;
    Copy.dstOffset := 0;
    Copy.size := UInt64(Rows) * SizeOf(Single);
    FApi.CmdCopyBuffer(FCmd, FDevY, FHostY, 1, @Copy);

    FApi.EndCommandBuffer(FCmd);

    if not SubmitAndWait then
    begin
      Inc(FFallbacks);
      Exit(False);
    end;

    Move(FHostYPtr^, Y^, Int64(Rows) * SizeOf(Single));
    Inc(FDispatches);
    Result := True;
  finally
    FLock.Leave;
  end;
end;

function TPrismVulkan.MatVec(const T: TQTensor; Y, X: PSingle): Boolean;
begin
  if T.IsEmpty then
    Exit(False);
  Result := MatVecRaw(PByte(T.Data), T.Typ, T.Rows, T.Cols, T.RowBytes,
    Pointer(T.Data), Y, X);
end;

{ ---------- self-test ----------
  Purpose: prove that the hand-transcribed bit layouts in the shader read the
  same bits as Prism.Vector does. That question has to be separated from a
  second, unrelated one -- how accurately each path sums the products.

  So the comparator is NOT TQTensor.MatVecCpu. For Q4_0/Q4_1/Q8_0 that kernel
  quantizes the ACTIVATION to int8 to get integer MACs (a deliberate CPU
  trade-off), while the shader multiplies in float, because float FMA is free
  on a GPU. Comparing the two conflates a layout bug with expected
  quantization noise, and the noise is the larger of the two.

  Instead we compare against TQTensor.DequantRow + a double-accumulated dot
  product: exact, activation-quantization-free, and a THIRD implementation of
  each layout, so agreement is real evidence rather than two copies of the
  same mistake.

  Tolerances are relative to sum|w_i * x_i|, not to the result. With random
  payload bytes the result is a random walk that can land near zero through
  cancellation; scaling by the result would then turn a rounding difference
  into an apparent 100% error. }

function TPrismVulkan.SelfTest(out Report: string): Boolean;
const
  ROWS = 3;
  COLS = 512;
  TYPES: array [0 .. 7] of TGgmlType =
    (gtF32, gtF16, gtQ4_0, gtQ4_1, gtQ8_0, gtQ4_K, gtQ5_K, gtQ6_K);
  NAMES: array [0 .. 7] of string =
    ('F32', 'F16', 'Q4_0', 'Q4_1', 'Q8_0', 'Q4_K', 'Q5_K', 'Q6_K');
  { A layout error moves a result by O(scale); float32 reduction noise stays
    around 1e-6 * scale. Anything above this is a real disagreement. }
  TOL = 1.0E-4;
var
  Ti, R, I, NB, NSB, B: Integer;
  T: TQTensor;
  X, YGpu, Row: TArray<Single>;
  Ref, Scale: TArray<Double>;
  Acc, Mag: Double;
  P: PByte;
  Seed: UInt32;
  Worst, Rel: Double;
  Bad, Detail: string;
  Ran: Boolean;

  function NextByte: Byte;
  begin
    Seed := Seed * 1664525 + 1013904223;
    Result := Byte((Seed shr 16) and $FF);
  end;

begin
  Report := '';
  Bad := '';
  Detail := '';
  Ran := False;
  Seed := 12345;

  SetLength(X, COLS);
  for I := 0 to COLS - 1 do
    X[I] := Sin(I * 0.37) * 1.7 - 0.4;
  SetLength(YGpu, ROWS);
  SetLength(Row, COLS);
  SetLength(Ref, ROWS);
  SetLength(Scale, ROWS);

  for Ti := 0 to High(TYPES) do
  begin
    T.Typ := TYPES[Ti];
    T.Rows := ROWS;
    T.Cols := COLS;
    SetLength(T.Data, T.TotalBytes);

    { Payload bytes are pseudo-random so every nibble and bit position gets
      exercised; only the f16 scale fields are forced to sane magnitudes so
      nothing turns into NaN or Inf. }
    for R := 0 to ROWS - 1 do
    begin
      P := PByte(T.Data) + Int64(R) * T.RowBytes;
      case T.Typ of
        gtF32:
          for I := 0 to COLS - 1 do
            PSingle(P)[I] := Cos(I * 0.11 + R) * 0.8;
        gtF16:
          for I := 0 to COLS - 1 do
            PWord(P)[I] := FloatToHalf(Cos(I * 0.11 + R) * 0.8);
        gtQ8_0:
          begin
            NB := COLS div QK;
            for B := 0 to NB - 1 do
            begin
              PWord(P + B * 34)^ := FloatToHalf(0.021 + B * 0.001);
              for I := 0 to QK - 1 do
                (P + B * 34 + 2)[I] := NextByte;
            end;
          end;
        gtQ4_0:
          begin
            NB := COLS div QK;
            for B := 0 to NB - 1 do
            begin
              PWord(P + B * 18)^ := FloatToHalf(0.033 + B * 0.002);
              for I := 0 to (QK div 2) - 1 do
                (P + B * 18 + 2)[I] := NextByte;
            end;
          end;
        gtQ4_1:
          begin
            NB := COLS div QK;
            for B := 0 to NB - 1 do
            begin
              PWord(P + B * 20)^ := FloatToHalf(0.017 + B * 0.001);
              PWord(P + B * 20 + 2)^ := FloatToHalf(-0.25 + B * 0.01);
              for I := 0 to (QK div 2) - 1 do
                (P + B * 20 + 4)[I] := NextByte;
            end;
          end;
        gtQ4_K:
          begin
            NSB := COLS div QK_K;
            for B := 0 to NSB - 1 do
            begin
              PWord(P + B * 144)^ := FloatToHalf(0.0012);
              PWord(P + B * 144 + 2)^ := FloatToHalf(0.0007);
              for I := 0 to 139 do
                (P + B * 144 + 4)[I] := NextByte;
            end;
          end;
        gtQ5_K:
          begin
            NSB := COLS div QK_K;
            for B := 0 to NSB - 1 do
            begin
              PWord(P + B * 176)^ := FloatToHalf(0.0009);
              PWord(P + B * 176 + 2)^ := FloatToHalf(0.0005);
              for I := 0 to 171 do
                (P + B * 176 + 4)[I] := NextByte;
            end;
          end;
        gtQ6_K:
          begin
            NSB := COLS div QK_K;
            for B := 0 to NSB - 1 do
            begin
              for I := 0 to 207 do
                (P + B * 210)[I] := NextByte;
              PWord(P + B * 210 + 208)^ := FloatToHalf(0.00045);
            end;
          end;
      end;
    end;

    { exact reference: dequantize the row, then dot in double }
    for R := 0 to ROWS - 1 do
    begin
      T.DequantRow(R, @Row[0]);
      Acc := 0;
      Mag := 0;
      for I := 0 to COLS - 1 do
      begin
        Acc := Acc + Double(Row[I]) * X[I];
        Mag := Mag + Abs(Double(Row[I]) * X[I]);
      end;
      Ref[R] := Acc;
      Scale[R] := Max(Mag, 1.0E-30);
    end;

    FillChar(YGpu[0], ROWS * SizeOf(Single), 0);
    if not MatVec(T, @YGpu[0], @X[0]) then
    begin
      Bad := Bad + NAMES[Ti] + ':not-dispatched ';
      T.Data := nil;
      Continue;
    end;
    Ran := True;

    Worst := 0;
    for R := 0 to ROWS - 1 do
    begin
      Rel := Abs(Ref[R] - YGpu[R]) / Scale[R];
      if Rel > Worst then
        Worst := Rel;
    end;
    Detail := Detail + Format('%s %.1e  ', [NAMES[Ti], Worst]);
    if Worst > TOL then
      Bad := Bad + Format('%s(exact=%.6f gpu=%.6f scale=%.3f) ',
        [NAMES[Ti], Ref[0], YGpu[0], Scale[0]]);

    { the synthetic tensors must not linger in VRAM }
    Evict(Pointer(T.Data));
    T.Data := nil;
  end;

  Result := Ran and (Bad = '');
  if Bad <> '' then
    Report := 'MISMATCH: ' + Bad + '| all: ' + Trim(Detail)
  else if not Ran then
    Report := 'no type could be dispatched'
  else
    Report := 'all 8 layouts match the CPU reference (worst relative error: ' +
      Trim(Detail) + ')';
end;

function TPrismVulkan.Init(BudgetMB: Integer; const ALog: TProc<string>;
  out Info: string): Boolean;
var
  Why, TestReport: string;
begin
  FLog := ALog;
  Result := False;
  Info := '';

  if not VkLoadLoader(FApi, Why) then
  begin
    Info := Why;
    Exit;
  end;
  if not CreateInstanceObj(Why) then
  begin
    Info := Why;
    Exit;
  end;
  if not PickDevice(Why) then
  begin
    Info := Why;
    Exit;
  end;
  if not CreateDeviceObj(Why) then
  begin
    Info := Why;
    Exit;
  end;
  if not CreatePipelines(Why) then
  begin
    Info := Why;
    Exit;
  end;
  if not CreatePools(Why) then
  begin
    Info := Why;
    Exit;
  end;
  if not CreateIoBuffers(DEF_X_FLOATS, DEF_Y_FLOATS, Why) then
  begin
    Info := Why;
    Exit;
  end;
  if not MakeBuffer(STAGE_BYTES, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    FStage, FStageMem) then
  begin
    Info := 'staging buffer allocation failed';
    Exit;
  end;
  if FApi.MapMemory(FDevice, FStageMem, 0, VK_WHOLE_SIZE, 0,
    @FStagePtr) <> VK_SUCCESS then
  begin
    Info := 'mapping the staging buffer failed';
    Exit;
  end;

  if BudgetMB > 0 then
    FBudget := Int64(BudgetMB) * 1024 * 1024
  else
    FBudget := Max(Int64(0),
      Min(Trunc(FVram * VRAM_FRACTION), FVram - VRAM_HEADROOM));
  FUsed := 0;
  FReady := True;

  { Hand-transcribed bit layouts are exactly the kind of thing that is wrong
    but plausible. Refuse to activate rather than produce quiet garbage. }
  if not SelfTest(TestReport) then
  begin
    FReady := False;
    Info := 'self-test failed: ' + TestReport;
    Exit;
  end;

  Note('Vulkan self-test: ' + TestReport);
  Info := Format('%s, %.1f GB VRAM, budget %.0f MB, Vulkan %d.%d.%d',
    [FDeviceName, FVram / (1024 * 1024 * 1024), FBudget / (1024 * 1024),
     VK_VERSION_MAJOR(FProps.apiVersion), VK_VERSION_MINOR(FProps.apiVersion),
     VK_VERSION_PATCH(FProps.apiVersion)]);
  Result := True;
end;

function TPrismVulkan.Stats: TVulkanStats;
begin
  FillChar(Result, SizeOf(Result), 0);
  Result.DeviceName := FDeviceName;
  Result.Discrete := FDiscrete;
  Result.VramBytes := FVram;
  Result.BudgetBytes := FBudget;
  Result.ResidentBytes := FUsed;
  Result.ResidentTensors := FResident.Count;
  Result.Dispatches := FDispatches;
  Result.CpuFallbacks := FFallbacks;
  Result.BudgetRejects := FRejects;
  Result.Broken := FBroken;
end;

{ ---------- module-level facade ---------- }

function VulkanInit(BudgetMB: Integer; const Log: TProc<string>;
  out Info: string): Boolean;
begin
  if GVk <> nil then
  begin
    Info := GVk.FDeviceName;
    Exit(True);
  end;
  GVk := TPrismVulkan.Create;
  Result := GVk.Init(BudgetMB, Log, Info);
  if not Result then
    FreeAndNil(GVk);
end;

procedure VulkanShutdown;
begin
  FreeAndNil(GVk);
end;

function VulkanReady: Boolean;
begin
  Result := (GVk <> nil) and GVk.FReady and (not GVk.FBroken);
end;

function VulkanStats: TVulkanStats;
begin
  if GVk <> nil then
    Result := GVk.Stats
  else
    FillChar(Result, SizeOf(Result), 0);
end;

function VulkanSelfTest(out Report: string): Boolean;
begin
  if GVk = nil then
  begin
    Report := 'Vulkan backend not initialised';
    Exit(False);
  end;
  Result := GVk.SelfTest(Report);
end;

function VulkanMatVec(const T: TQTensor; Y, X: PSingle): Boolean;
begin
  Result := (GVk <> nil) and GVk.MatVec(T, Y, X);
end;

function VulkanMatVecF32(Y, W, X: PSingle; Rows, Cols: Integer;
  Bias: PSingle; Owner: Pointer): Boolean;
var
  I: Integer;
begin
  Result := (GVk <> nil) and GVk.MatVecRaw(PByte(W), gtF32, Rows, Cols,
    Int64(Cols) * SizeOf(Single), Owner, Y, X);
  { The kernel has no bias binding -- adding Rows floats on the host is far
    cheaper than a second descriptor and another upload. }
  if Result and (Bias <> nil) then
    for I := 0 to Rows - 1 do
      Y[I] := Y[I] + Bias[I];
end;

procedure VulkanEvict(const T: TQTensor);
begin
  if (GVk <> nil) and (Pointer(T.Data) <> nil) then
    GVk.Evict(Pointer(T.Data));
end;

procedure VulkanEvictOwner(Owner: Pointer);
begin
  if (GVk <> nil) and (Owner <> nil) then
    GVk.EvictByOwner(Owner);
end;

procedure VulkanEvictAll;
begin
  if GVk <> nil then
    GVk.EvictAll;
end;

initialization
  { nothing to do -- the backend is created on demand by VulkanInit }

finalization
  VulkanShutdown;
  VkUnloadLoader;

end.

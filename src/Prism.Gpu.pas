unit Prism.Gpu;

{ Experimental GPU backend via OpenCL - WITHOUT third-party libraries:
  the system's/driver's OpenCL library is loaded dynamically at runtime
  (Windows: OpenCL.dll, Linux/Android: libOpenCL.so,
  macOS: OpenCL.framework). If no OpenCL is available (e.g. iOS),
  the CPU backend runs transparently.

  Accelerated are the large F32 MatVec operations (Prism's own models).
  Weight rows are uploaded once into GPU buffers and cached; the
  streaming layer reports evictions via InvalidateWeights.
  Quantized GGUF kernels currently run on the CPU (roadmap). }

{$POINTERMATH ON}

interface

uses
  System.SysUtils, System.Classes, System.SyncObjs,
  System.Generics.Collections, Prism.Vector;

type
  TComputeBackend = class
  public
    function Name: string; virtual;
    procedure MatVecF32W(Y, W, X: PSingle; Rows, Cols: Integer;
      Bias: PSingle; WKey: Pointer; WKeyOff: Int64); virtual;
    procedure InvalidateWeights(WKey: Pointer); virtual;
  end;

function Backend: TComputeBackend;
function TryInitGpuBackend(out Info: string): Boolean;
procedure ShutdownGpuBackend;

implementation

{$IFDEF MSWINDOWS}
uses Winapi.Windows;
{$ELSE}
uses Posix.Dlfcn;
{$ENDIF}

type
  { Winapi.Windows declares its own PSingle - disambiguate it here }
  PSingle = System.PSingle;

  TCpuBackend = class(TComputeBackend);

function TComputeBackend.Name: string;
begin
  Result := 'CPU (' + IntToStr(TThread.ProcessorCount) + ' threads)';
end;

procedure TComputeBackend.MatVecF32W(Y, W, X: PSingle; Rows, Cols: Integer;
  Bias: PSingle; WKey: Pointer; WKeyOff: Int64);
begin
  MatVecF32(Y, W, X, Rows, Cols, Bias);
end;

procedure TComputeBackend.InvalidateWeights(WKey: Pointer);
begin
end;

{ ---------- OpenCL (dynamically loaded) ---------- }

const
  CL_DEVICE_TYPE_GPU = 4;
  CL_MEM_READ_ONLY = 4;
  CL_MEM_WRITE_ONLY = 2;
  CL_MEM_COPY_HOST_PTR = 32;
  CL_DEVICE_NAME = $102B;

type
  TCLInt = Int32;
  TCLHandle = Pointer;
  PCLHandle = ^TCLHandle;

  TclGetPlatformIDs = function(NumEntries: UInt32; Platforms: PCLHandle;
    NumPlatforms: PUInt32): TCLInt; {$IFDEF MSWINDOWS} stdcall; {$ELSE} cdecl; {$ENDIF}
  TclGetDeviceIDs = function(Platform_: TCLHandle; DeviceType: UInt64;
    NumEntries: UInt32; Devices: PCLHandle; NumDevices: PUInt32): TCLInt;
    {$IFDEF MSWINDOWS} stdcall; {$ELSE} cdecl; {$ENDIF}
  TclGetDeviceInfo = function(Device: TCLHandle; ParamName: UInt32;
    ValueSize: NativeUInt; Value: Pointer; SizeRet: PNativeUInt): TCLInt;
    {$IFDEF MSWINDOWS} stdcall; {$ELSE} cdecl; {$ENDIF}
  TclCreateContext = function(Properties: Pointer; NumDevices: UInt32;
    Devices: PCLHandle; Notify, UserData: Pointer; ErrRet: PInteger): TCLHandle;
    {$IFDEF MSWINDOWS} stdcall; {$ELSE} cdecl; {$ENDIF}
  TclCreateCommandQueue = function(Context, Device: TCLHandle;
    Properties: UInt64; ErrRet: PInteger): TCLHandle;
    {$IFDEF MSWINDOWS} stdcall; {$ELSE} cdecl; {$ENDIF}
  TclCreateBuffer = function(Context: TCLHandle; Flags: UInt64;
    Size: NativeUInt; HostPtr: Pointer; ErrRet: PInteger): TCLHandle;
    {$IFDEF MSWINDOWS} stdcall; {$ELSE} cdecl; {$ENDIF}
  TclCreateProgramWithSource = function(Context: TCLHandle; Count: UInt32;
    Strings: PPAnsiChar; Lengths: PNativeUInt; ErrRet: PInteger): TCLHandle;
    {$IFDEF MSWINDOWS} stdcall; {$ELSE} cdecl; {$ENDIF}
  TclBuildProgram = function(Prog: TCLHandle; NumDevices: UInt32;
    Devices: PCLHandle; Options: PAnsiChar; Notify, UserData: Pointer): TCLInt;
    {$IFDEF MSWINDOWS} stdcall; {$ELSE} cdecl; {$ENDIF}
  TclCreateKernel = function(Prog: TCLHandle; Name: PAnsiChar;
    ErrRet: PInteger): TCLHandle;
    {$IFDEF MSWINDOWS} stdcall; {$ELSE} cdecl; {$ENDIF}
  TclSetKernelArg = function(Kernel: TCLHandle; ArgIdx: UInt32;
    ArgSize: NativeUInt; ArgValue: Pointer): TCLInt;
    {$IFDEF MSWINDOWS} stdcall; {$ELSE} cdecl; {$ENDIF}
  TclEnqueueWriteBuffer = function(Queue, Mem: TCLHandle; Blocking: UInt32;
    Offset, Size: NativeUInt; Ptr: Pointer; NumEvents: UInt32;
    WaitList, Event: Pointer): TCLInt;
    {$IFDEF MSWINDOWS} stdcall; {$ELSE} cdecl; {$ENDIF}
  TclEnqueueReadBuffer = function(Queue, Mem: TCLHandle; Blocking: UInt32;
    Offset, Size: NativeUInt; Ptr: Pointer; NumEvents: UInt32;
    WaitList, Event: Pointer): TCLInt;
    {$IFDEF MSWINDOWS} stdcall; {$ELSE} cdecl; {$ENDIF}
  TclEnqueueNDRangeKernel = function(Queue, Kernel: TCLHandle;
    WorkDim: UInt32; GlobalOffset, GlobalSize, LocalSize: PNativeUInt;
    NumEvents: UInt32; WaitList, Event: Pointer): TCLInt;
    {$IFDEF MSWINDOWS} stdcall; {$ELSE} cdecl; {$ENDIF}
  TclFinish = function(Queue: TCLHandle): TCLInt;
    {$IFDEF MSWINDOWS} stdcall; {$ELSE} cdecl; {$ENDIF}
  TclReleaseMemObject = function(Mem: TCLHandle): TCLInt;
    {$IFDEF MSWINDOWS} stdcall; {$ELSE} cdecl; {$ENDIF}

const
  KERNEL_SRC: AnsiString =
    '__kernel void matvec(__global const float* W, __global const float* x,' +
    ' __global float* y, const int rows, const int cols) {' +
    '  int r = get_global_id(0);' +
    '  if (r < rows) {' +
    '    float s = 0.0f;' +
    '    __global const float* row = W + (long)r * cols;' +
    '    for (int i = 0; i < cols; i++) s += row[i] * x[i];' +
    '    y[r] = s;' +
    '  }' +
    '}';

type
  TWeightBufEntry = record
    Mem: TCLHandle;
    Bytes: Int64;
    Key: Pointer;
  end;

  TOpenCLBackend = class(TComputeBackend)
  private
    FLib: NativeUInt;
    FDeviceName: string;
    FContext, FQueue, FProgram, FKernel, FDevice: TCLHandle;
    FXBuf, FYBuf: TCLHandle;
    FXCap, FYCap: NativeUInt;
    FWeightBufs: TDictionary<string, TWeightBufEntry>;
    FCacheBytes: Int64;
    FBroken: Boolean;
    FLock: TCriticalSection;
    clCreateBuffer: TclCreateBuffer;
    clSetKernelArg: TclSetKernelArg;
    clEnqueueWriteBuffer: TclEnqueueWriteBuffer;
    clEnqueueReadBuffer: TclEnqueueReadBuffer;
    clEnqueueNDRangeKernel: TclEnqueueNDRangeKernel;
    clFinish: TclFinish;
    clReleaseMemObject: TclReleaseMemObject;
    function Init(out Info: string): Boolean;
    function GetWeightBuf(W: PSingle; Rows, Cols: Integer; WKey: Pointer;
      WKeyOff: Int64): TCLHandle;
  public
    constructor Create;
    destructor Destroy; override;
    function Name: string; override;
    procedure MatVecF32W(Y, W, X: PSingle; Rows, Cols: Integer;
      Bias: PSingle; WKey: Pointer; WKeyOff: Int64); override;
    procedure InvalidateWeights(WKey: Pointer); override;
  end;

const
  MAX_GPU_CACHE_BYTES: Int64 = 256 * 1024 * 1024;
  GPU_MIN_WORK = 256 * 1024; // below this the transfer overhead dominates

var
  GBackend: TComputeBackend = nil;
  GCpuBackend: TComputeBackend = nil;

function LoadCLLibrary(out Lib: NativeUInt): Boolean;
begin
{$IFDEF MSWINDOWS}
  Lib := NativeUInt(LoadLibrary('OpenCL.dll'));
{$ELSE}
{$IFDEF MACOS}
  Lib := NativeUInt(dlopen(
    PAnsiChar(AnsiString('/System/Library/Frameworks/OpenCL.framework/OpenCL')),
    RTLD_LAZY));
{$ELSE}
  Lib := NativeUInt(dlopen(PAnsiChar(AnsiString('libOpenCL.so')), RTLD_LAZY));
{$ENDIF}
{$ENDIF}
  Result := Lib <> 0;
end;

function GetCLProc(Lib: NativeUInt; const Name: string): Pointer;
begin
{$IFDEF MSWINDOWS}
  Result := GetProcAddress(HMODULE(Lib), PChar(Name));
{$ELSE}
  Result := dlsym(Lib, PAnsiChar(AnsiString(Name)));
{$ENDIF}
end;

{ TOpenCLBackend }

constructor TOpenCLBackend.Create;
begin
  inherited Create;
  FLock := TCriticalSection.Create;
  FWeightBufs := TDictionary<string, TWeightBufEntry>.Create;
end;

destructor TOpenCLBackend.Destroy;
var
  E: TWeightBufEntry;
begin
  if Assigned(clReleaseMemObject) then
  begin
    for E in FWeightBufs.Values do
      clReleaseMemObject(E.Mem);
    if FXBuf <> nil then
      clReleaseMemObject(FXBuf);
    if FYBuf <> nil then
      clReleaseMemObject(FYBuf);
  end;
  FWeightBufs.Free;
  FLock.Free;
  inherited;
end;

function TOpenCLBackend.Init(out Info: string): Boolean;
var
  clGetPlatformIDs: TclGetPlatformIDs;
  clGetDeviceIDs: TclGetDeviceIDs;
  clGetDeviceInfo: TclGetDeviceInfo;
  clCreateContext: TclCreateContext;
  clCreateCommandQueue: TclCreateCommandQueue;
  clCreateProgramWithSource: TclCreateProgramWithSource;
  clBuildProgram: TclBuildProgram;
  clCreateKernel: TclCreateKernel;
  Platforms: array [0 .. 7] of TCLHandle;
  NumPlat, NumDev: UInt32;
  P: Integer;
  Err: Integer;
  Src: PAnsiChar;
  SrcLen: NativeUInt;
  NameBuf: array [0 .. 255] of AnsiChar;
begin
  Result := False;
  Info := '';
  if not LoadCLLibrary(FLib) then
  begin
    Info := 'OpenCL library not found';
    Exit;
  end;
  clGetPlatformIDs := TclGetPlatformIDs(GetCLProc(FLib, 'clGetPlatformIDs'));
  clGetDeviceIDs := TclGetDeviceIDs(GetCLProc(FLib, 'clGetDeviceIDs'));
  clGetDeviceInfo := TclGetDeviceInfo(GetCLProc(FLib, 'clGetDeviceInfo'));
  clCreateContext := TclCreateContext(GetCLProc(FLib, 'clCreateContext'));
  clCreateCommandQueue := TclCreateCommandQueue(GetCLProc(FLib,
    'clCreateCommandQueue'));
  clCreateProgramWithSource := TclCreateProgramWithSource(GetCLProc(FLib,
    'clCreateProgramWithSource'));
  clBuildProgram := TclBuildProgram(GetCLProc(FLib, 'clBuildProgram'));
  clCreateKernel := TclCreateKernel(GetCLProc(FLib, 'clCreateKernel'));
  clCreateBuffer := TclCreateBuffer(GetCLProc(FLib, 'clCreateBuffer'));
  clSetKernelArg := TclSetKernelArg(GetCLProc(FLib, 'clSetKernelArg'));
  clEnqueueWriteBuffer := TclEnqueueWriteBuffer(GetCLProc(FLib,
    'clEnqueueWriteBuffer'));
  clEnqueueReadBuffer := TclEnqueueReadBuffer(GetCLProc(FLib,
    'clEnqueueReadBuffer'));
  clEnqueueNDRangeKernel := TclEnqueueNDRangeKernel(GetCLProc(FLib,
    'clEnqueueNDRangeKernel'));
  clFinish := TclFinish(GetCLProc(FLib, 'clFinish'));
  clReleaseMemObject := TclReleaseMemObject(GetCLProc(FLib,
    'clReleaseMemObject'));
  if not (Assigned(clGetPlatformIDs) and Assigned(clGetDeviceIDs) and
    Assigned(clCreateContext) and Assigned(clCreateCommandQueue) and
    Assigned(clCreateProgramWithSource) and Assigned(clBuildProgram) and
    Assigned(clCreateKernel) and Assigned(clCreateBuffer) and
    Assigned(clSetKernelArg) and Assigned(clEnqueueWriteBuffer) and
    Assigned(clEnqueueReadBuffer) and Assigned(clEnqueueNDRangeKernel) and
    Assigned(clFinish) and Assigned(clReleaseMemObject)) then
  begin
    Info := 'OpenCL symbols incomplete';
    Exit;
  end;
  if (clGetPlatformIDs(8, @Platforms[0], @NumPlat) <> 0) or (NumPlat = 0) then
  begin
    Info := 'No OpenCL platform';
    Exit;
  end;
  FDevice := nil;
  for P := 0 to Integer(NumPlat) - 1 do
    if (clGetDeviceIDs(Platforms[P], CL_DEVICE_TYPE_GPU, 1, @FDevice,
      @NumDev) = 0) and (NumDev > 0) then
      Break
    else
      FDevice := nil;
  if FDevice = nil then
  begin
    Info := 'No GPU found';
    Exit;
  end;
  FillChar(NameBuf, SizeOf(NameBuf), 0);
  if Assigned(clGetDeviceInfo) then
    clGetDeviceInfo(FDevice, CL_DEVICE_NAME, SizeOf(NameBuf), @NameBuf, nil);
  FDeviceName := string(AnsiString(NameBuf));
  FContext := clCreateContext(nil, 1, @FDevice, nil, nil, @Err);
  if (FContext = nil) or (Err <> 0) then
  begin
    Info := 'clCreateContext failed';
    Exit;
  end;
  FQueue := clCreateCommandQueue(FContext, FDevice, 0, @Err);
  if (FQueue = nil) or (Err <> 0) then
  begin
    Info := 'clCreateCommandQueue failed';
    Exit;
  end;
  Src := PAnsiChar(KERNEL_SRC);
  SrcLen := Length(KERNEL_SRC);
  FProgram := clCreateProgramWithSource(FContext, 1, @Src, @SrcLen, @Err);
  if (FProgram = nil) or (Err <> 0) then
  begin
    Info := 'Kernel source failed';
    Exit;
  end;
  if clBuildProgram(FProgram, 1, @FDevice, '', nil, nil) <> 0 then
  begin
    Info := 'Kernel build failed';
    Exit;
  end;
  FKernel := clCreateKernel(FProgram, 'matvec', @Err);
  if (FKernel = nil) or (Err <> 0) then
  begin
    Info := 'clCreateKernel failed';
    Exit;
  end;
  Info := 'OpenCL: ' + FDeviceName;
  Result := True;
end;

function TOpenCLBackend.Name: string;
begin
  Result := 'GPU/OpenCL: ' + FDeviceName;
end;

function TOpenCLBackend.GetWeightBuf(W: PSingle; Rows, Cols: Integer;
  WKey: Pointer; WKeyOff: Int64): TCLHandle;
var
  Key: string;
  E: TWeightBufEntry;
  Bytes: Int64;
  Err: Integer;
begin
  Key := IntToStr(NativeUInt(WKey)) + '_' + IntToStr(WKeyOff);
  if FWeightBufs.TryGetValue(Key, E) then
    Exit(E.Mem);
  Bytes := Int64(Rows) * Cols * SizeOf(Single);
  if FCacheBytes + Bytes > MAX_GPU_CACHE_BYTES then
    Exit(nil); // cache full -> CPU fallback for this matrix
  E.Mem := clCreateBuffer(FContext, CL_MEM_READ_ONLY or CL_MEM_COPY_HOST_PTR,
    Bytes, W, @Err);
  if (E.Mem = nil) or (Err <> 0) then
    Exit(nil);
  E.Bytes := Bytes;
  E.Key := WKey;
  FWeightBufs.Add(Key, E);
  Inc(FCacheBytes, Bytes);
  Result := E.Mem;
end;

procedure TOpenCLBackend.InvalidateWeights(WKey: Pointer);
var
  Key: string;
  E: TWeightBufEntry;
  Doomed: TList<string>;
begin
  FLock.Enter;
  try
    Doomed := TList<string>.Create;
    try
      for Key in FWeightBufs.Keys do
        if FWeightBufs[Key].Key = WKey then
          Doomed.Add(Key);
      for Key in Doomed do
      begin
        E := FWeightBufs[Key];
        clReleaseMemObject(E.Mem);
        Dec(FCacheBytes, E.Bytes);
        FWeightBufs.Remove(Key);
      end;
    finally
      Doomed.Free;
    end;
  finally
    FLock.Leave;
  end;
end;

procedure TOpenCLBackend.MatVecF32W(Y, W, X: PSingle; Rows, Cols: Integer;
  Bias: PSingle; WKey: Pointer; WKeyOff: Int64);
var
  WBuf: TCLHandle;
  XBytes, YBytes: NativeUInt;
  GlobalSize: NativeUInt;
  Err, I: Integer;
begin
  if FBroken or (Int64(Rows) * Cols < GPU_MIN_WORK) then
  begin
    inherited;
    Exit;
  end;
  FLock.Enter;
  try
    WBuf := GetWeightBuf(W, Rows, Cols, WKey, WKeyOff);
    if WBuf = nil then
    begin
      FLock.Leave;
      try
        inherited;
      finally
        FLock.Enter;
      end;
      Exit;
    end;
    XBytes := NativeUInt(Cols) * SizeOf(Single);
    YBytes := NativeUInt(Rows) * SizeOf(Single);
    if XBytes > FXCap then
    begin
      if FXBuf <> nil then
        clReleaseMemObject(FXBuf);
      FXBuf := clCreateBuffer(FContext, CL_MEM_READ_ONLY, XBytes, nil, @Err);
      FXCap := XBytes;
    end;
    if YBytes > FYCap then
    begin
      if FYBuf <> nil then
        clReleaseMemObject(FYBuf);
      FYBuf := clCreateBuffer(FContext, CL_MEM_WRITE_ONLY, YBytes, nil, @Err);
      FYCap := YBytes;
    end;
    Err := clEnqueueWriteBuffer(FQueue, FXBuf, 1, 0, XBytes, X, 0, nil, nil);
    if Err = 0 then
      Err := clSetKernelArg(FKernel, 0, SizeOf(TCLHandle), @WBuf);
    if Err = 0 then
      Err := clSetKernelArg(FKernel, 1, SizeOf(TCLHandle), @FXBuf);
    if Err = 0 then
      Err := clSetKernelArg(FKernel, 2, SizeOf(TCLHandle), @FYBuf);
    if Err = 0 then
      Err := clSetKernelArg(FKernel, 3, SizeOf(Integer), @Rows);
    if Err = 0 then
      Err := clSetKernelArg(FKernel, 4, SizeOf(Integer), @Cols);
    if Err = 0 then
    begin
      GlobalSize := (NativeUInt(Rows) + 63) and not NativeUInt(63);
      Err := clEnqueueNDRangeKernel(FQueue, FKernel, 1, nil, @GlobalSize,
        nil, 0, nil, nil);
    end;
    if Err = 0 then
      Err := clEnqueueReadBuffer(FQueue, FYBuf, 1, 0, YBytes, Y, 0, nil, nil);
    if Err = 0 then
      clFinish(FQueue);
    if Err <> 0 then
    begin
      FBroken := True; // fall back to CPU permanently
      FLock.Leave;
      try
        inherited;
      finally
        FLock.Enter;
      end;
      Exit;
    end;
    if Bias <> nil then
      for I := 0 to Rows - 1 do
        Y[I] := Y[I] + Bias[I];
  finally
    FLock.Leave;
  end;
end;

{ ---------- Public API ---------- }

function Backend: TComputeBackend;
begin
  if GBackend <> nil then
    Result := GBackend
  else
    Result := GCpuBackend;
end;

function TryInitGpuBackend(out Info: string): Boolean;
var
  B: TOpenCLBackend;
begin
  Result := False;
  if GBackend <> nil then
  begin
    Info := GBackend.Name;
    Exit(True);
  end;
  B := TOpenCLBackend.Create;
  if B.Init(Info) then
  begin
    GBackend := B;
    Result := True;
  end
  else
    B.Free;
end;

procedure ShutdownGpuBackend;
begin
  FreeAndNil(GBackend);
end;

initialization
  GCpuBackend := TCpuBackend.Create;

finalization
  ShutdownGpuBackend;
  FreeAndNil(GCpuBackend);

end.

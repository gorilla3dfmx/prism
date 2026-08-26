unit Prism.Vulkan.Api;

{ Minimal Vulkan 1.0 bindings for Prism's compute backend -- no SDK, no
  third-party headers. The loader (vulkan-1.dll / libvulkan.so.1 /
  libMoltenVK.dylib) is opened at runtime; if it is absent, Prism simply
  keeps running on the CPU.

  Only what a compute-only pipeline needs is declared here: instance, device,
  queue, buffers, memory, descriptor sets, compute pipelines, command
  buffers, fences. No surfaces, swapchains, images, or render passes.

  Two conventions worth knowing before editing:

  * DISPATCHABLE handles (VkInstance, VkPhysicalDevice, VkDevice, VkQueue,
    VkCommandBuffer) are POINTERS. NON-dispatchable handles (VkBuffer,
    VkDeviceMemory, VkPipeline, ...) are ALWAYS UInt64 -- also on 32-bit
    targets. Mixing the two up compiles fine and crashes at runtime.

  * VkPhysicalDeviceLimits is transcribed in full rather than accessed via
    hand-computed offsets. It is long and we need only three fields, but
    letting the compiler lay the record out removes a whole class of silent
    ABI bugs. Do not "tidy" it by deleting unused fields. }

interface

{$MINENUMSIZE 4}
{$ALIGN 8}

uses
  System.SysUtils;

{ ---------------------------------------------------------------- handles
  VKAPI_CALL is __stdcall on Win32 and cdecl elsewhere; on Win64 both map
  onto the single native convention. Delphi has no valued $DEFINE, so the
  convention is spelled out inline on every prototype below. }

type
  { dispatchable handles -- pointers }
  TVkInstance       = Pointer;
  TVkPhysicalDevice = Pointer;
  TVkDevice         = Pointer;
  TVkQueue          = Pointer;
  TVkCommandBuffer  = Pointer;

  { non-dispatchable handles -- 64-bit on every target }
  TVkBuffer              = UInt64;
  TVkDeviceMemory        = UInt64;
  TVkShaderModule        = UInt64;
  TVkPipeline            = UInt64;
  TVkPipelineLayout      = UInt64;
  TVkPipelineCache       = UInt64;
  TVkDescriptorSetLayout = UInt64;
  TVkDescriptorPool      = UInt64;
  TVkDescriptorSet       = UInt64;
  TVkCommandPool         = UInt64;
  TVkFence               = UInt64;
  TVkSampler             = UInt64;

  TVkDeviceSize = UInt64;
  TVkResult     = Int32;
  TVkBool32     = UInt32;
  TVkFlags      = UInt32;

  PVkInstance       = ^TVkInstance;
  PVkPhysicalDevice = ^TVkPhysicalDevice;
  PVkDevice         = ^TVkDevice;
  PVkQueue          = ^TVkQueue;
  PVkCommandBuffer  = ^TVkCommandBuffer;
  PVkBuffer              = ^TVkBuffer;
  PVkDeviceMemory        = ^TVkDeviceMemory;
  PVkPipeline            = ^TVkPipeline;
  PVkDescriptorSetLayout = ^TVkDescriptorSetLayout;
  PVkDescriptorSet       = ^TVkDescriptorSet;
  PVkFence               = ^TVkFence;
  PVkDeviceSize          = ^TVkDeviceSize;

const
  VK_SUCCESS                     = 0;
  VK_ERROR_OUT_OF_HOST_MEMORY    = -1;
  VK_ERROR_OUT_OF_DEVICE_MEMORY  = -2;
  VK_ERROR_INITIALIZATION_FAILED = -3;
  VK_ERROR_INCOMPATIBLE_DRIVER   = -9;
  VK_TIMEOUT                     = 2;

  VK_WHOLE_SIZE          = UInt64($FFFFFFFFFFFFFFFF);
  VK_QUEUE_FAMILY_IGNORED = UInt32($FFFFFFFF);
  VK_MAX_PHYSICAL_DEVICE_NAME_SIZE = 256;
  VK_UUID_SIZE                     = 16;
  VK_MAX_MEMORY_TYPES              = 32;
  VK_MAX_MEMORY_HEAPS              = 16;

  { structure types }
  VK_STRUCTURE_TYPE_APPLICATION_INFO                = 0;
  VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO            = 1;
  VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO        = 2;
  VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO              = 3;
  VK_STRUCTURE_TYPE_SUBMIT_INFO                     = 4;
  VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO            = 5;
  VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE             = 6;
  VK_STRUCTURE_TYPE_FENCE_CREATE_INFO               = 8;
  VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO              = 12;
  VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO       = 16;
  VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18;
  VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO    = 29;
  VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO     = 30;
  VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32;
  VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO     = 33;
  VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO    = 34;
  VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET            = 35;
  VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO        = 39;
  VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO    = 40;
  VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO       = 42;
  VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER           = 44;

  { device types }
  VK_PHYSICAL_DEVICE_TYPE_OTHER          = 0;
  VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1;
  VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU   = 2;
  VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU    = 3;
  VK_PHYSICAL_DEVICE_TYPE_CPU            = 4;

  { queue flags }
  VK_QUEUE_GRAPHICS_BIT = $01;
  VK_QUEUE_COMPUTE_BIT  = $02;
  VK_QUEUE_TRANSFER_BIT = $04;

  { buffer usage }
  VK_BUFFER_USAGE_TRANSFER_SRC_BIT   = $0001;
  VK_BUFFER_USAGE_TRANSFER_DST_BIT   = $0002;
  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = $0020;

  VK_SHARING_MODE_EXCLUSIVE = 0;

  { memory property flags }
  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT  = $01;
  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT  = $02;
  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = $04;
  VK_MEMORY_PROPERTY_HOST_CACHED_BIT   = $08;

  VK_MEMORY_HEAP_DEVICE_LOCAL_BIT = $01;

  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7;
  VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT = $01;

  VK_SHADER_STAGE_COMPUTE_BIT   = $0020;
  VK_PIPELINE_BIND_POINT_COMPUTE = 1;

  VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0;
  VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = $02;
  VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT     = $01;

  { access flags }
  VK_ACCESS_SHADER_READ_BIT    = $0020;
  VK_ACCESS_SHADER_WRITE_BIT   = $0040;
  VK_ACCESS_TRANSFER_READ_BIT  = $0800;
  VK_ACCESS_TRANSFER_WRITE_BIT = $1000;

  { pipeline stages }
  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = $0800;
  VK_PIPELINE_STAGE_TRANSFER_BIT       = $1000;

  { macOS/iOS: the loader refuses MoltenVK unless portability is requested }
  VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR = $01;

function VK_MAKE_VERSION(Major, Minor, Patch: Cardinal): Cardinal; inline;
function VK_VERSION_MAJOR(V: Cardinal): Cardinal; inline;
function VK_VERSION_MINOR(V: Cardinal): Cardinal; inline;
function VK_VERSION_PATCH(V: Cardinal): Cardinal; inline;

type
  TVkExtent3D = record
    Width, Height, Depth: UInt32;
  end;

  TVkApplicationInfo = record
    sType: UInt32;
    pNext: Pointer;
    pApplicationName: PAnsiChar;
    applicationVersion: UInt32;
    pEngineName: PAnsiChar;
    engineVersion: UInt32;
    apiVersion: UInt32;
  end;
  PVkApplicationInfo = ^TVkApplicationInfo;

  TVkInstanceCreateInfo = record
    sType: UInt32;
    pNext: Pointer;
    flags: TVkFlags;
    pApplicationInfo: PVkApplicationInfo;
    enabledLayerCount: UInt32;
    ppEnabledLayerNames: PPAnsiChar;
    enabledExtensionCount: UInt32;
    ppEnabledExtensionNames: PPAnsiChar;
  end;

  { Full Vulkan 1.0 limits block. We read maxStorageBufferRange,
    maxComputeWorkGroupCount and maxComputeSharedMemorySize; the rest is here
    purely so the compiler gets the offsets right. Do not prune. }
  TVkPhysicalDeviceLimits = record
    maxImageDimension1D: UInt32;
    maxImageDimension2D: UInt32;
    maxImageDimension3D: UInt32;
    maxImageDimensionCube: UInt32;
    maxImageArrayLayers: UInt32;
    maxTexelBufferElements: UInt32;
    maxUniformBufferRange: UInt32;
    maxStorageBufferRange: UInt32;
    maxPushConstantsSize: UInt32;
    maxMemoryAllocationCount: UInt32;
    maxSamplerAllocationCount: UInt32;
    bufferImageGranularity: TVkDeviceSize;
    sparseAddressSpaceSize: TVkDeviceSize;
    maxBoundDescriptorSets: UInt32;
    maxPerStageDescriptorSamplers: UInt32;
    maxPerStageDescriptorUniformBuffers: UInt32;
    maxPerStageDescriptorStorageBuffers: UInt32;
    maxPerStageDescriptorSampledImages: UInt32;
    maxPerStageDescriptorStorageImages: UInt32;
    maxPerStageDescriptorInputAttachments: UInt32;
    maxPerStageResources: UInt32;
    maxDescriptorSetSamplers: UInt32;
    maxDescriptorSetUniformBuffers: UInt32;
    maxDescriptorSetUniformBuffersDynamic: UInt32;
    maxDescriptorSetStorageBuffers: UInt32;
    maxDescriptorSetStorageBuffersDynamic: UInt32;
    maxDescriptorSetSampledImages: UInt32;
    maxDescriptorSetStorageImages: UInt32;
    maxDescriptorSetInputAttachments: UInt32;
    maxVertexInputAttributes: UInt32;
    maxVertexInputBindings: UInt32;
    maxVertexInputAttributeOffset: UInt32;
    maxVertexInputBindingStride: UInt32;
    maxVertexOutputComponents: UInt32;
    maxTessellationGenerationLevel: UInt32;
    maxTessellationPatchSize: UInt32;
    maxTessellationControlPerVertexInputComponents: UInt32;
    maxTessellationControlPerVertexOutputComponents: UInt32;
    maxTessellationControlPerPatchOutputComponents: UInt32;
    maxTessellationControlTotalOutputComponents: UInt32;
    maxTessellationEvaluationInputComponents: UInt32;
    maxTessellationEvaluationOutputComponents: UInt32;
    maxGeometryShaderInvocations: UInt32;
    maxGeometryInputComponents: UInt32;
    maxGeometryOutputComponents: UInt32;
    maxGeometryOutputVertices: UInt32;
    maxGeometryTotalOutputComponents: UInt32;
    maxFragmentInputComponents: UInt32;
    maxFragmentOutputAttachments: UInt32;
    maxFragmentDualSrcAttachments: UInt32;
    maxFragmentCombinedOutputResources: UInt32;
    maxComputeSharedMemorySize: UInt32;
    maxComputeWorkGroupCount: array [0 .. 2] of UInt32;
    maxComputeWorkGroupInvocations: UInt32;
    maxComputeWorkGroupSize: array [0 .. 2] of UInt32;
    subPixelPrecisionBits: UInt32;
    subTexelPrecisionBits: UInt32;
    mipmapPrecisionBits: UInt32;
    maxDrawIndexedIndexValue: UInt32;
    maxDrawIndirectCount: UInt32;
    maxSamplerLodBias: Single;
    maxSamplerAnisotropy: Single;
    maxViewports: UInt32;
    maxViewportDimensions: array [0 .. 1] of UInt32;
    viewportBoundsRange: array [0 .. 1] of Single;
    viewportSubPixelBits: UInt32;
    minMemoryMapAlignment: NativeUInt;
    minTexelBufferOffsetAlignment: TVkDeviceSize;
    minUniformBufferOffsetAlignment: TVkDeviceSize;
    minStorageBufferOffsetAlignment: TVkDeviceSize;
    minTexelOffset: Int32;
    maxTexelOffset: UInt32;
    minTexelGatherOffset: Int32;
    maxTexelGatherOffset: UInt32;
    minInterpolationOffset: Single;
    maxInterpolationOffset: Single;
    subPixelInterpolationOffsetBits: UInt32;
    maxFramebufferWidth: UInt32;
    maxFramebufferHeight: UInt32;
    maxFramebufferLayers: UInt32;
    framebufferColorSampleCounts: TVkFlags;
    framebufferDepthSampleCounts: TVkFlags;
    framebufferStencilSampleCounts: TVkFlags;
    framebufferNoAttachmentsSampleCounts: TVkFlags;
    maxColorAttachments: UInt32;
    sampledImageColorSampleCounts: TVkFlags;
    sampledImageIntegerSampleCounts: TVkFlags;
    sampledImageDepthSampleCounts: TVkFlags;
    sampledImageStencilSampleCounts: TVkFlags;
    storageImageSampleCounts: TVkFlags;
    maxSampleMaskWords: UInt32;
    timestampComputeAndGraphics: TVkBool32;
    timestampPeriod: Single;
    maxClipDistances: UInt32;
    maxCullDistances: UInt32;
    maxCombinedClipAndCullDistances: UInt32;
    discreteQueuePriorities: UInt32;
    pointSizeRange: array [0 .. 1] of Single;
    lineWidthRange: array [0 .. 1] of Single;
    pointSizeGranularity: Single;
    lineWidthGranularity: Single;
    strictLines: TVkBool32;
    standardSampleLocations: TVkBool32;
    optimalBufferCopyOffsetAlignment: TVkDeviceSize;
    optimalBufferCopyRowPitchAlignment: TVkDeviceSize;
    nonCoherentAtomSize: TVkDeviceSize;
  end;

  TVkPhysicalDeviceSparseProperties = record
    residencyStandard2DBlockShape: TVkBool32;
    residencyStandard2DMultisampleBlockShape: TVkBool32;
    residencyStandard3DBlockShape: TVkBool32;
    residencyAlignedMipSize: TVkBool32;
    residencyNonResidentStrict: TVkBool32;
  end;

  TVkPhysicalDeviceProperties = record
    apiVersion: UInt32;
    driverVersion: UInt32;
    vendorID: UInt32;
    deviceID: UInt32;
    deviceType: UInt32;
    deviceName: array [0 .. VK_MAX_PHYSICAL_DEVICE_NAME_SIZE - 1] of AnsiChar;
    pipelineCacheUUID: array [0 .. VK_UUID_SIZE - 1] of Byte;
    limits: TVkPhysicalDeviceLimits;
    sparseProperties: TVkPhysicalDeviceSparseProperties;
  end;
  PVkPhysicalDeviceProperties = ^TVkPhysicalDeviceProperties;

  TVkMemoryType = record
    propertyFlags: TVkFlags;
    heapIndex: UInt32;
  end;

  TVkMemoryHeap = record
    size: TVkDeviceSize;
    flags: TVkFlags;
  end;

  TVkPhysicalDeviceMemoryProperties = record
    memoryTypeCount: UInt32;
    memoryTypes: array [0 .. VK_MAX_MEMORY_TYPES - 1] of TVkMemoryType;
    memoryHeapCount: UInt32;
    memoryHeaps: array [0 .. VK_MAX_MEMORY_HEAPS - 1] of TVkMemoryHeap;
  end;
  PVkPhysicalDeviceMemoryProperties = ^TVkPhysicalDeviceMemoryProperties;

  TVkQueueFamilyProperties = record
    queueFlags: TVkFlags;
    queueCount: UInt32;
    timestampValidBits: UInt32;
    minImageTransferGranularity: TVkExtent3D;
  end;
  PVkQueueFamilyProperties = ^TVkQueueFamilyProperties;

  TVkExtensionProperties = record
    extensionName: array [0 .. 255] of AnsiChar;
    specVersion: UInt32;
  end;
  PVkExtensionProperties = ^TVkExtensionProperties;

  TVkDeviceQueueCreateInfo = record
    sType: UInt32;
    pNext: Pointer;
    flags: TVkFlags;
    queueFamilyIndex: UInt32;
    queueCount: UInt32;
    pQueuePriorities: PSingle;
  end;
  PVkDeviceQueueCreateInfo = ^TVkDeviceQueueCreateInfo;

  TVkDeviceCreateInfo = record
    sType: UInt32;
    pNext: Pointer;
    flags: TVkFlags;
    queueCreateInfoCount: UInt32;
    pQueueCreateInfos: PVkDeviceQueueCreateInfo;
    enabledLayerCount: UInt32;
    ppEnabledLayerNames: PPAnsiChar;
    enabledExtensionCount: UInt32;
    ppEnabledExtensionNames: PPAnsiChar;
    pEnabledFeatures: Pointer;
  end;

  TVkBufferCreateInfo = record
    sType: UInt32;
    pNext: Pointer;
    flags: TVkFlags;
    size: TVkDeviceSize;
    usage: TVkFlags;
    sharingMode: UInt32;
    queueFamilyIndexCount: UInt32;
    pQueueFamilyIndices: PUInt32;
  end;

  TVkMemoryRequirements = record
    size: TVkDeviceSize;
    alignment: TVkDeviceSize;
    memoryTypeBits: UInt32;
  end;
  PVkMemoryRequirements = ^TVkMemoryRequirements;

  TVkMemoryAllocateInfo = record
    sType: UInt32;
    pNext: Pointer;
    allocationSize: TVkDeviceSize;
    memoryTypeIndex: UInt32;
  end;

  TVkShaderModuleCreateInfo = record
    sType: UInt32;
    pNext: Pointer;
    flags: TVkFlags;
    codeSize: NativeUInt;
    pCode: PUInt32;
  end;

  TVkDescriptorSetLayoutBinding = record
    binding: UInt32;
    descriptorType: UInt32;
    descriptorCount: UInt32;
    stageFlags: TVkFlags;
    pImmutableSamplers: Pointer;
  end;
  PVkDescriptorSetLayoutBinding = ^TVkDescriptorSetLayoutBinding;

  TVkDescriptorSetLayoutCreateInfo = record
    sType: UInt32;
    pNext: Pointer;
    flags: TVkFlags;
    bindingCount: UInt32;
    pBindings: PVkDescriptorSetLayoutBinding;
  end;

  TVkPushConstantRange = record
    stageFlags: TVkFlags;
    offset: UInt32;
    size: UInt32;
  end;
  PVkPushConstantRange = ^TVkPushConstantRange;

  TVkPipelineLayoutCreateInfo = record
    sType: UInt32;
    pNext: Pointer;
    flags: TVkFlags;
    setLayoutCount: UInt32;
    pSetLayouts: PVkDescriptorSetLayout;
    pushConstantRangeCount: UInt32;
    pPushConstantRanges: PVkPushConstantRange;
  end;

  TVkPipelineShaderStageCreateInfo = record
    sType: UInt32;
    pNext: Pointer;
    flags: TVkFlags;
    stage: TVkFlags;
    module: TVkShaderModule;
    pName: PAnsiChar;
    pSpecializationInfo: Pointer;
  end;

  TVkComputePipelineCreateInfo = record
    sType: UInt32;
    pNext: Pointer;
    flags: TVkFlags;
    stage: TVkPipelineShaderStageCreateInfo;
    layout: TVkPipelineLayout;
    basePipelineHandle: TVkPipeline;
    basePipelineIndex: Int32;
  end;
  PVkComputePipelineCreateInfo = ^TVkComputePipelineCreateInfo;

  TVkDescriptorPoolSize = record
    typ: UInt32;
    descriptorCount: UInt32;
  end;
  PVkDescriptorPoolSize = ^TVkDescriptorPoolSize;

  TVkDescriptorPoolCreateInfo = record
    sType: UInt32;
    pNext: Pointer;
    flags: TVkFlags;
    maxSets: UInt32;
    poolSizeCount: UInt32;
    pPoolSizes: PVkDescriptorPoolSize;
  end;

  TVkDescriptorSetAllocateInfo = record
    sType: UInt32;
    pNext: Pointer;
    descriptorPool: TVkDescriptorPool;
    descriptorSetCount: UInt32;
    pSetLayouts: PVkDescriptorSetLayout;
  end;

  TVkDescriptorBufferInfo = record
    buffer: TVkBuffer;
    offset: TVkDeviceSize;
    range: TVkDeviceSize;
  end;
  PVkDescriptorBufferInfo = ^TVkDescriptorBufferInfo;

  TVkWriteDescriptorSet = record
    sType: UInt32;
    pNext: Pointer;
    dstSet: TVkDescriptorSet;
    dstBinding: UInt32;
    dstArrayElement: UInt32;
    descriptorCount: UInt32;
    descriptorType: UInt32;
    pImageInfo: Pointer;
    pBufferInfo: PVkDescriptorBufferInfo;
    pTexelBufferView: Pointer;
  end;
  PVkWriteDescriptorSet = ^TVkWriteDescriptorSet;

  TVkCommandPoolCreateInfo = record
    sType: UInt32;
    pNext: Pointer;
    flags: TVkFlags;
    queueFamilyIndex: UInt32;
  end;

  TVkCommandBufferAllocateInfo = record
    sType: UInt32;
    pNext: Pointer;
    commandPool: TVkCommandPool;
    level: UInt32;
    commandBufferCount: UInt32;
  end;

  TVkCommandBufferBeginInfo = record
    sType: UInt32;
    pNext: Pointer;
    flags: TVkFlags;
    pInheritanceInfo: Pointer;
  end;

  TVkSubmitInfo = record
    sType: UInt32;
    pNext: Pointer;
    waitSemaphoreCount: UInt32;
    pWaitSemaphores: Pointer;
    pWaitDstStageMask: PUInt32;
    commandBufferCount: UInt32;
    pCommandBuffers: PVkCommandBuffer;
    signalSemaphoreCount: UInt32;
    pSignalSemaphores: Pointer;
  end;
  PVkSubmitInfo = ^TVkSubmitInfo;

  TVkFenceCreateInfo = record
    sType: UInt32;
    pNext: Pointer;
    flags: TVkFlags;
  end;

  TVkBufferCopy = record
    srcOffset: TVkDeviceSize;
    dstOffset: TVkDeviceSize;
    size: TVkDeviceSize;
  end;
  PVkBufferCopy = ^TVkBufferCopy;

  TVkBufferMemoryBarrier = record
    sType: UInt32;
    pNext: Pointer;
    srcAccessMask: TVkFlags;
    dstAccessMask: TVkFlags;
    srcQueueFamilyIndex: UInt32;
    dstQueueFamilyIndex: UInt32;
    buffer: TVkBuffer;
    offset: TVkDeviceSize;
    size: TVkDeviceSize;
  end;
  PVkBufferMemoryBarrier = ^TVkBufferMemoryBarrier;

{ ------------------------------------------------------------------ entry
  points. Global/instance level are fetched through vkGetInstanceProcAddr,
  device level through vkGetDeviceProcAddr (skips the loader's dispatch
  trampoline on every call). }

type
  TvkGetInstanceProcAddr = function(Instance: TVkInstance;
    pName: PAnsiChar): Pointer; {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkGetDeviceProcAddr = function(Device: TVkDevice;
    pName: PAnsiChar): Pointer; {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};

  TvkCreateInstance = function(const pCreateInfo: TVkInstanceCreateInfo;
    pAllocator: Pointer; pInstance: PVkInstance): TVkResult;
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkDestroyInstance = procedure(Instance: TVkInstance; pAllocator: Pointer);
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkEnumeratePhysicalDevices = function(Instance: TVkInstance;
    pCount: PUInt32; pDevices: PVkPhysicalDevice): TVkResult;
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkGetPhysicalDeviceProperties = procedure(Device: TVkPhysicalDevice;
    pProps: PVkPhysicalDeviceProperties);
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkGetPhysicalDeviceMemoryProperties = procedure(Device: TVkPhysicalDevice;
    pProps: PVkPhysicalDeviceMemoryProperties);
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkGetPhysicalDeviceQueueFamilyProperties = procedure(
    Device: TVkPhysicalDevice; pCount: PUInt32;
    pProps: PVkQueueFamilyProperties);
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkEnumerateDeviceExtensionProperties = function(Device: TVkPhysicalDevice;
    pLayerName: PAnsiChar; pCount: PUInt32;
    pProps: PVkExtensionProperties): TVkResult;
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkCreateDevice = function(Physical: TVkPhysicalDevice;
    const pCreateInfo: TVkDeviceCreateInfo; pAllocator: Pointer;
    pDevice: PVkDevice): TVkResult;
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkDestroyDevice = procedure(Device: TVkDevice; pAllocator: Pointer);
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkGetDeviceQueue = procedure(Device: TVkDevice;
    QueueFamilyIndex, QueueIndex: UInt32; pQueue: PVkQueue);
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkDeviceWaitIdle = function(Device: TVkDevice): TVkResult;
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};

  TvkCreateBuffer = function(Device: TVkDevice;
    const pCreateInfo: TVkBufferCreateInfo; pAllocator: Pointer;
    pBuffer: PVkBuffer): TVkResult;
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkDestroyBuffer = procedure(Device: TVkDevice; Buffer: TVkBuffer;
    pAllocator: Pointer); {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkGetBufferMemoryRequirements = procedure(Device: TVkDevice;
    Buffer: TVkBuffer; pReq: PVkMemoryRequirements);
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkAllocateMemory = function(Device: TVkDevice;
    const pAllocInfo: TVkMemoryAllocateInfo; pAllocator: Pointer;
    pMemory: PVkDeviceMemory): TVkResult;
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkFreeMemory = procedure(Device: TVkDevice; Memory: TVkDeviceMemory;
    pAllocator: Pointer); {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkBindBufferMemory = function(Device: TVkDevice; Buffer: TVkBuffer;
    Memory: TVkDeviceMemory; Offset: TVkDeviceSize): TVkResult;
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkMapMemory = function(Device: TVkDevice; Memory: TVkDeviceMemory;
    Offset, Size: TVkDeviceSize; Flags: TVkFlags;
    ppData: PPointer): TVkResult;
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkUnmapMemory = procedure(Device: TVkDevice; Memory: TVkDeviceMemory);
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};

  TvkCreateShaderModule = function(Device: TVkDevice;
    const pCreateInfo: TVkShaderModuleCreateInfo; pAllocator: Pointer;
    pModule: PUInt64): TVkResult;
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkDestroyShaderModule = procedure(Device: TVkDevice;
    Module: TVkShaderModule; pAllocator: Pointer);
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkCreateDescriptorSetLayout = function(Device: TVkDevice;
    const pCreateInfo: TVkDescriptorSetLayoutCreateInfo; pAllocator: Pointer;
    pLayout: PUInt64): TVkResult;
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkDestroyDescriptorSetLayout = procedure(Device: TVkDevice;
    Layout: TVkDescriptorSetLayout; pAllocator: Pointer);
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkCreatePipelineLayout = function(Device: TVkDevice;
    const pCreateInfo: TVkPipelineLayoutCreateInfo; pAllocator: Pointer;
    pLayout: PUInt64): TVkResult;
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkDestroyPipelineLayout = procedure(Device: TVkDevice;
    Layout: TVkPipelineLayout; pAllocator: Pointer);
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkCreateComputePipelines = function(Device: TVkDevice;
    Cache: TVkPipelineCache; CreateInfoCount: UInt32;
    pCreateInfos: PVkComputePipelineCreateInfo; pAllocator: Pointer;
    pPipelines: PVkPipeline): TVkResult;
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkDestroyPipeline = procedure(Device: TVkDevice; Pipeline: TVkPipeline;
    pAllocator: Pointer); {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};

  TvkCreateDescriptorPool = function(Device: TVkDevice;
    const pCreateInfo: TVkDescriptorPoolCreateInfo; pAllocator: Pointer;
    pPool: PUInt64): TVkResult;
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkDestroyDescriptorPool = procedure(Device: TVkDevice;
    Pool: TVkDescriptorPool; pAllocator: Pointer);
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkAllocateDescriptorSets = function(Device: TVkDevice;
    const pAllocInfo: TVkDescriptorSetAllocateInfo;
    pSets: PVkDescriptorSet): TVkResult;
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkFreeDescriptorSets = function(Device: TVkDevice;
    Pool: TVkDescriptorPool; Count: UInt32;
    pSets: PVkDescriptorSet): TVkResult;
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkUpdateDescriptorSets = procedure(Device: TVkDevice;
    WriteCount: UInt32; pWrites: PVkWriteDescriptorSet;
    CopyCount: UInt32; pCopies: Pointer);
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};

  TvkCreateCommandPool = function(Device: TVkDevice;
    const pCreateInfo: TVkCommandPoolCreateInfo; pAllocator: Pointer;
    pPool: PUInt64): TVkResult;
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkDestroyCommandPool = procedure(Device: TVkDevice; Pool: TVkCommandPool;
    pAllocator: Pointer); {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkAllocateCommandBuffers = function(Device: TVkDevice;
    const pAllocInfo: TVkCommandBufferAllocateInfo;
    pBuffers: PVkCommandBuffer): TVkResult;
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkFreeCommandBuffers = procedure(Device: TVkDevice; Pool: TVkCommandPool;
    Count: UInt32; pBuffers: PVkCommandBuffer);
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkBeginCommandBuffer = function(CmdBuf: TVkCommandBuffer;
    const pBeginInfo: TVkCommandBufferBeginInfo): TVkResult;
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkEndCommandBuffer = function(CmdBuf: TVkCommandBuffer): TVkResult;
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkResetCommandBuffer = function(CmdBuf: TVkCommandBuffer;
    Flags: TVkFlags): TVkResult;
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};

  TvkCmdBindPipeline = procedure(CmdBuf: TVkCommandBuffer;
    BindPoint: UInt32; Pipeline: TVkPipeline);
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkCmdBindDescriptorSets = procedure(CmdBuf: TVkCommandBuffer;
    BindPoint: UInt32; Layout: TVkPipelineLayout;
    FirstSet, SetCount: UInt32; pSets: PVkDescriptorSet;
    DynOffsetCount: UInt32; pDynOffsets: PUInt32);
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkCmdPushConstants = procedure(CmdBuf: TVkCommandBuffer;
    Layout: TVkPipelineLayout; StageFlags: TVkFlags;
    Offset, Size: UInt32; pValues: Pointer);
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkCmdDispatch = procedure(CmdBuf: TVkCommandBuffer;
    X, Y, Z: UInt32); {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkCmdCopyBuffer = procedure(CmdBuf: TVkCommandBuffer;
    Src, Dst: TVkBuffer; RegionCount: UInt32; pRegions: PVkBufferCopy);
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkCmdPipelineBarrier = procedure(CmdBuf: TVkCommandBuffer;
    SrcStage, DstStage: TVkFlags; DependencyFlags: TVkFlags;
    MemBarrierCount: UInt32; pMemBarriers: Pointer;
    BufBarrierCount: UInt32; pBufBarriers: PVkBufferMemoryBarrier;
    ImgBarrierCount: UInt32; pImgBarriers: Pointer);
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};

  TvkQueueSubmit = function(Queue: TVkQueue; SubmitCount: UInt32;
    pSubmits: PVkSubmitInfo; Fence: TVkFence): TVkResult;
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkQueueWaitIdle = function(Queue: TVkQueue): TVkResult;
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkCreateFence = function(Device: TVkDevice;
    const pCreateInfo: TVkFenceCreateInfo; pAllocator: Pointer;
    pFence: PUInt64): TVkResult;
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkDestroyFence = procedure(Device: TVkDevice; Fence: TVkFence;
    pAllocator: Pointer); {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkWaitForFences = function(Device: TVkDevice; Count: UInt32;
    pFences: PVkFence; WaitAll: TVkBool32; Timeout: UInt64): TVkResult;
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};
  TvkResetFences = function(Device: TVkDevice; Count: UInt32;
    pFences: PVkFence): TVkResult;
    {$IFDEF MSWINDOWS} stdcall {$ELSE} cdecl {$ENDIF};

type
  { Everything Prism calls, grouped so the backend can hold one instance and
    so a failed symbol lookup is a single, obvious check. }
  TVulkanApi = record
    GetInstanceProcAddr: TvkGetInstanceProcAddr;
    GetDeviceProcAddr: TvkGetDeviceProcAddr;

    CreateInstance: TvkCreateInstance;
    DestroyInstance: TvkDestroyInstance;
    EnumeratePhysicalDevices: TvkEnumeratePhysicalDevices;
    GetPhysicalDeviceProperties: TvkGetPhysicalDeviceProperties;
    GetPhysicalDeviceMemoryProperties: TvkGetPhysicalDeviceMemoryProperties;
    GetPhysicalDeviceQueueFamilyProperties
      : TvkGetPhysicalDeviceQueueFamilyProperties;
    EnumerateDeviceExtensionProperties: TvkEnumerateDeviceExtensionProperties;
    CreateDevice: TvkCreateDevice;
    DestroyDevice: TvkDestroyDevice;
    GetDeviceQueue: TvkGetDeviceQueue;
    DeviceWaitIdle: TvkDeviceWaitIdle;

    CreateBuffer: TvkCreateBuffer;
    DestroyBuffer: TvkDestroyBuffer;
    GetBufferMemoryRequirements: TvkGetBufferMemoryRequirements;
    AllocateMemory: TvkAllocateMemory;
    FreeMemory: TvkFreeMemory;
    BindBufferMemory: TvkBindBufferMemory;
    MapMemory: TvkMapMemory;
    UnmapMemory: TvkUnmapMemory;

    CreateShaderModule: TvkCreateShaderModule;
    DestroyShaderModule: TvkDestroyShaderModule;
    CreateDescriptorSetLayout: TvkCreateDescriptorSetLayout;
    DestroyDescriptorSetLayout: TvkDestroyDescriptorSetLayout;
    CreatePipelineLayout: TvkCreatePipelineLayout;
    DestroyPipelineLayout: TvkDestroyPipelineLayout;
    CreateComputePipelines: TvkCreateComputePipelines;
    DestroyPipeline: TvkDestroyPipeline;

    CreateDescriptorPool: TvkCreateDescriptorPool;
    DestroyDescriptorPool: TvkDestroyDescriptorPool;
    AllocateDescriptorSets: TvkAllocateDescriptorSets;
    FreeDescriptorSets: TvkFreeDescriptorSets;
    UpdateDescriptorSets: TvkUpdateDescriptorSets;

    CreateCommandPool: TvkCreateCommandPool;
    DestroyCommandPool: TvkDestroyCommandPool;
    AllocateCommandBuffers: TvkAllocateCommandBuffers;
    FreeCommandBuffers: TvkFreeCommandBuffers;
    BeginCommandBuffer: TvkBeginCommandBuffer;
    EndCommandBuffer: TvkEndCommandBuffer;
    ResetCommandBuffer: TvkResetCommandBuffer;

    CmdBindPipeline: TvkCmdBindPipeline;
    CmdBindDescriptorSets: TvkCmdBindDescriptorSets;
    CmdPushConstants: TvkCmdPushConstants;
    CmdDispatch: TvkCmdDispatch;
    CmdCopyBuffer: TvkCmdCopyBuffer;
    CmdPipelineBarrier: TvkCmdPipelineBarrier;

    QueueSubmit: TvkQueueSubmit;
    QueueWaitIdle: TvkQueueWaitIdle;
    CreateFence: TvkCreateFence;
    DestroyFence: TvkDestroyFence;
    WaitForFences: TvkWaitForFences;
    ResetFences: TvkResetFences;
  end;

{ Opens the platform loader and resolves vkGetInstanceProcAddr plus the
  global entry points. Returns False (with Why filled in) when Vulkan is
  simply not present -- that is a normal outcome, not an error. }
function VkLoadLoader(var Api: TVulkanApi; out Why: string): Boolean;

{ Resolves the instance- and device-level entry points. Call after
  vkCreateInstance / vkCreateDevice respectively. }
function VkLoadInstanceProcs(var Api: TVulkanApi; Instance: TVkInstance;
  out Why: string): Boolean;
function VkLoadDeviceProcs(var Api: TVulkanApi; Device: TVkDevice;
  out Why: string): Boolean;

procedure VkUnloadLoader;

function VkResultStr(R: TVkResult): string;

implementation

{$IFDEF MSWINDOWS}
uses
  Winapi.Windows;
{$ELSE}
uses
  Posix.Dlfcn;
{$ENDIF}

var
  GLib: NativeUInt = 0;

function VK_MAKE_VERSION(Major, Minor, Patch: Cardinal): Cardinal;
begin
  Result := (Major shl 22) or (Minor shl 12) or Patch;
end;

function VK_VERSION_MAJOR(V: Cardinal): Cardinal;
begin
  Result := V shr 22;
end;

function VK_VERSION_MINOR(V: Cardinal): Cardinal;
begin
  Result := (V shr 12) and $3FF;
end;

function VK_VERSION_PATCH(V: Cardinal): Cardinal;
begin
  Result := V and $FFF;
end;

function VkResultStr(R: TVkResult): string;
begin
  case R of
    0: Result := 'VK_SUCCESS';
    1: Result := 'VK_NOT_READY';
    2: Result := 'VK_TIMEOUT';
    -1: Result := 'VK_ERROR_OUT_OF_HOST_MEMORY';
    -2: Result := 'VK_ERROR_OUT_OF_DEVICE_MEMORY';
    -3: Result := 'VK_ERROR_INITIALIZATION_FAILED';
    -4: Result := 'VK_ERROR_DEVICE_LOST';
    -5: Result := 'VK_ERROR_MEMORY_MAP_FAILED';
    -6: Result := 'VK_ERROR_LAYER_NOT_PRESENT';
    -7: Result := 'VK_ERROR_EXTENSION_NOT_PRESENT';
    -8: Result := 'VK_ERROR_FEATURE_NOT_PRESENT';
    -9: Result := 'VK_ERROR_INCOMPATIBLE_DRIVER';
    -10: Result := 'VK_ERROR_TOO_MANY_OBJECTS';
    -11: Result := 'VK_ERROR_FORMAT_NOT_SUPPORTED';
    -12: Result := 'VK_ERROR_FRAGMENTED_POOL';
    -13: Result := 'VK_ERROR_UNKNOWN';
  else
    Result := 'VkResult(' + IntToStr(R) + ')';
  end;
end;

{ ---------- platform loader ---------- }

function OpenLoader(out Lib: NativeUInt): string;
{$IFDEF MSWINDOWS}
begin
  Lib := NativeUInt(LoadLibrary('vulkan-1.dll'));
  if Lib = 0 then
    Result := 'vulkan-1.dll not found'
  else
    Result := '';
end;
{$ELSE}

  function Try_(const S: string): NativeUInt;
  begin
    Result := NativeUInt(dlopen(PAnsiChar(AnsiString(S)), RTLD_NOW or RTLD_LOCAL));
  end;

begin
  Lib := 0;
{$IFDEF MACOS}
  { MoltenVK ships either as a plain dylib or inside a framework }
  Lib := Try_('libvulkan.1.dylib');
  if Lib = 0 then Lib := Try_('libvulkan.dylib');
  if Lib = 0 then Lib := Try_('libMoltenVK.dylib');
  if Lib = 0 then Lib := Try_('/usr/local/lib/libvulkan.1.dylib');
{$ELSE}
  Lib := Try_('libvulkan.so.1');
  if Lib = 0 then Lib := Try_('libvulkan.so');
{$ENDIF}
  if Lib = 0 then
    Result := 'Vulkan loader library not found'
  else
    Result := '';
end;
{$ENDIF}

function LibSym(Lib: NativeUInt; const Name: string): Pointer;
begin
{$IFDEF MSWINDOWS}
  Result := GetProcAddress(HMODULE(Lib), PChar(Name));
{$ELSE}
  Result := dlsym(Lib, PAnsiChar(AnsiString(Name)));
{$ENDIF}
end;

function VkLoadLoader(var Api: TVulkanApi; out Why: string): Boolean;
begin
  Result := False;
  Why := '';
  if GLib = 0 then
  begin
    Why := OpenLoader(GLib);
    if GLib = 0 then
      Exit;
  end;

  Api.GetInstanceProcAddr :=
    TvkGetInstanceProcAddr(LibSym(GLib, 'vkGetInstanceProcAddr'));
  if not Assigned(Api.GetInstanceProcAddr) then
  begin
    Why := 'vkGetInstanceProcAddr missing';
    Exit;
  end;

  { global commands: instance handle must be nil }
  Api.CreateInstance :=
    TvkCreateInstance(Api.GetInstanceProcAddr(nil, 'vkCreateInstance'));
  if not Assigned(Api.CreateInstance) then
  begin
    Why := 'vkCreateInstance missing';
    Exit;
  end;
  Result := True;
end;

function VkLoadInstanceProcs(var Api: TVulkanApi; Instance: TVkInstance;
  out Why: string): Boolean;
var
  Missing: string;

  function G(const Name: string): Pointer;
  begin
    Result := Api.GetInstanceProcAddr(Instance, PAnsiChar(AnsiString(Name)));
    if Result = nil then
      Missing := Missing + Name + ' ';
  end;

begin
  Missing := '';
  Api.DestroyInstance := TvkDestroyInstance(G('vkDestroyInstance'));
  Api.EnumeratePhysicalDevices :=
    TvkEnumeratePhysicalDevices(G('vkEnumeratePhysicalDevices'));
  Api.GetPhysicalDeviceProperties :=
    TvkGetPhysicalDeviceProperties(G('vkGetPhysicalDeviceProperties'));
  Api.GetPhysicalDeviceMemoryProperties :=
    TvkGetPhysicalDeviceMemoryProperties(
      G('vkGetPhysicalDeviceMemoryProperties'));
  Api.GetPhysicalDeviceQueueFamilyProperties :=
    TvkGetPhysicalDeviceQueueFamilyProperties(
      G('vkGetPhysicalDeviceQueueFamilyProperties'));
  Api.EnumerateDeviceExtensionProperties :=
    TvkEnumerateDeviceExtensionProperties(
      G('vkEnumerateDeviceExtensionProperties'));
  Api.CreateDevice := TvkCreateDevice(G('vkCreateDevice'));
  Api.GetDeviceProcAddr := TvkGetDeviceProcAddr(G('vkGetDeviceProcAddr'));

  Why := Trim(Missing);
  Result := Why = '';
  if not Result then
    Why := 'instance entry points missing: ' + Why;
end;

function VkLoadDeviceProcs(var Api: TVulkanApi; Device: TVkDevice;
  out Why: string): Boolean;
var
  Missing: string;

  function G(const Name: string): Pointer;
  begin
    Result := Api.GetDeviceProcAddr(Device, PAnsiChar(AnsiString(Name)));
    if Result = nil then
      Missing := Missing + Name + ' ';
  end;

begin
  Missing := '';
  Api.DestroyDevice := TvkDestroyDevice(G('vkDestroyDevice'));
  Api.GetDeviceQueue := TvkGetDeviceQueue(G('vkGetDeviceQueue'));
  Api.DeviceWaitIdle := TvkDeviceWaitIdle(G('vkDeviceWaitIdle'));

  Api.CreateBuffer := TvkCreateBuffer(G('vkCreateBuffer'));
  Api.DestroyBuffer := TvkDestroyBuffer(G('vkDestroyBuffer'));
  Api.GetBufferMemoryRequirements :=
    TvkGetBufferMemoryRequirements(G('vkGetBufferMemoryRequirements'));
  Api.AllocateMemory := TvkAllocateMemory(G('vkAllocateMemory'));
  Api.FreeMemory := TvkFreeMemory(G('vkFreeMemory'));
  Api.BindBufferMemory := TvkBindBufferMemory(G('vkBindBufferMemory'));
  Api.MapMemory := TvkMapMemory(G('vkMapMemory'));
  Api.UnmapMemory := TvkUnmapMemory(G('vkUnmapMemory'));

  Api.CreateShaderModule := TvkCreateShaderModule(G('vkCreateShaderModule'));
  Api.DestroyShaderModule :=
    TvkDestroyShaderModule(G('vkDestroyShaderModule'));
  Api.CreateDescriptorSetLayout :=
    TvkCreateDescriptorSetLayout(G('vkCreateDescriptorSetLayout'));
  Api.DestroyDescriptorSetLayout :=
    TvkDestroyDescriptorSetLayout(G('vkDestroyDescriptorSetLayout'));
  Api.CreatePipelineLayout :=
    TvkCreatePipelineLayout(G('vkCreatePipelineLayout'));
  Api.DestroyPipelineLayout :=
    TvkDestroyPipelineLayout(G('vkDestroyPipelineLayout'));
  Api.CreateComputePipelines :=
    TvkCreateComputePipelines(G('vkCreateComputePipelines'));
  Api.DestroyPipeline := TvkDestroyPipeline(G('vkDestroyPipeline'));

  Api.CreateDescriptorPool :=
    TvkCreateDescriptorPool(G('vkCreateDescriptorPool'));
  Api.DestroyDescriptorPool :=
    TvkDestroyDescriptorPool(G('vkDestroyDescriptorPool'));
  Api.AllocateDescriptorSets :=
    TvkAllocateDescriptorSets(G('vkAllocateDescriptorSets'));
  Api.FreeDescriptorSets :=
    TvkFreeDescriptorSets(G('vkFreeDescriptorSets'));
  Api.UpdateDescriptorSets :=
    TvkUpdateDescriptorSets(G('vkUpdateDescriptorSets'));

  Api.CreateCommandPool := TvkCreateCommandPool(G('vkCreateCommandPool'));
  Api.DestroyCommandPool := TvkDestroyCommandPool(G('vkDestroyCommandPool'));
  Api.AllocateCommandBuffers :=
    TvkAllocateCommandBuffers(G('vkAllocateCommandBuffers'));
  Api.FreeCommandBuffers :=
    TvkFreeCommandBuffers(G('vkFreeCommandBuffers'));
  Api.BeginCommandBuffer := TvkBeginCommandBuffer(G('vkBeginCommandBuffer'));
  Api.EndCommandBuffer := TvkEndCommandBuffer(G('vkEndCommandBuffer'));
  Api.ResetCommandBuffer := TvkResetCommandBuffer(G('vkResetCommandBuffer'));

  Api.CmdBindPipeline := TvkCmdBindPipeline(G('vkCmdBindPipeline'));
  Api.CmdBindDescriptorSets :=
    TvkCmdBindDescriptorSets(G('vkCmdBindDescriptorSets'));
  Api.CmdPushConstants := TvkCmdPushConstants(G('vkCmdPushConstants'));
  Api.CmdDispatch := TvkCmdDispatch(G('vkCmdDispatch'));
  Api.CmdCopyBuffer := TvkCmdCopyBuffer(G('vkCmdCopyBuffer'));
  Api.CmdPipelineBarrier :=
    TvkCmdPipelineBarrier(G('vkCmdPipelineBarrier'));

  Api.QueueSubmit := TvkQueueSubmit(G('vkQueueSubmit'));
  Api.QueueWaitIdle := TvkQueueWaitIdle(G('vkQueueWaitIdle'));
  Api.CreateFence := TvkCreateFence(G('vkCreateFence'));
  Api.DestroyFence := TvkDestroyFence(G('vkDestroyFence'));
  Api.WaitForFences := TvkWaitForFences(G('vkWaitForFences'));
  Api.ResetFences := TvkResetFences(G('vkResetFences'));

  Why := Trim(Missing);
  Result := Why = '';
  if not Result then
    Why := 'device entry points missing: ' + Why;
end;

procedure VkUnloadLoader;
begin
  if GLib <> 0 then
  begin
{$IFDEF MSWINDOWS}
    FreeLibrary(HMODULE(GLib));
{$ELSE}
    dlclose(Pointer(GLib));
{$ENDIF}
    GLib := 0;
  end;
end;

end.

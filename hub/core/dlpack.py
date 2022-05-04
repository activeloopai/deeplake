#import ctypes
#
#_c_str_dltensor = b"dltensor"
#
#
#class DLDeviceType(ctypes.c_int):
#    """The enum that encodes the type of the device where
#    DLTensor memory is allocated.
#    """
#    kDLCPU = 1
#    kDLCUDA = 2
#    kDLCUDAHost = 3
#    kDLOpenCL = 4
#    kDLVulkan = 7
#    kDLMetal = 8
#    kDLVPI = 9
#    kDLROCM = 10
#    kDLROCMHost = 11
#    kDLCUDAManaged = 13
#    kDLOneAPI = 14
#
#    def __str__(self):
#        return {
#            self.kDLCPU : "CPU",
#            self.kDLCUDA: "CUDA",
#            self.kDLCUDAHost: "CUDAHost",
#            self.kDLOpenCL: "OpenCL",
#            self.kDLVulkan: "Vulkan",
#            self.kDLMetal: "Metal",
#            self.kDLVPI: "VPI",
#            self.kDLROCM: "ROCM",
#            self.kDLROCMHost: "ROMCHost",
#            self.kDLCUDAManaged: "CUDAManaged",
#            self.kDLOneAPI: "oneAPI",
#            }[self.value]
#
#
#class DLDevice(ctypes.Structure):
#    """Represents the device where DLTensor memory is allocated.
#    The device is represented by the pair of fields:
#       device_type: DLDeviceType
#       device_id: c_int
#    """
#    _fields_ = [
#        ("device_type", DLDeviceType),
#        ("device_id", ctypes.c_int),
#    ]
#
#
#class DLDataTypeCode(ctypes.c_uint8):
#    """An integer that encodes the category of DLTensor elements' data type."""
#    kDLInt = 0
#    kDLUInt = 1
#    kDLFloat = 2
#    kDLOpaquePointer = 3
#    kDLBfloat = 4
#    kDLComplex = 5
#
#    def __str__(self):
#        return {
#            self.kDLInt: "int",
#            self.kDLUInt: "uint",
#            self.kDLFloat: "float",
#            self.kDLBfloat: "bfloat",
#            self.kDLComplex: "complex",
#            self.kDLOpaquePointer: "void_p"
#        }[self.value]
#
#
#class DLDataType(ctypes.Structure):
#    """Descriptor of data type for elements of DLTensor.
#    The data type is described by a triple, `DLDataType.type_code`,
#    `DLDataType.bits`, and `DLDataType.lanes`.
#
#    The element is understood as packed `lanes` repetitions of
#    elements from `type_code` data-category of width `bits`.
#    """
#    _fields_ = [
#        ("type_code", DLDataTypeCode),
#        ("bits", ctypes.c_uint8),
#        ("lanes", ctypes.c_uint16),
#    ]
#    TYPE_MAP = {
#        "bool": (DLDataTypeCode.kDLUInt, 1, 1),
#        "int8": (DLDataTypeCode.kDLInt, 8, 1),
#        "int16": (DLDataTypeCode.kDLInt, 16, 1),
#        "int32": (DLDataTypeCode.kDLInt, 32, 1),
#        "int64": (DLDataTypeCode.kDLInt, 64, 1),
#        "uint8": (DLDataTypeCode.kDLUInt, 8, 1),
#        "uint16": (DLDataTypeCode.kDLUInt, 16, 1),
#        "uint32": (DLDataTypeCode.kDLUInt, 32, 1),
#        "uint64": (DLDataTypeCode.kDLUInt, 64, 1),
#        "float16": (DLDataTypeCode.kDLFloat, 16, 1),
#        "float32": (DLDataTypeCode.kDLFloat, 32, 1),
#        "float64": (DLDataTypeCode.kDLFloat, 64, 1),
#        "complex64": (DLDataTypeCode.kDLComplex, 64, 1),
#        "complex128": (DLDataTypeCode.kDLComplex, 128, 1)
#    }
#
#
#class DLTensor(ctypes.Structure):
#    """Structure describing strided layout of DLTensor.
#    Fields are:
#       data:  void pointer
#       device: DLDevice
#       ndim: number of indices needed to reference an
#             element of the tensor
#       dtype: data type descriptor
#       shape: tuple with lengths of the corresponding
#              tensor dimensions
#       strides: tuple of numbers of array elements to
#                step in each dimension when traversing
#                the tensor
#       byte_offset: data + byte_offset gives the address of
#                tensor element with index (0,) * ndim
#    """
#    _fields_ = [
#        ("data", ctypes.c_void_p),
#        ("device", DLDevice),
#        ("ndim", ctypes.c_int),
#        ("dtype", DLDataType),
#        ("shape", ctypes.POINTER(ctypes.c_int64)),
#        ("strides", ctypes.POINTER(ctypes.c_int64)),
#        ("byte_offset", ctypes.c_uint64),
#    ]
#
#
#class DLManagedTensor(ctypes.Structure):
#    """Structure storing the pointer to the tensor descriptor,
#    deleter callable for the tensor descriptor, and pointer to
#    some additional data. These are stored in fields `dl_tensor`,
#    `deleter`, and `manager_ctx`."""
#    _fields_ = [
#        ("dl_tensor", DLTensor),
#        ("manager_ctx", ctypes.c_void_p),
#        ("deleter", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
#    ]

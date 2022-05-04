# from typing import Callable
# from  hub.core.tensor import Tensor
# import ctypes

# from .dlpack import DLManagedTensor, DLDevice, DLDataType, _c_str_dltensor


# ctypes.pythonapi.PyMem_RawMalloc.restype = ctypes.c_void_p
# ctypes.pythonapi.PyMem_RawFree.argtypes = [ctypes.c_void_p]

# ctypes.pythonapi.PyCapsule_New.restype=ctypes.py_object
# ctypes.pythonapi.PyCapsule_New.argtypes=[ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]


# class _Holder:
#     """A wrapper around a Tensor array to keep track of references to the underlying memory.

#     Parameters
#     ----------
#     tensor : Tensor
#         The tesnsor array that will be converted to a DLPack tensor and must be managed.
#     """

#     def __init__(self, tensor: Tensor) -> None:
#         self.data = tensor.raw_data #, ctypes.c_void_p)
#         #tensor.raw_data().data_as(ctypes.c_void_p)
#         #.data_as(ctypes.c_void_p)
#         self.shape = tensor.shape.shape_as(ctypes.c_int64)
#         #self.strides = tensor.ctypes.strides_as(ctypes.c_int64)
#         #for i in range(tensor.ndim):
#         #    self.strides[i] //= np_array.itemsize

#     def _as_manager_ctx(self) -> ctypes.c_void_p:
#         py_obj = ctypes.py_object(self)
#         py_obj_ptr = ctypes.pointer(py_obj)
#         ctypes.pythonapi.Py_IncRef(py_obj)
#         ctypes.pythonapi.Py_IncRef(ctypes.py_object(py_obj_ptr))
#         return ctypes.cast(py_obj_ptr, ctypes.c_void_p)


# @ctypes.CFUNCTYPE(None, ctypes.c_void_p)
# def _tensor_array_deleter(handle: ctypes.c_void_p) -> None:
#     """A function to deallocate the memory of a numpy array."""
#     dl_managed_tensor = DLManagedTensor.from_address(handle)
#     py_obj_ptr = ctypes.cast(
#         dl_managed_tensor.manager_ctx, ctypes.POINTER(ctypes.py_object)
#     )
#     py_obj = py_obj_ptr.contents
#     ctypes.pythonapi.Py_DecRef(py_obj)
#     ctypes.pythonapi.Py_DecRef(ctypes.py_object(py_obj_ptr))
#     ctypes.pythonapi.PyMem_RawFree(handle)


# @ctypes.CFUNCTYPE(None, ctypes.c_void_p)
# def _tensor_pycapsule_deleter(handle: ctypes.c_void_p) -> None:
#     """A function to deallocate a pycapsule that wraps a numpy array."""
#     pycapsule: ctypes.py_object = ctypes.cast(handle, ctypes.py_object)
#     if ctypes.pythonapi.PyCapsule_IsValid(pycapsule, _c_str_dltensor):
#         dl_managed_tensor = ctypes.pythonapi.PyCapsule_GetPointer(
#             pycapsule, _c_str_dltensor
#         )
#         _tensor_array_deleter(dl_managed_tensor)
#         ctypes.pythonapi.PyCapsule_SetDestructor(pycapsule, None)


# def from_tensor(tensor: Tensor):
#     """Convert a Tensor array to another type of dlpack compatible array.

#     Parameters
#     ----------
#     tensor : Tensor
#         The source tensor array that will be converted.

#     Returns
#     -------
#     pycapsule : PyCapsule
#         A pycapsule containing a DLManagedTensor that can be converted
#         to other array formats without copying the underlying memory.
#     """
#     print("A")
#     holder = _Holder(tensor)
#     # print("B")
#     # size = ctypes.c_size_t(ctypes.sizeof(DLManagedTensor))
#     # print("C")
#     # dl_managed_tensor = DLManagedTensor.from_address(
#     #     ctypes.pythonapi.PyMem_RawMalloc(size)
#     # )
#     # print("Ds")
#     # dl_managed_tensor.dl_tensor.data = holder.data
#     # dl_managed_tensor.dl_tensor.device = DLDevice(1, 0)
#     # dl_managed_tensor.dl_tensor.ndim = tensor.ndim()
#     # dl_managed_tensor.dl_tensor.dtype = DLDataType.TYPE_MAP[str(tensor.dtype())]
#     # dl_managed_tensor.dl_tensor.shape = holder.shape
#     # dl_managed_tensor.dl_tensor.strides = holder.strides
#     # dl_managed_tensor.dl_tensor.byte_offset = 0
#     # dl_managed_tensor.manager_ctx = holder._as_manager_ctx()
#     # dl_managed_tensor.deleter = _tensor_array_deleter
#     # print(5)
#     # pycapsule = ctypes.pythonapi.PyCapsule_New(
#     #     ctypes.byref(dl_managed_tensor),
#     #     _c_str_dltensor,
#     #     _tensor_pycapsule_deleter,
#     # )
#     # return pycapsule

import numpy as np
import numpy.ctypeslib as npc
import ctypes

#https://github.com/numpy/numpy/issues/6239
array_1d_int_base = npc.ndpointer(dtype=np.int32, ndim=1, flags='CONTIGUOUS')

def int_from_param(cls, obj):
    if obj is None:
        return obj
    return array_1d_int_base.from_param(obj)

array_1d_int = type(
    'IntArrayType',
    (array_1d_int_base,),
    {'from_param': classmethod(int_from_param)}
)

#https://github.com/numpy/numpy/issues/6239
array_1d_double_base = npc.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')

def double_from_param(cls, obj):
    if obj is None:
        return obj
    return array_1d_double_base.from_param(obj)

array_1d_double = type(
    'DoubleArrayType',
    (array_1d_double_base,),
    {'from_param': classmethod(double_from_param)}
)

#https://github.com/numpy/numpy/issues/6239
array_2d_int_base = npc.ndpointer(dtype=np.int32, ndim=2, flags='CONTIGUOUS')

def int_2d_from_param(cls, obj):
    if obj is None:
        return obj
    return array_2d_int_base.from_param(obj)

array_2d_int = type(
    'IntArrayType',
    (array_2d_int_base,),
    {'from_param': classmethod(int_2d_from_param)}
)

def ctypes2numpy(cptr, length, dtype=np.int32):
    """
    Convert a ctypes pointer array to a numpy array.
    """
    res = np.zeros(length, dtype=dtype)
    if not ctypes.memmove(res.ctypes.data, cptr, length * res.strides[0]):
        raise RuntimeError('memmove failed')
    return res

def numpy2ctypes(cptr, array):
    if not ctypes.memmove(cptr, array.data[:], len(array.data)):
        raise RuntimeError('memmove failed')
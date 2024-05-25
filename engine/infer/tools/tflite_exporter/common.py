from functools import reduce
from operator import mul

import tflite

BUILTIN_TENSORTYPEINFO = {
    tflite.TensorType.UINT8: [1, 'UINT8'],
    tflite.TensorType.INT8: [1, 'INT8'],
    tflite.TensorType.UINT16: [2, 'UINT16'],
    tflite.TensorType.INT16: [2, 'INT16'],
    tflite.TensorType.UINT32: [4, 'UINT32'],
    tflite.TensorType.INT32: [4, 'INT32'],
    tflite.TensorType.FLOAT32: [4, 'FLOAT32'],
}

def get_tensor_size(tensor):
    size = reduce(mul, tensor.ShapeAsNumpy(), 1)
    size *= BUILTIN_TENSORTYPEINFO[tensor.Type()][0]
    return size
    
def get_tensor_type_name(tensor_type):
    return BUILTIN_TENSORTYPEINFO[tensor_type][1]

def check_value_in_dict(value, dict):
    for key in dict:
        if value == key:
            return True
    return False
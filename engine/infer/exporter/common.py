import sys
import math
import numpy as np
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

# format
def format_tensor_shape(tensor):
    shape_as_numpy = tensor.ShapeAsNumpy()
        
    shape_str = "{"
    for i in range(len(shape_as_numpy) - 1):
        shape_str = shape_str + str(shape_as_numpy[i]) + ", "
    shape_str = shape_str + str(shape_as_numpy[len(shape_as_numpy) - 1]) + "}"
    shape_str = "{{ .dims_count = {0}, .dims = {1} }}".format(str(len(shape_as_numpy)), shape_str)
    return shape_str

def format_tensor_type(tensor_type):
    if tensor_type is tflite.TensorType.FLOAT32:
        return "kPaiInferFloat32"
    elif tensor_type is tflite.TensorType.INT32:
        return "kPaiInferInt32"
    elif tensor_type is tflite.TensorType.UINT32:
        return "kPaiInferUInt32"
    elif tensor_type is tflite.TensorType.INT16:
        return "kPaiInferInt16"
    elif tensor_type is tflite.TensorType.UINT16:
        return "kPaiInferUInt16"
    elif tensor_type is tflite.TensorType.INT8:
        return "kPaiInferInt8"
    elif tensor_type is tflite.TensorType.UINT8:
        return "kPaiInferUInt8"
    else:
        print("Error: format_tensor_type %d is not supported.\n", tensor_type)
        
def format_tensor(tensor, tensor_id, data_name):
    shape_str = format_tensor_shape(tensor)
    tensor_str = "{{ .id = {0}, .type = {1}, .shape = {2}, .data = (void *){3} }}"\
        .format(str(tensor_id), format_tensor_type(tensor.Type()), \
            shape_str, data_name)
    return tensor_str
    
def format_weight_bias(data, type, var_name):
    if type is tflite.TensorType.FLOAT32:
        type_str = "float_t"
    elif type is tflite.TensorType.INT8:
        type_str = "int8_t"
    elif type is tflite.TensorType.INT32:
        type_str = "int32_t"
    else:
        print("Error: format_weight_bias -> Type = %d is not supported!" %type)
    data_carray = ",".join(str(i) for i in data)
    string = "{0} {1}[] __attribute__((aligned(16))) = {{{2}}};\n\n".format(type_str, var_name, data_carray)
    return string, var_name
    
def format_multiplier(data, var_name):
    type_str = "int32_t"
    data_carray = ",".join(str(i) for i in data)
    string = "{0} {1}[] __attribute__((aligned(16))) = {{{2}}};\n\n".format(type_str, var_name, data_carray)
    return string

def export_padding_type(option, op_params):
    padding_mode = option.Padding()
    if padding_mode is tflite.Padding.SAME:
        op_params = op_params.replace('<PaddingType>', 'PaddingType::kSame')
    elif padding_mode is tflite.Padding.VALID:
        op_params = op_params.replace('<PaddingType>', ' PaddingType::kValid')
    else:
        op_params = op_params.replace('<PaddingType>', ' PaddingType::kNone')
    return op_params 

# algo

# tensorflow/lite/kernels/kernel_util.h#L133  CalculateActivationRange
def export_fused_activation_float(option, op_params):
    # .float_activation_min = <float_activation_min>,
    # .float_activation_max = <float_activation_max>
    faf = option.FusedActivationFunction()
    if faf is tflite.ActivationFunctionType.RELU:
        op_params = op_params.replace('<float_activation_min>', str(0))
        op_params = op_params.replace('<float_activation_max>', 'kTfLiteActRelu')
    elif faf is tflite.ActivationFunctionType.RELU6:
        op_params = op_params.replace('<float_activation_min>', str(0))
        op_params = op_params.replace('<float_activation_max>', str(6))
    elif faf is tflite.ActivationFunctionType.RELU_N1_TO_1:
        op_params = op_params.replace('<float_activation_min>', str(-1))
        op_params = op_params.replace('<float_activation_max>', str(1))
    else:
        op_params = op_params.replace('<float_activation_min>', str(sys.float_info.min))
        op_params = op_params.replace('<float_activation_max>', str(sys.float_info.max))
    return op_params
    
# tensorflow/lite/kernels/kernel_util.cc#L244    CalculateActivationRangeQuantized
# 量化时，如为relu，但其数值范围是int8，则仍为-127，128
def export_fused_activation_quant(output_type, op_params):
    # .quantized_activation_min = <quantized_activation_min>,
    # .quantized_activation_max = <quantized_activation_max>,
    if output_type is tflite.TensorType.UINT8:
        op_params = op_params.replace('<quantized_activation_min>', str(0))    # std::numeric_limits<uint8_t>::min()
        op_params = op_params.replace('<quantized_activation_max>', str(255))  # std::numeric_limits<uint8_t>::max()
    elif output_type is tflite.TensorType.INT8:
        op_params = op_params.replace('<quantized_activation_min>', str(-128)) # std::numeric_limits<int8_t>::min()
        op_params = op_params.replace('<quantized_activation_max>', str(127))  # std::numeric_limits<int8_t>::max()
    elif output_type is tflite.TensorType.INT16:
        op_params = op_params.replace('<quantized_activation_min>', str(-32768))  # std::numeric_limits<int16_t>::min()
        op_params = op_params.replace('<quantized_activation_max>', str(32767))   # std::numeric_limits<int16_t>::max()
    else:
        print("Error: export_fused_activation_quant -> output type: %d is not supported.\n", output_type)
    return op_params
    
# The scaling factor from input to output (aka the 'real multiplier') can
# be represented as a fixed point multiplier plus a left shift.
# tensorflow\lite\kernels\internal\quantization_util.cc#L53  QuantizeMultiplier
# tensorflow\lite\kernels\kernel_util.cc#L114
def quantize_multiplier(double_multiplier):
    if double_multiplier == 0.:
        return 0, 0

    q, shift = math.frexp(double_multiplier)
    q_fixed = round(q * (1 << 31))
        
    assert q_fixed <= (1<<31)
    if q_fixed == (1<<31):
        q_fixed /= 2
        shift += 1
    assert q_fixed <= (1<<31)-1 # int32 max
        
    if (shift < -31):
        shift = 0
        q_fixed = 0
        
    return q_fixed, shift

# tensorflow/lite/kernels/padding.h: ComputePaddingHeightWidth
def compute_padding_size(padding_mode, input_size, kernel_size, stride, dilation):
    assert(len(input_size) == len(kernel_size))
    assert(len(input_size) == len(stride))
    assert(len(input_size) == len(dilation))

    # compute output shape
    ones = np.ones_like(input_size)
    effective_filter_size = np.add(np.multiply(np.subtract(kernel_size, ones), dilation), ones)
    if padding_mode is tflite.Padding.SAME:
        oshape = np.divide(np.subtract(np.add(input_size, stride), ones), stride)
    elif padding_mode is tflite.Padding.VALID:
        oshape = np.divide(np.subtract(np.add(input_size, stride), effective_filter_size), stride)
    else:
        raise ValueError("Unknown padding mode!")
    oshape = oshape.astype('int')

    # infer the padding
    total_padding = np.add(np.multiply(np.subtract(oshape, ones), stride),
                        np.subtract(effective_filter_size, input_size))
    total_padding = np.maximum(total_padding, np.zeros_like(input_size))
    total_padding = total_padding.astype('int')

    # ONNX semantic
    pre_padding = total_padding // 2 # [padding_values.height, padding_values.width]
    post_padding = np.subtract(total_padding, pre_padding) # [padding_values.height_offset, padding_values.width_offset]
    padding = np.concatenate((pre_padding, post_padding))

    return padding.flatten()

def export_multiplier_per_tensor(input_tensor, output_tensor, weights_tensor, op_params):
    input_scale = input_tensor.Quantization().Scale(0)
    output_scale = output_tensor.Quantization().Scale(0)
        
    weight_scale = weights_tensor.Quantization().ScaleAsNumpy()
    assert(weight_scale.size, 1)
    effective_output_scale = input_scale * weight_scale[0] / output_scale
    # scale 等效于 multiplier 和 shift，用整型计算代替浮点计算
    output_multiplier, output_shift = quantize_multiplier(effective_output_scale)
    op_params = op_params.replace('<output_multiplier>', str(output_multiplier))
    op_params = op_params.replace('<output_shift>', str(output_shift))
    return op_params
    
def export_multiplier_per_channel(input_tensor, output_tensor, weights_tensor, name_prefix, op_id, fp, op_params):
    input_scale = input_tensor.Quantization().Scale(0)
    output_scale = output_tensor.Quantization().Scale(0)
    weight_scale = weights_tensor.Quantization().ScaleAsNumpy()
    
    outputs_multiplier = []
    outputs_shift = []
    for ch in range(weight_scale.size):
        effective_output_scale = input_scale * weight_scale[ch] / output_scale
        # scale 等效于 multiplier 和 shift，用整型计算代替浮点计算
        output_multiplier, output_shift = quantize_multiplier(effective_output_scale)
        outputs_multiplier.append(output_multiplier)
        outputs_shift.append(output_shift)
    
    m_str = format_multiplier(outputs_multiplier, name_prefix + "_outputs_multiplier_" + str(op_id))
    s_str = format_multiplier(outputs_shift, name_prefix + "_output_shift_" + str(op_id))
    # print("outputs_multiplier:", outputs_multiplier, ", outputs_shift:", outputs_shift) 
    fp["params"].write(m_str)
    fp["params"].write(s_str)
    op_params = op_params.replace('<output_multiplier>', name_prefix + "_outputs_multiplier_" + str(op_id))
    op_params = op_params.replace('<output_shift>', name_prefix + "_output_shift_" + str(op_id))
    return op_params
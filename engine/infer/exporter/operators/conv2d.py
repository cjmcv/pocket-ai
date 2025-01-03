import numpy as np
import tflite

import exporter.common as tfcom
from exporter.operators.operator import Operator

class Conv2D(Operator):
    header_quant = '#include "engine/infer/kernels/conv_per_channel.hpp"\n'
    header_float = '#include "engine/infer/kernels/conv.hpp"\n'

    def __init__(self, graph, op, op_id):
        super().__init__(graph, op, op_id)
        self.attr["code"] = tflite.BuiltinOperator.CONV_2D
        
        self.attr["input_index"] = [0]
        self.attr["weight_index"] = 1
        self.attr["bias_index"] = 2
        self.attr["output_index"] = [0]
    
    def export_common(self, fp, model, io_tensors, name_prefix, op_params):
        op_params = op_params.replace('<op_id>', str(self.id))
        # io tensors
        op_params, input_tensor, output_tensor = self.export_io_tensors(name_prefix, op_params, io_tensors, False, fp)
        # weight
        op_params, weights_tensor = self.export_weight(self.is_quant(), name_prefix, model, op_params, fp)
        # bias
        assert(self.op.InputsLength() == 3) # bias must exist
        op_params, bias_tensor = self.export_bias(self.is_quant(), name_prefix, model, op_params, fp)
        
        op_opt = self.op.BuiltinOptions()
        option = tflite.Conv2DOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)
        
        stride_width = option.StrideW()
        stride_height = option.StrideH()
        dilation_width_factor = option.DilationWFactor()
        dilation_height_factor = option.DilationHFactor()
        op_params = op_params.replace('<stride_width>', str(stride_width))
        op_params = op_params.replace('<stride_height>', str(stride_height))
        op_params = op_params.replace('<dilation_width_factor>', str(dilation_width_factor))
        op_params = op_params.replace('<dilation_height_factor>', str(dilation_height_factor))
        
        # Padding
        tfcom.export_padding_type(option, op_params)
        input_height, input_width = input_tensor.ShapeAsNumpy()[1:3]
        weights_height, weights_width = weights_tensor.ShapeAsNumpy()[1:3]
        padding_size = tfcom.compute_padding_size(option.Padding(), [input_height, input_width],
                                    [weights_height, weights_width],
                                    [stride_height, stride_width], 
                                    [dilation_height_factor, dilation_width_factor])
        padding_size_str = '{ .width = ' + str(padding_size[1]) + ", .height = " + str(padding_size[0]) + '}'
        op_params = op_params.replace('<PaddingValues>', padding_size_str)
        return op_params, input_tensor, output_tensor, weights_tensor, option
        
    def export_float(self, fp, model, dynamic_buffer):
        op_params = \
'''
ConvParams conv_params_<op_id> = {
    .op_id = <op_id>,
    
    // common
    .padding_values = <PaddingValues>,
    .stride_height = <stride_height>,
    .stride_width = <stride_width>,
    .dilation_height_factor = <dilation_height_factor>,
    .dilation_width_factor = <dilation_width_factor>,
    // float
    .float_activation_min = <float_activation_min>,
    .float_activation_max = <float_activation_max>,
    //
    .filter_tensor = <filter_tensor>,
    .bias_tensor = <bias_tensor>,
    //
    .input_tensor = <input_tensor_ptr>,
    .output_tensor = <output_tensor_ptr>,
    // <scratch_buffer_size> = ((output_height * output_width * input_channel * kernel_height * kernel_width) + (output_height* output_width* output_channel)) * sizeof(float)
    .temp_buffer = nullptr, 
};
'''
        name_prefix = 'conv' 
        self.oprun_str = "Conv(conv_params_{0});".format(str(self.id))
        op_params, input_tensor, output_tensor, weights_tensor, option = \
            self.export_common(fp, model, dynamic_buffer.io_tensors, name_prefix, op_params)
        
        assert(output_tensor.Type() == tflite.TensorType.FLOAT32)
        op_params = tfcom.export_activation_range_float(option, op_params)
        
        output_height = output_tensor.ShapeAsNumpy()[1]
        output_width = output_tensor.ShapeAsNumpy()[2]
        output_channel = output_tensor.ShapeAsNumpy()[3]
        input_channel = input_tensor.ShapeAsNumpy()[-1]
        kernel_height = weights_tensor.ShapeAsNumpy()[1]
        kernel_width = weights_tensor.ShapeAsNumpy()[2]
        scratch_buffer_size = output_height * output_width * input_channel * kernel_height * kernel_width * 4 # For im2col, 4: sizeof(float)
        str_prefix = "conv_params_" + str(self.id)
        op_params = tfcom.export_scratch_buffer(dynamic_buffer, str_prefix, scratch_buffer_size, op_params)

        return op_params
        
    def export_quant(self, fp, model, dynamic_buffer):
        # ConvParams
        op_params = \
'''
ConvPerChannelParams conv_params_<op_id> = {
    .op_id = <op_id>,
    // common
    .padding_values = <PaddingValues>,
    .stride_height = <stride_height>,
    .stride_width = <stride_width>,
    .dilation_height_factor = <dilation_height_factor>,
    .dilation_width_factor = <dilation_width_factor>,
    // int8
    .input_offset = <input_offset>,
    //.weights_offset = <weights_offset>,
    .output_offset = <output_offset>,
    .output_multiplier = <output_multiplier>,
    .output_shift = <output_shift>,
    .quantized_activation_min = <quantized_activation_min>,
    .quantized_activation_max = <quantized_activation_max>,
    //
    .filter_tensor = <filter_tensor>,
    .bias_tensor = <bias_tensor>,
    //
    .input_tensor = <input_tensor_ptr>,
    .output_tensor = <output_tensor_ptr>,
    // <scratch_buffer_size> = ((output_height * output_width * input_channel * kernel_height * kernel_width) * sizeof(int8_t) + (output_height* output_width* output_channel)) * sizeof(int32_t)
    .temp_buffer = nullptr, 
};
'''
        name_prefix = 'conv' 
        self.oprun_str = "ConvPerChannel(conv_params_{0});".format(str(self.id))

        op_params, input_tensor, output_tensor, weights_tensor, option = \
            self.export_common(fp, model, dynamic_buffer.io_tensors, name_prefix, op_params)

        assert(output_tensor.Type() == tflite.TensorType.INT8)
        input_zero_point = input_tensor.Quantization().ZeroPoint(0)
        op_params = op_params.replace('<input_offset>', str(-input_zero_point)) # tensorflow\lite\micro\kernels\conv_common.cc: ConvParamsQuantized
        weights_zero_point = weights_tensor.Quantization().ZeroPoint(0)
        op_params = op_params.replace('<weights_offset>', str(-weights_zero_point))
        output_zero_point = output_tensor.Quantization().ZeroPoint(0)
        op_params = op_params.replace('<output_offset>', str(output_zero_point))
            
        op_params = tfcom.export_multiplier_per_channel(True, input_tensor, output_tensor, weights_tensor, 
                                                            name_prefix, self.id, fp, op_params)

        op_params = tfcom.export_activation_range_quant(output_tensor.Type(), op_params)
        

        # ((output_height * output_width * input_channel * kernel_height * kernel_width) + (output_height* output_width* output_channel)) * sizeof(int32_t)
        output_height, output_width, output_channel = output_tensor.ShapeAsNumpy()[1:4]
        input_channel = input_tensor.ShapeAsNumpy()[3]
        kernel_height, kernel_width = weights_tensor.ShapeAsNumpy()[1:3]
        scratch_buffer_size0 = output_height * output_width * input_channel * kernel_height * kernel_width * 1 # For im2col, 1: sizeof(int8_t)
        scratch_buffer_size1 = output_height * output_width* output_channel * 4                                # For gemm int32 output, 4: sizeof(int32_t)
        scratch_buffer_size = scratch_buffer_size0 + scratch_buffer_size1 
        str_prefix = "conv_params_" + str(self.id)
        op_params = tfcom.export_scratch_buffer(dynamic_buffer, str_prefix, scratch_buffer_size, op_params)

        return op_params
        
    def export(self, fp, model, dynamic_buffer):
        self.scan_iotensor_lifetime(dynamic_buffer)
        if self.is_quant():
            op_params = self.export_quant(fp, model, dynamic_buffer)
        else:
            op_params = self.export_float(fp, model, dynamic_buffer)
        fp["model"].write(op_params+"\n")
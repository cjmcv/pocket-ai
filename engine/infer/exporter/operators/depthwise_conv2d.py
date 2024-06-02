import numpy as np
import tflite

import exporter.common as tfcom
from exporter.operators.conv2d import Conv2D
from exporter.operators.operator import Operator

class DepthwiseConv2D(Operator):
    header_quant = '#include "engine/infer/kernels/depthwise_conv_per_channel.hpp"\n'
    header_float = '#include "engine/infer/kernels/depthwise_conv.hpp"\n'

    def __init__(self, graph, op, op_id):
        super().__init__(graph, op, op_id)
        self.attr["code"] = tflite.BuiltinOperator.DEPTHWISE_CONV_2D
        
        self.attr["input_index"] = [0]
        self.attr["weight_index"] = 1
        self.attr["bias_index"] = 2
        self.attr["output_index"] = [0]
    
    def export_float(self, fp, model, io_tensors):
        op_params = \
'''
DepthwiseParams conv_params_<op_id> = {
    .padding_values = <PaddingValues>,
    .stride_width = <stride_width>,
    .stride_height = <stride_height>,
    .dilation_width_factor = <dilation_width_factor>,
    .dilation_height_factor = <dilation_height_factor>,

    .float_activation_min = <float_activation_min>,
    .float_activation_max = <float_activation_max>,
};
'''
    
    def export_quant(self, fp, model, io_tensors):
        # ConvParams
        op_params = \
'''
DepthwisePerChannelParams depthwise_conv_params_<op_id> = {
    .op_id = <op_id>,
    
    .padding_values = <PaddingValues>,
    .stride_height = <stride_height>,
    .stride_width = <stride_width>,
    .dilation_height_factor = <dilation_height_factor>,
    .dilation_width_factor = <dilation_width_factor>,
    .depth_multiplier = <depth_multiplier>,
    
    .input_offset = <input_offset>,
    //.weights_offset = <weights_offset>,
    .output_offset = <output_offset>,
    .output_multiplier = <output_multiplier>,
    .output_shift = <output_shift>,
    // 
    .quantized_activation_min = <quantized_activation_min>,
    .quantized_activation_max = <quantized_activation_max>,
    //
    .filter_tensor = <filter_tensor>,
    .bias_tensor = <bias_tensor>,
    //
    .input_tensor = <input_tensor_ptr>,
    .output_tensor = <output_tensor_ptr>,
};
'''
        name_prefix = 'depthwise_conv'
        self.oprun_str = "DepthwiseConvPerChannel(depthwise_conv_params_{0});".format(str(self.id))
        op_params = op_params.replace('<op_id>', str(self.id))
        
        # io tensors
        op_params, input_tensor, output_tensor = self.export_io_tensors(name_prefix, op_params, io_tensors, False, fp)
        # weight
        op_params, weights_tensor = self.export_weight_quant(name_prefix, model, op_params, fp)
        # bias
        assert(self.op.InputsLength(), 2) # bias must exist
        op_params, bias_tensor = self.export_bias_quant(name_prefix, model, op_params, fp)
        
        op_opt = self.op.BuiltinOptions()
        option = tflite.DepthwiseConv2DOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)
        
        stride_width = option.StrideW()
        stride_height = option.StrideH()
        dilation_width_factor = option.DilationWFactor()
        dilation_height_factor = option.DilationHFactor()
        depth_multiplier = option.DepthMultiplier()
        op_params = op_params.replace('<stride_width>', str(stride_width))
        op_params = op_params.replace('<stride_height>', str(stride_height))
        op_params = op_params.replace('<dilation_width_factor>', str(dilation_width_factor))
        op_params = op_params.replace('<dilation_height_factor>', str(dilation_height_factor))
        op_params = op_params.replace('<depth_multiplier>', str(depth_multiplier))
        
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
        
        weight_scale = weights_tensor.Quantization().ScaleAsNumpy()
        if output_tensor.Type() is tflite.TensorType.FLOAT32:
            op_params = tfcom.export_fused_activation_float(option, op_params)
        else:
            input_zero_point = input_tensor.Quantization().ZeroPoint(0)
            op_params = op_params.replace('<input_offset>', str(-input_zero_point)) # tensorflow\lite\micro\kernels\conv_common.cc: ConvParamsQuantized
            output_zero_point = output_tensor.Quantization().ZeroPoint(0)
            op_params = op_params.replace('<output_offset>', str(output_zero_point))
            
            input_scale = input_tensor.Quantization().Scale(0)
            output_scale = output_tensor.Quantization().Scale(0)
            outputs_multiplier = []
            outputs_shift = []
            for ch in range(weight_scale.size):
                effective_output_scale = input_scale * weight_scale[ch] / output_scale
                # scale 等效于 multiplier 和 shift，用整型计算代替浮点计算
                output_multiplier, output_shift = tfcom.quantize_multiplier(effective_output_scale)
                outputs_multiplier.append(output_multiplier)
                outputs_shift.append(output_shift)
            
            m_str = tfcom.format_multiplier(outputs_multiplier, name_prefix + "_outputs_multiplier_" + str(self.id))
            s_str = tfcom.format_multiplier(outputs_shift, name_prefix + "_output_shift_" + str(self.id))
            # print("outputs_multiplier:", outputs_multiplier, ", outputs_shift:", outputs_shift) 
            fp["params"].write(m_str)
            fp["params"].write(s_str)
            op_params = op_params.replace('<output_multiplier>', name_prefix + "_outputs_multiplier_" + str(self.id))
            op_params = op_params.replace('<output_shift>', name_prefix + "_output_shift_" + str(self.id))            
            op_params = tfcom.export_fused_activation_quant(output_tensor.Type(), op_params)
        
        return op_params
        
    def export(self, fp, model, io_tensors):
        if self.is_quant():
            op_params = self.export_quant(fp, model, io_tensors)
        else:
            op_params = self.export_float(fp, model, io_tensors)
        fp["model"].write(op_params+"\n")
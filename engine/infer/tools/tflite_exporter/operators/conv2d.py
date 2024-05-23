
import sys
import numpy as np
import math
import tflite

from operators.operator import Operator

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
    
    def export_float(self, fp, model):
        op_params = \
'''
ConvParams conv_params_<op_id> = {
    .padding_values = <PaddingValues>,
    .stride_width = <stride_width>,
    .stride_height = <stride_height>,
    .dilation_width_factor = <dilation_width_factor>,
    .dilation_height_factor = <dilation_height_factor>,

    .float_activation_min = <float_activation_min>,
    .float_activation_max = <float_activation_max>,
};
'''
        # op_params = op_params.replace('<op_id>', str(self.op_id))
        
        # input_tensor = self.graph.Tensors(self.op.Inputs(self.attr["input_index"][0]))
        # input_height, input_width = input_tensor.ShapeAsNumpy()[1:3]
        
        
        # weights_tensor = self.graph.Tensors(self.op.Inputs(self.attr["weight_index"]))
        # weights_buffer = model.Buffers(weights_tensor.Buffer())
        # if weights_tensor.Type() == tflite.TensorType.FLOAT32:
        #     weight_data = np.frombuffer(weights_buffer.DataAsNumpy(), dtype=np.float32)
        # elif weights_tensor.Type() == tflite.TensorType.INT8:
        #     weight_data = np.frombuffer(weights_buffer.DataAsNumpy(), dtype=np.int8)
        #     weight_scale = weights_tensor.Quantization().ScaleAsNumpy() # .Scale(0)
        #     weight_zero_point = weights_tensor.Quantization().ZeroPointAsNumpy() #.ZeroPoint(0)
        #     op_params = op_params.replace('<weights_offset>', str(weight_zero_point))
            
        # weight_str, weight_name = self.format_weight_bias(weight_data, weights_tensor.Type(), "conv_weights")
        # fp["params"].write(weight_str)
        
        # print("weight: ", weights_tensor.ShapeAsNumpy())
        # print(weight_data, weight_scale, weight_zero_point)
        # weights_height, weights_width = weights_tensor.ShapeAsNumpy()[1:3]
        
        # if self.op.InputsLength() > 2:
        #     bias_tensor = self.graph.Tensors(self.op.Inputs(self.attr["bias_index"]))
        #     bias_buffer = model.Buffers(bias_tensor.Buffer())
        #     if bias_tensor.Type() == tflite.TensorType.FLOAT32:
        #         bias_data = np.frombuffer(bias_buffer.DataAsNumpy(), dtype=np.float32)
        #     elif bias_tensor.Type() == tflite.TensorType.INT32:
        #         bias_data = np.frombuffer(bias_buffer.DataAsNumpy(), dtype=np.int32) 
        #         bias_scale = bias_tensor.Quantization().ScaleAsNumpy()
        #         bias_zero_point = bias_tensor.Quantization().ZeroPointAsNumpy()
        #     else:
        #         print("Error: bias_tensor.Type(): %d is unsupported!\n", bias_tensor.Type())
                
        #     bias_str = self.format_weight_bias(bias_data, bias_tensor.Type(), "conv_bias")
        #     fp["params"].write(bias_str)
               
        #     print("bias: ", bias_tensor.ShapeAsNumpy())
        #     print(bias_data, bias_scale, bias_zero_point)
            

        # op_opt = self.op.BuiltinOptions()
        # option = tflite.Conv2DOptions()
        # option.Init(op_opt.Bytes, op_opt.Pos)
        
        # stride_width = option.StrideW()
        # stride_height = option.StrideH()
        # dilation_width_factor = option.DilationWFactor()
        # dilation_height_factor = option.DilationHFactor()
        # op_params = op_params.replace('<stride_width>', str(stride_width))
        # op_params = op_params.replace('<stride_height>', str(stride_height))
        # op_params = op_params.replace('<dilation_width_factor>', str(dilation_width_factor))
        # op_params = op_params.replace('<dilation_height_factor>', str(dilation_height_factor))
        
        # # Padding
        # self.export_padding_type(option, op_params)
        # padding_size = self.computePaddingSize(option.Padding(), [input_height, input_width],
        #                             [weights_height, weights_width],
        #                             [stride_height, stride_width], 
        #                             [dilation_height_factor, dilation_width_factor])
        # padding_size_str = '{ .width = ' + str(padding_size[1]) + ", .height = " + str(padding_size[0]) + '}'
        # op_params = op_params.replace('<PaddingValues>', padding_size_str)
        
        # output_tensor = self.graph.Tensors(self.op.Outputs(self.attr["output_index"][0]))
        # if output_tensor.Type() is tflite.TensorType.FLOAT32:
        #     op_params = self.export_fused_activation_float(option, op_params)
        # else:
        #     input_zero_point = input_tensor.Quantization().ZeroPoint(0)
        #     op_params = op_params.replace('<input_offset>', str(input_zero_point))
        #     output_zero_point = output_tensor.Quantization().ZeroPoint(0)
        #     op_params = op_params.replace('<output_offset>', str(output_zero_point))
            
        #     input_scale = input_tensor.Quantization().Scale(0)
        #     output_scale = output_tensor.Quantization().Scale(0)
        #     outputs_multiplier = []
        #     outputs_shift = []
        #     for ch in range(weight_scale.size):
        #         effective_output_scale = input_scale * weight_scale[ch] / output_scale
        #         # scale 等效于 multiplier 和 shift，用整型计算代替浮点计算
        #         output_multiplier, output_shift = self.quantize_multiplier(effective_output_scale)
        #         outputs_multiplier.append(output_multiplier)
        #         outputs_shift.append(output_shift)
            
        #     m_str = self.format_multiplier(outputs_multiplier, "conv_outputs_multiplier")
        #     s_str = self.format_multiplier(outputs_shift, "conv_output_shift")
        #     print("outputs_multiplier:", outputs_multiplier, ", outputs_shift:", outputs_shift) 
        #     fp["params"].write(m_str)
        #     fp["params"].write(s_str)
        #     op_params = op_params.replace('<output_multiplier>', "conv_outputs_multiplier_" + str(self.op_id))
        #     op_params = op_params.replace('<output_shift>', "conv_output_shift_" + str(self.op_id))            
        #     op_params = self.export_fused_activation_quant(output_tensor.Type(), op_params)
            
        # fp["model"].write(op_params)


    def export_quant(self, fp, model):
        op_params = \
'''
ConvPerChannelParams conv_params_<op_id> = {
    .padding_values = <PaddingValues>,
    // TODO(starka): This was just "stride", so check that width+height is OK.
    .stride_width = <stride_width>,
    .stride_height = <stride_height>,
    .dilation_width_factor = <dilation_width_factor>,
    .dilation_height_factor = <dilation_height_factor>,
    // uint8_t inference params.
    .input_offset = <input_offset>,
    //.weights_offset = <weights_offset>,
    .output_offset = <output_offset>,
    .output_multiplier = <output_multiplier>,
    .output_shift = <output_shift>,
    // uint8_t, etc, activation params.
    .quantized_activation_min = <quantized_activation_min>,
    .quantized_activation_max = <quantized_activation_max>,
    //
    .filter_shape = <filter_shape>,
    .filter_data = <filter_data>,
    .bias_shape = <bias_shape>,
    .bias_data = <bias_data>
};
'''
        op_params = op_params.replace('<op_id>', str(self.op_id))
        
        input_tensor = self.graph.Tensors(self.op.Inputs(self.attr["input_index"][0]))
        input_height, input_width = input_tensor.ShapeAsNumpy()[1:3]
        
        
        weights_tensor = self.graph.Tensors(self.op.Inputs(self.attr["weight_index"]))
        weights_buffer = model.Buffers(weights_tensor.Buffer())
        if weights_tensor.Type() == tflite.TensorType.FLOAT32:
            weight_data = np.frombuffer(weights_buffer.DataAsNumpy(), dtype=np.float32)
        elif weights_tensor.Type() == tflite.TensorType.INT8:
            weight_data = np.frombuffer(weights_buffer.DataAsNumpy(), dtype=np.int8)
            weight_scale = weights_tensor.Quantization().ScaleAsNumpy() # .Scale(0)
            weight_zero_point = weights_tensor.Quantization().ZeroPointAsNumpy() #.ZeroPoint(0)
            op_params = op_params.replace('<weights_offset>', str(weight_zero_point))
            
        weight_str, weight_var_name = self.format_weight_bias(weight_data, weights_tensor.Type(), "conv_weights")
        fp["params"].write(weight_str)
        op_params = op_params.replace('<filter_data>', weight_var_name)
        
        weight_shape_str = self.format_tensor_shape(weights_tensor)
        op_params = op_params.replace('<filter_shape>', weight_shape_str)
        
        print(weight_data, weight_scale, weight_zero_point)
        weights_height, weights_width = weights_tensor.ShapeAsNumpy()[1:3]
        
        if self.op.InputsLength() > 2:
            bias_tensor = self.graph.Tensors(self.op.Inputs(self.attr["bias_index"]))
            bias_buffer = model.Buffers(bias_tensor.Buffer())
            if bias_tensor.Type() == tflite.TensorType.FLOAT32:
                bias_data = np.frombuffer(bias_buffer.DataAsNumpy(), dtype=np.float32)
            elif bias_tensor.Type() == tflite.TensorType.INT32:
                bias_data = np.frombuffer(bias_buffer.DataAsNumpy(), dtype=np.int32) 
                bias_scale = bias_tensor.Quantization().ScaleAsNumpy()
                bias_zero_point = bias_tensor.Quantization().ZeroPointAsNumpy()
            else:
                print("Error: bias_tensor.Type(): %d is unsupported!\n", bias_tensor.Type())
                
            bias_str, bias_var_name = self.format_weight_bias(bias_data, bias_tensor.Type(), "conv_bias")
            op_params = op_params.replace('<bias_data>', bias_var_name)
            fp["params"].write(bias_str)

            bias_shape_str = self.format_tensor_shape(bias_tensor)
            op_params = op_params.replace('<bias_shape>', bias_shape_str)
        
            print(bias_data, bias_scale, bias_zero_point)
            

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
        self.export_padding_type(option, op_params)
        padding_size = self.compute_padding_size(option.Padding(), [input_height, input_width],
                                    [weights_height, weights_width],
                                    [stride_height, stride_width], 
                                    [dilation_height_factor, dilation_width_factor])
        padding_size_str = '{ .width = ' + str(padding_size[1]) + ", .height = " + str(padding_size[0]) + '}'
        op_params = op_params.replace('<PaddingValues>', padding_size_str)
        
        output_tensor = self.graph.Tensors(self.op.Outputs(self.attr["output_index"][0]))
        if output_tensor.Type() is tflite.TensorType.FLOAT32:
            op_params = self.export_fused_activation_float(option, op_params)
        else:
            input_zero_point = input_tensor.Quantization().ZeroPoint(0)
            op_params = op_params.replace('<input_offset>', str(input_zero_point))
            output_zero_point = output_tensor.Quantization().ZeroPoint(0)
            op_params = op_params.replace('<output_offset>', str(output_zero_point))
            
            input_scale = input_tensor.Quantization().Scale(0)
            output_scale = output_tensor.Quantization().Scale(0)
            outputs_multiplier = []
            outputs_shift = []
            for ch in range(weight_scale.size):
                effective_output_scale = input_scale * weight_scale[ch] / output_scale
                # scale 等效于 multiplier 和 shift，用整型计算代替浮点计算
                output_multiplier, output_shift = self.quantize_multiplier(effective_output_scale)
                outputs_multiplier.append(output_multiplier)
                outputs_shift.append(output_shift)
            
            m_str = self.format_multiplier(outputs_multiplier, "conv_outputs_multiplier")
            s_str = self.format_multiplier(outputs_shift, "conv_output_shift")
            print("outputs_multiplier:", outputs_multiplier, ", outputs_shift:", outputs_shift) 
            fp["params"].write(m_str)
            fp["params"].write(s_str)
            op_params = op_params.replace('<output_multiplier>', "conv_outputs_multiplier_" + str(self.op_id))
            op_params = op_params.replace('<output_shift>', "conv_output_shift_" + str(self.op_id))            
            op_params = self.export_fused_activation_quant(output_tensor.Type(), op_params)
            
        fp["model"].write(op_params)
                 
    def export(self, fp, model):
        if self.is_quant():
            self.export_quant(fp, model)
        else:
            self.export_float(fp, model)
import numpy as np
import tflite

import exporter.common as tfcom
from exporter.operators.operator import Operator

class FullyConnected(Operator):
    header_quant = '#include "engine/infer/kernels/fully_connected_quant.hpp"\n'
    header_float = '#include "engine/infer/kernels/fully_connected.hpp"\n'

    def __init__(self, graph, op, op_id):
        super().__init__(graph, op, op_id)
        self.attr["code"] = tflite.BuiltinOperator.FULLY_CONNECTED
        
        self.attr["input_index"] = [0]
        self.attr["weight_index"] = 1
        self.attr["bias_index"] = 2
        self.attr["output_index"] = [0]
    
    def export_float(self, fp, model, io_tensors):
        op_params = \
'''
FullyConnectedPerChannelParams fully_connected_params_<op_id> = {
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
FullyConnectedQuantParams fully_connected_params_<op_id> = {
    .op_id = <op_id>,
    
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
        name_prefix = 'fully_connected'
        op_params = op_params.replace('<op_id>', str(self.id))
        
        input_tensor_id = self.op.Inputs(self.attr["input_index"][0])
        input_tensor = self.graph.Tensors(input_tensor_id)
        op_params = tfcom.write_io_tensor(name_prefix, 'input', self.id, input_tensor, input_tensor_id, io_tensors, op_params, fp["model"])

        output_tensor_id = self.op.Outputs(self.attr["output_index"][0])
        output_tensor = self.graph.Tensors(output_tensor_id)
        op_params = tfcom.write_io_tensor(name_prefix, 'output', self.id, output_tensor, output_tensor_id, io_tensors, op_params, fp["model"])
        
        # weight
        weights_tensor_id = self.op.Inputs(self.attr["weight_index"])
        weights_tensor = self.graph.Tensors(weights_tensor_id)
        weights_buffer = model.Buffers(weights_tensor.Buffer())
        assert(weights_tensor.Type(), tflite.TensorType.INT8)
        
        weight_data = np.frombuffer(weights_buffer.DataAsNumpy(), dtype=np.int8)
        weight_scale = weights_tensor.Quantization().ScaleAsNumpy() # .Scale(0)
        weight_zero_point = weights_tensor.Quantization().ZeroPointAsNumpy() #.ZeroPoint(0)
        op_params = op_params.replace('<weights_offset>', str(weight_zero_point))
            
        weight_str, weight_var_name = tfcom.format_weight_bias(weight_data, weights_tensor.Type(), name_prefix + "_weights_" + str(self.id))
        fp["params"].write(weight_str)

        filter_tensor_str = tfcom.format_tensor(weights_tensor, weights_tensor_id, weight_var_name)
        op_params = op_params.replace('<filter_tensor>', filter_tensor_str)
        
        # bias
        assert(self.op.InputsLength(), 2) # bias must exist
        
        bias_tensor_id = self.op.Inputs(self.attr["bias_index"])
        bias_tensor = self.graph.Tensors(self.op.Inputs(self.attr["bias_index"]))
        bias_buffer = model.Buffers(bias_tensor.Buffer())

        assert(bias_tensor.Type(), tflite.TensorType.INT32)
        bias_data = np.frombuffer(bias_buffer.DataAsNumpy(), dtype=np.int32) 

        bias_str, bias_var_name = tfcom.format_weight_bias(bias_data, bias_tensor.Type(), name_prefix + "_bias_" + str(self.id))
        fp["params"].write(bias_str)

        bias_tensor_str = tfcom.format_tensor(bias_tensor, bias_tensor_id, bias_var_name)
        op_params = op_params.replace('<bias_tensor>', bias_tensor_str)
    
        # print(bias_data, bias_scale, bias_zero_point)
            
        assert(output_tensor.Type(), tflite.TensorType.INT8)
        input_zero_point = input_tensor.Quantization().ZeroPoint(0)
        op_params = op_params.replace('<input_offset>', str(-input_zero_point)) # FullyConnectedParamsQuantized
        output_zero_point = output_tensor.Quantization().ZeroPoint(0)
        op_params = op_params.replace('<output_offset>', str(output_zero_point))
        
        input_scale = input_tensor.Quantization().Scale(0)
        output_scale = output_tensor.Quantization().Scale(0)
        
        assert(weight_scale.size, 1)
        effective_output_scale = input_scale * weight_scale[0] / output_scale
        # scale 等效于 multiplier 和 shift，用整型计算代替浮点计算
        output_multiplier, output_shift = tfcom.quantize_multiplier(effective_output_scale)
        op_params = op_params.replace('<output_multiplier>', str(output_multiplier))
        op_params = op_params.replace('<output_shift>', str(output_shift))
        op_params = tfcom.export_fused_activation_quant(output_tensor.Type(), op_params)
            
        self.oprun_str = "FullyConnected({0}_params_{1});".format(name_prefix, str(self.id))
        
        return op_params
        
    def export(self, fp, model, io_tensors):
        if self.is_quant():
            op_params = self.export_quant(fp, model, io_tensors)
        else:
            op_params = self.export_float(fp, model, io_tensors)
        fp["model"].write(op_params+"\n")
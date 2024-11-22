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
    
    def export_common(self, fp, model, io_tensors, name_prefix, op_params):
        op_params = op_params.replace('<op_id>', str(self.id))
        # io tensors
        op_params, input_tensor, output_tensor = self.export_io_tensors(name_prefix, op_params, io_tensors, False, fp)
        # weight
        op_params, weights_tensor = self.export_weight(self.is_quant(), name_prefix, model, op_params, fp)
        # bias
        assert(self.op.InputsLength() == 3) # bias must exist
        op_params, bias_tensor = self.export_bias(self.is_quant(), name_prefix, model, op_params, fp)

        return op_params, input_tensor, output_tensor, weights_tensor
    
    def export_float(self, fp, model, dynamic_buffer):
        op_params = \
'''
FullyConnectedParams fully_connected_params_<op_id> = {
    .op_id = <op_id>,
    
    .float_activation_min = <float_activation_min>,
    .float_activation_max = <float_activation_max>,

    .filter_tensor = <filter_tensor>,
    .bias_tensor = <bias_tensor>,
    //
    .input_tensor = <input_tensor_ptr>,
    .output_tensor = <output_tensor_ptr>,
    .temp_buffer = nullptr,  // <scratch_buffer_size> = batch_num * output_nodes * sizeof(float)
};
'''
        name_prefix = 'fully_connected'
        self.oprun_str = "FullyConnected({0}_params_{1});".format(name_prefix, str(self.id))
        op_params, input_tensor, output_tensor, weights_tensor = \
            self.export_common(fp, model, dynamic_buffer.io_tensors, name_prefix, op_params)
            
        op_opt = self.op.BuiltinOptions()
        option = tflite.FullyConnectedOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)
        
        assert(output_tensor.Type() == tflite.TensorType.FLOAT32)
        op_params = tfcom.export_activation_range_float(option, op_params)

        scratch_buffer_size = tfcom.get_tensor_element_num(output_tensor) / output_tensor.ShapeAsNumpy()[0] * 4 # 4: sizeof(float)
        str_prefix = "fully_connected_params_" + str(self.id)
        op_params = tfcom.export_scratch_buffer(dynamic_buffer, str_prefix, scratch_buffer_size, op_params)
        return op_params
    
    def export_quant(self, fp, model, dynamic_buffer):
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
    .temp_buffer = nullptr,  // <scratch_buffer_size> = batch_num * output_nodes * sizeof(int32_t)
};
'''     
        name_prefix = 'fully_connected'
        self.oprun_str = "FullyConnectedQuant({0}_params_{1});".format(name_prefix, str(self.id))
        op_params, input_tensor, output_tensor, weights_tensor = \
            self.export_common(fp, model, dynamic_buffer.io_tensors, name_prefix, op_params)

        input_zero_point = input_tensor.Quantization().ZeroPoint(0)
        op_params = op_params.replace('<input_offset>', str(-input_zero_point)) # FullyConnectedParamsQuantized
        weights_zero_point = weights_tensor.Quantization().ZeroPoint(0)
        op_params = op_params.replace('<weights_offset>', str(-weights_zero_point))
        
        assert(output_tensor.Type() == tflite.TensorType.INT8)
        output_zero_point = output_tensor.Quantization().ZeroPoint(0)
        op_params = op_params.replace('<output_offset>', str(output_zero_point))
        
        op_params = tfcom.export_multiplier_per_tensor(input_tensor, output_tensor, weights_tensor, op_params)
        op_params = tfcom.export_activation_range_quant(output_tensor.Type(), op_params)

        scratch_buffer_size = tfcom.get_tensor_element_num(output_tensor) / output_tensor.ShapeAsNumpy()[0] * 4 # 4: sizeof(int32_t)
        str_prefix = "fully_connected_params_" + str(self.id)
        op_params = tfcom.export_scratch_buffer(dynamic_buffer, str_prefix, scratch_buffer_size, op_params)
        return op_params
        
    def export(self, fp, model, dynamic_buffer):
        self.scan_iotensor_lifetime(dynamic_buffer)
        if self.is_quant():
            op_params = self.export_quant(fp, model, dynamic_buffer)
        else:
            op_params = self.export_float(fp, model, dynamic_buffer)
        fp["model"].write(op_params+"\n")


import numpy as np
import tflite

import exporter.common as tfcom
from exporter.operators.operator import Operator

class MaxPooling(Operator):
    header_quant = '#include "engine/infer/kernels/max_pooling_quant.hpp"\n'
    header_float = '#include "engine/infer/kernels/max_pooling.hpp"\n'

    def __init__(self, graph, op, op_id):
        super().__init__(graph, op, op_id)
        self.attr["code"] = tflite.BuiltinOperator.MAX_POOL_2D
        
        self.attr["input_index"] = [0]
        self.attr["output_index"] = [0]
    
    def export_common(self, fp, model, io_tensors, name_prefix, op_params):
        op_params = op_params.replace('<op_id>', str(self.id))

        # io tensors
        op_params, input_tensor, output_tensor = self.export_io_tensors(name_prefix, op_params, io_tensors, False, fp)
        
        op_opt = self.op.BuiltinOptions()
        option = tflite.Pool2DOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)
        
        stride_height = option.StrideH()
        stride_width = option.StrideW()
        filter_height = option.FilterHeight()
        filter_width = option.FilterWidth()
        
        op_params = op_params.replace('<stride_height>', str(stride_height))
        op_params = op_params.replace('<stride_width>', str(stride_width))
        op_params = op_params.replace('<filter_height>', str(filter_height))
        op_params = op_params.replace('<filter_width>', str(filter_width))
        
        # Padding
        tfcom.export_padding_type(option, op_params)
        input_height, input_width = input_tensor.ShapeAsNumpy()[1:3]
        padding_size = tfcom.compute_padding_size(option.Padding(), 
                                                  [input_height, input_width],
                                                  [filter_height, filter_width],
                                                  [stride_height, stride_width], 
                                                  [1, 1])
        padding_size_str = '{ .width = ' + str(padding_size[1]) + ", .height = " + str(padding_size[0]) + '}'
        op_params = op_params.replace('<PaddingValues>', padding_size_str)
        
        return op_params, input_tensor, output_tensor, option
    
    def export_float(self, fp, model, io_tensors):
        op_params = \
'''
PoolParams pooling_params_<op_id> {
    .op_id = <op_id>,
    
    .padding_values = <PaddingValues>,
    .stride_height = <stride_height>,
    .stride_width = <stride_width>,
    .filter_height = <filter_height>,
    .filter_width = <filter_width>,
    // float activation params.
    .float_activation_min = <float_activation_min>,
    .float_activation_max = <float_activation_max>,
    //
    .input_tensor = <input_tensor_ptr>,
    .output_tensor = <output_tensor_ptr>,
};
'''
        name_prefix = 'maxpooling'
        self.oprun_str = "MaxPool(pooling_params_{0});".format(str(self.id))
        
        op_params, input_tensor, output_tensor, option = \
            self.export_common(fp, model, io_tensors, name_prefix, op_params)
            
        # Actication
        assert(output_tensor.Type() == tflite.TensorType.FLOAT32)
        op_params = tfcom.export_fused_activation_float(option, op_params)
        
        return op_params
    
    def export_quant(self, fp, model, io_tensors):
        # PoolParams
        op_params = \
'''
PoolQuantParams pooling_q_params_<op_id> = {
    .op_id = <op_id>,
    
    .padding_values = <PaddingValues>,
    .stride_height = <stride_height>,
    .stride_width = <stride_width>,
    .filter_height = <filter_height>,
    .filter_width = <filter_width>,
    // uint8_t, etc, activation params.
    .quantized_activation_min = <quantized_activation_min>,
    .quantized_activation_max = <quantized_activation_max>,
    //
    .input_tensor = <input_tensor_ptr>,
    .output_tensor = <output_tensor_ptr>,
};
'''
        name_prefix = 'maxpooling'
        self.oprun_str = "MaxPoolQuant(pooling_q_params_{0});".format(str(self.id))
        
        op_params, input_tensor, output_tensor, option = \
            self.export_common(fp, model, io_tensors, name_prefix, op_params)
            
        # Actication
        assert(output_tensor.Type() == tflite.TensorType.INT8)
        op_params = tfcom.export_fused_activation_quant(output_tensor.Type(), op_params)
        
        return op_params
    
    def export(self, fp, model, io_tensors):
        if self.is_quant():
            op_params = self.export_quant(fp, model, io_tensors)
        else:
            op_params = self.export_float(fp, model, io_tensors)
        fp["model"].write(op_params+"\n")
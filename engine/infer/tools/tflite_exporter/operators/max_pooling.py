

import numpy as np
import tflite

import tflite_exporter.common as tfcom
from tflite_exporter.operators.operator import Operator

class MaxPooling(Operator):
    header_quant = '#include "engine/infer/kernels/max_pooling_quant.hpp"\n'
    header_float = '#include "engine/infer/kernels/max_pooling.hpp"\n'

    def __init__(self, graph, op, op_id):
        super().__init__(graph, op, op_id)
        self.attr["code"] = tflite.BuiltinOperator.MAX_POOL_2D
        
        self.attr["input_index"] = [0]
        self.attr["output_index"] = [0]
    
    def export_float(self, fp, model, io_tensors):
        op_params = \
'''
typedef struct {
    .op_id = <op_id>,
    
    .padding_values = <PaddingValues>,
    .stride_width = <stride_width>,
    .stride_height = <stride_height>,
    .filter_width = <filter_width>,
    .filter_height = <filter_height>,
    // uint8_t, etc, activation params.
    .float_activation_min = <float_activation_min>,
    .float_activation_max = <float_activation_max>
    //
    .input_tensor = <input_tensor_ptr>,
    .output_tensor = <output_tensor_ptr>,
} PoolQuantParams;

'''
    
    def export_quant(self, fp, model, io_tensors):
        # PoolParams
        op_params = \
'''
PoolQuantParams pooling_params_<op_id> = {
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
        op_params = op_params.replace('<op_id>', str(self.id))
        
        input_tensor_id = self.op.Inputs(self.attr["input_index"][0])
        input_tensor = self.graph.Tensors(input_tensor_id)
        input_height, input_width = input_tensor.ShapeAsNumpy()[1:3]
        
        op_params = tfcom.write_io_tensor('maxpooling', 'input', self.id, input_tensor, input_tensor_id, io_tensors, op_params, fp["model"])

        output_tensor_id = self.op.Outputs(self.attr["output_index"][0])
        output_tensor = self.graph.Tensors(output_tensor_id)
        
        op_params = tfcom.write_io_tensor('maxpooling', 'output', self.id, output_tensor, output_tensor_id, io_tensors, op_params, fp["model"])
        
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
        padding_size = tfcom.compute_padding_size(option.Padding(), 
                                                  [input_height, input_width],
                                                  [filter_height, filter_width],
                                                  [stride_height, stride_width], 
                                                  [1, 1])
        padding_size_str = '{ .width = ' + str(padding_size[1]) + ", .height = " + str(padding_size[0]) + '}'
        op_params = op_params.replace('<PaddingValues>', padding_size_str)
        
        # Actication
        op_params = tfcom.export_fused_activation_quant(output_tensor.Type(), op_params)
        
        self.oprun_str = "MaxPool(pooling_params_{0});".format(str(self.id))
        return op_params
    
    def export(self, fp, model, io_tensors):
        if self.is_quant():
            op_params = self.export_quant(fp, model, io_tensors)
        else:
            op_params = self.export_float(fp, model, io_tensors)
        fp["model"].write(op_params+"\n")
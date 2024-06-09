

import numpy as np
import math
import tflite

import exporter.common as tfcom
from exporter.operators.operator import Operator

class Dequantize(Operator):
    header_quant = '#include "engine/infer/kernels/dequantize.hpp"\n'
    header_float = '#include "engine/infer/kernels/dequantize.hpp"\n'

    def __init__(self, graph, op, op_id):
        super().__init__(graph, op, op_id)
        self.attr["code"] = tflite.BuiltinOperator.DEQUANTIZE
        
        self.attr["input_index"] = [0]
        self.attr["output_index"] = [0]

    def export(self, fp, model, io_tensors):
        op_params = \
'''
DequantizationParams dequantize_params_<op_id> = {
    .op_id = <op_id>,
    
    .zero_point = <zero_point>,
    .scale = <scale>,
    
    .input_tensor = <input_tensor_ptr>,
    .output_tensor = <output_tensor_ptr>,
};
'''
        name_prefix = 'dequantize'
        self.oprun_str = "Dequantize(dequantize_params_{0});".format(str(self.id))
        op_params = op_params.replace('<op_id>', str(self.id))
        
         # io tensors
        op_params, input_tensor, output_tensor = self.export_io_tensors(name_prefix, op_params, io_tensors, False, fp)
        
        assert(input_tensor.Type() == tflite.TensorType.INT8)
        assert(output_tensor.Type() == tflite.TensorType.FLOAT32)
        
        # ref: tensorflow\lite\micro\kernels\dequantize_common.cc: DequantizePrepare
        output_zero_point = input_tensor.Quantization().ZeroPoint(0)
        output_scale = input_tensor.Quantization().Scale(0)
        
        op_params = op_params.replace('<zero_point>', str(output_zero_point))
        op_params = op_params.replace('<scale>', str(output_scale))
        
        fp["model"].write(op_params+"\n")
        
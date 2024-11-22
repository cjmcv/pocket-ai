

import numpy as np
import math
import tflite

import exporter.common as tfcom
from exporter.operators.operator import Operator

class Quantize(Operator):
    header_quant = '#include "engine/infer/kernels/quantize.hpp"\n'
    header_float = '#include "engine/infer/kernels/quantize.hpp"\n'

    def __init__(self, graph, op, op_id):
        super().__init__(graph, op, op_id)
        self.attr["code"] = tflite.BuiltinOperator.QUANTIZE
        
        self.attr["input_index"] = [0]
        self.attr["output_index"] = [0]

    def export(self, fp, model, dynamic_buffer):
        self.scan_iotensor_lifetime(dynamic_buffer)
        op_params = \
'''
AffineQuantizationParams quantize_params_<op_id> = {
    .op_id = <op_id>,
    
    .zero_point = <zero_point>,
    .scale = <scale>,
    
    .input_tensor = <input_tensor_ptr>,
    .output_tensor = <output_tensor_ptr>,
};
'''
        name_prefix = 'quantize'
        self.oprun_str = "AffineQuantize(quantize_params_{0});".format(str(self.id))
        op_params = op_params.replace('<op_id>', str(self.id))
        
         # io tensors
        op_params, input_tensor, output_tensor = self.export_io_tensors(name_prefix, op_params, dynamic_buffer.io_tensors, False, fp)
        
        assert(input_tensor.Type() == tflite.TensorType.FLOAT32)
        assert(output_tensor.Type() == tflite.TensorType.INT8)
        
        # ref: tensorflow\lite\micro\kernels\quantize_common.cc: PrepareQuantizeReference
        output_zero_point = output_tensor.Quantization().ZeroPoint(0)
        output_scale = output_tensor.Quantization().Scale(0)
        
        op_params = op_params.replace('<zero_point>', str(output_zero_point))
        op_params = op_params.replace('<scale>', str(output_scale))
        
        fp["model"].write(op_params+"\n")
        
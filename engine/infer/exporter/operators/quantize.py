

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

    def export_float(self, fp, model, io_tensors):
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
        op_params, input_tensor, output_tensor = self.export_io_tensors(name_prefix, op_params, io_tensors, True, fp)
        
        op_opt = self.op.BuiltinOptions()
        option = tflite.QuantizeOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)

        # ref: tensorflow\lite\micro\kernels\quantize_common.cc: PrepareQuantizeReference
        output_zero_point = output_tensor.Quantization().ZeroPoint(0)
        output_scale = output_tensor.Quantization().Scale(0)
        
        op_params = op_params.replace('<zero_point>', str(output_zero_point))
        op_params = op_params.replace('<scale>', str(output_scale))
        return op_params

    def export_quant(self, fp, model, io_tensors):
        op_params = \
'''
QuantizationParams quantize_params_<op_id> = {
    .op_id = <op_id>,
    
    .effective_scale_multiplier = <effective_scale_multiplier>,
    .effective_scale_shift = <effective_scale_shift>,
    .input_zeropoint = <input_zeropoint>,
    
    .input_tensor = <input_tensor_ptr>,
    .output_tensor = <output_tensor_ptr>,
};
'''
        return op_params
        
    def export(self, fp, model, io_tensors):
        if self.is_quant():
            op_params = self.export_quant(fp, model, io_tensors)
        else:
            op_params = self.export_float(fp, model, io_tensors)
        fp["model"].write(op_params+"\n")
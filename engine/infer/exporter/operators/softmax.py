

import numpy as np
import math
import tflite

import exporter.common as tfcom
from exporter.operators.operator import Operator

class Softmax(Operator):
    header_quant = '#include "engine/infer/kernels/softmax_quant.hpp"\n'
    header_float = '#include "engine/infer/kernels/softmax.hpp"\n'

    def __init__(self, graph, op, op_id):
        super().__init__(graph, op, op_id)
        self.attr["code"] = tflite.BuiltinOperator.SOFTMAX
        
        self.attr["input_index"] = [0]
        self.attr["output_index"] = [0]

    def export_float(self, fp, model, io_tensors):
        op_params = \
'''
SoftmaxQuantParams softmax_params_<op_id> = {
    .op_id = <op_id>,
    .input_tensor = <input_tensor_ptr>,
    .output_tensor = <output_tensor_ptr>,
};
'''
        return op_params

    def export_quant(self, fp, model, io_tensors):
        op_params = \
'''
SoftmaxQuantParams softmax_params_<op_id> = {
    .op_id = <op_id>,
    
    .input_multiplier = <input_multiplier>,
    .input_left_shift = <input_left_shift>,
    .diff_min = <diff_min>,
    
    .input_tensor = <input_tensor_ptr>,
    .output_tensor = <output_tensor_ptr>,
};
'''
        name_prefix = 'softmax'
        self.oprun_str = "SoftmaxQuant(softmax_params_{0});".format(str(self.id))
        op_params = op_params.replace('<op_id>', str(self.id))
        
         # io tensors
        op_params, input_tensor, output_tensor = self.export_io_tensors(name_prefix, op_params, io_tensors, True, fp)
        
        op_opt = self.op.BuiltinOptions()
        option = tflite.SoftmaxOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)
        

        # ref: tensorflow\lite\micro\kernels\softmax_common.cc: CalculateSoftmaxParams
        # ref: tensorflow\lite\kernels\internal\quantization_util.cc: PreprocessSoftmaxScaling
        beta = option.Beta()
        input_integer_bits = 5
        input_scale = input_tensor.Quantization().Scale(0)
        # output_scale = output_tensor.Quantization().Scale(0)
        max_real_multiplier = (1 << 31) - 1.0
        input_beta_real_multiplier = min(beta * input_scale * (1 << (31 - input_integer_bits)), max_real_multiplier)
        input_multiplier, input_left_shift = tfcom.quantize_multiplier(input_beta_real_multiplier)
        op_params = op_params.replace('<input_multiplier>', str(input_multiplier))
        op_params = op_params.replace('<input_left_shift>', str(input_left_shift))
    
        # ref: tensorflow\lite\kernels\internal\quantization_util.cc: CalculateInputRadius
        total_signed_bits = 31
        max_input_rescaled = 1.0 * ((1 << input_integer_bits) - 1) * (1 << (total_signed_bits - input_integer_bits)) / (1 << input_left_shift)
        op_params = op_params.replace('<diff_min>', str(-1 * math.floor(max_input_rescaled)))

        return op_params
        
    def export(self, fp, model, io_tensors):
        if self.is_quant():
            op_params = self.export_quant(fp, model, io_tensors)
        else:
            op_params = self.export_float(fp, model, io_tensors)
        fp["model"].write(op_params+"\n")
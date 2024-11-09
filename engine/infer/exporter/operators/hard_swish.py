import numpy as np
import tflite

import exporter.common as tfcom
from exporter.operators.operator import Operator

class HardSwish(Operator):
    header_quant = '#include "engine/infer/kernels/hard_swish_quant.hpp"\n'
    header_float = '#include "engine/infer/kernels/hard_swish.hpp"\n'

    def __init__(self, graph, op, op_id):
        super().__init__(graph, op, op_id)
        self.attr["code"] = tflite.BuiltinOperator.HARD_SWISH
        
        self.attr["input_index"] = [0]
        self.attr["output_index"] = [0]
    
    def export_float(self, fp, model, io_tensors):
        op_params = \
'''
HardSwishParams hard_swish_params_<op_id> {
    .op_id = <op_id>,
    
    .input_tensor = <input_tensor_ptr>,
    .output_tensor = <output_tensor_ptr>,
};
'''
        name_prefix = 'hard_swish'
        self.oprun_str = "HardSwish({0}_params_{1});".format(name_prefix, str(self.id))
        op_params = op_params.replace('<op_id>', str(self.id))
        op_params, input_tensor, output_tensor = self.export_io_tensors(name_prefix, op_params, io_tensors, False, fp)
        return op_params
    
    def export_quant(self, fp, model, io_tensors):
        op_params = \
'''
HardSwishQuantParams hard_swish_q_params_<op_id> {
    .op_id = <op_id>,
    
    .input_zero_point = <input_zero_point>,
    .output_zero_point = <output_zero_point>,
    .reluish_multiplier_fixedpoint_int16 = <reluish_multiplier_fixedpoint_int16>,
    .reluish_multiplier_exponent = <reluish_multiplier_exponent>,
    .output_multiplier_fixedpoint_int16 = <output_multiplier_fixedpoint_int16>,
    .output_multiplier_exponent = <output_multiplier_exponent>,
    
    .input_tensor = <input_tensor_ptr>,
    .output_tensor = <output_tensor_ptr>,
};
'''     
        name_prefix = 'hard_swish_q'
        self.oprun_str = "HardSwishQuant({0}_params_{1});".format(name_prefix, str(self.id))
        op_params = op_params.replace('<op_id>', str(self.id))
        op_params, input_tensor, output_tensor = self.export_io_tensors(name_prefix, op_params, io_tensors, False, fp)
        
        # ref: tensorflow\lite\micro\kernels\hard_swish_common.cc: HardSwishPrepare        
        input_zero_point = input_tensor.Quantization().ZeroPoint(0)
        op_params = op_params.replace('<input_zero_point>', str(input_zero_point))
        output_zero_point = output_tensor.Quantization().ZeroPoint(0)
        op_params = op_params.replace('<output_zero_point>', str(output_zero_point))
        
        input_scale = input_tensor.Quantization().Scale(0)
        hires_input_scale = (1.0 / 128.0) * input_scale
        reluish_scale = 3.0 / 32768.0
        output_scale = output_tensor.Quantization().Scale(0)
        
        output_multiplier = hires_input_scale / output_scale
        output_multiplier_fixedpoint_int32, output_multiplier_exponent = tfcom.quantize_multiplier(output_multiplier)
        op_params = op_params.replace('<output_multiplier_exponent>', str(output_multiplier_exponent))
        output_multiplier_fixedpoint_int16 = tfcom.downscale_int32_to_int16_multiplier(output_multiplier_fixedpoint_int32)
        op_params = op_params.replace('<output_multiplier_fixedpoint_int16>', str(output_multiplier_fixedpoint_int16))
    
        reluish_multiplier = hires_input_scale / reluish_scale
        reluish_multiplier_fixedpoint_int32, reluish_multiplier_exponent = tfcom.quantize_multiplier(reluish_multiplier)
        op_params = op_params.replace('<reluish_multiplier_exponent>', str(reluish_multiplier_exponent))
        reluish_multiplier_fixedpoint_int16 = tfcom.downscale_int32_to_int16_multiplier(reluish_multiplier_fixedpoint_int32)
        op_params = op_params.replace('<reluish_multiplier_fixedpoint_int16>', str(reluish_multiplier_fixedpoint_int16))
        
        return op_params
        
    def export(self, fp, model, dynamic_buffer):
        if self.is_quant():
            op_params = self.export_quant(fp, model, dynamic_buffer.io_tensors)
        else:
            op_params = self.export_float(fp, model, dynamic_buffer.io_tensors)
        fp["model"].write(op_params+"\n")
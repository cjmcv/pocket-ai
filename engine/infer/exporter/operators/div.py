

import numpy as np
import tflite

import exporter.common as tfcom
from exporter.operators.operator import Operator

# 对于Add / Sub / Mul 算子提取公共部分代码到Operator的binary中
# 如Mean / Sum / Max 等算子则提取公共部分代码到Operator的reduce中
class Div(Operator):
    header_quant = '#include "engine/infer/kernels/div_quant.hpp"\n'
    header_float = '#include "engine/infer/kernels/div.hpp"\n'

    def __init__(self, graph, op, op_id):
        super().__init__(graph, op, op_id)
        self.attr["code"] = tflite.BuiltinOperator.DIV
        
        self.attr["input_index"] = [0, 1]
        self.attr["output_index"] = [0]
    
    def export_float(self, fp, model, io_tensors):
        op_params = \
'''
DivParams div_params_<op_id> {
    .op_id = <op_id>,
    .requires_broadcast = <requires_broadcast>,
    
    .float_activation_min = <float_activation_min>,
    .float_activation_max = <float_activation_max>,
    //
    .input_tensor = {<input_tensor_ptr>, <input_tensor_ptr1>},
    .output_tensor = <output_tensor_ptr>,
};
'''
        name_prefix = 'div'
        self.oprun_str = "Div(div_params_{0});".format(str(self.id))
        op_params = op_params.replace('<op_id>', str(self.id))

        # io tensors
        op_params, input_tensors, output_tensors = self.export_io_tensors(name_prefix, op_params, io_tensors, False, fp)
        assert(len(input_tensors) == 2)
        # ref: tensorflow\lite\micro\kernels\add_common.cc: CalculateOpDataAdd
        if (input_tensors[0].ShapeAsNumpy().any() == input_tensors[1].ShapeAsNumpy().any()):
            op_params = op_params.replace('<requires_broadcast>', "false")
        else:
            op_params = op_params.replace('<requires_broadcast>', "true")
            print(input_tensors[0].ShapeAsNumpy(), input_tensors[1].ShapeAsNumpy())
            
        op_opt = self.op.BuiltinOptions()
        option = tflite.DivOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)

        assert(output_tensors[0].Type() == tflite.TensorType.FLOAT32)
        op_params = tfcom.export_fused_activation_float(option, op_params)
        return op_params
    
    def export_quant(self, fp, model, io_tensors):
        # PoolParams
        op_params = \
'''
DivQuantParams div_q_params_<op_id> {
    .op_id = <op_id>,
    
    .requires_broadcast = <requires_broadcast>,
    // uint8_t inference params.
    .input1_offset = <input1_offset>,
    .input2_offset = <input2_offset>,
    .output_offset = <output_offset>,
    .output_multiplier = <output_multiplier>,
    .output_shift = <output_shift>,

    // uint8_t, etc, activation params.
    .quantized_activation_min = <quantized_activation_min>,
    .quantized_activation_max = <quantized_activation_max>,
    
    .input_tensor = {<input_tensor_ptr>, <input_tensor_ptr1>},
    .output_tensor = <output_tensor_ptr>,
};
'''
        name_prefix = 'div'
        self.oprun_str = "DivQuant(div_q_params_{0});".format(str(self.id))
        op_params = op_params.replace('<op_id>', str(self.id))

        # io tensors
        op_params, input_tensors, output_tensors = self.export_io_tensors(name_prefix, op_params, io_tensors, False, fp)
        assert(len(input_tensors) == 2)
        
        # ref: tensorflow\lite\micro\kernels\div_common.cc: CalculateOpDataDiv + EvalQuantized
        if (input_tensors[0].ShapeAsNumpy().any() == input_tensors[1].ShapeAsNumpy().any()):
            op_params = op_params.replace('<requires_broadcast>', "false")
        else:
            op_params = op_params.replace('<requires_broadcast>', "true")
            print(input_tensors[0].ShapeAsNumpy(), input_tensors[1].ShapeAsNumpy())

        op_params = op_params.replace('<input1_offset>', str(-input_tensors[0].Quantization().ZeroPoint(0)))
        op_params = op_params.replace('<input2_offset>', str(-input_tensors[1].Quantization().ZeroPoint(0)))
        op_params = op_params.replace('<output_offset>', str(output_tensors[0].Quantization().ZeroPoint(0)))
        
        real_multiplier = input_tensors[0].Quantization().Scale(0) /  \
                         (input_tensors[1].Quantization().Scale(0) * output_tensors[0].Quantization().Scale(0))
        output_multiplier, output_shift = tfcom.quantize_multiplier(real_multiplier)
        op_params = op_params.replace('<output_multiplier>', str(output_multiplier))
        op_params = op_params.replace('<output_shift>', str(output_shift))
    
        op_params = tfcom.export_fused_activation_quant(output_tensors[0].Type(), op_params)
        return op_params
    
    def export(self, fp, model, io_tensors):
        if self.is_quant():
            op_params = self.export_quant(fp, model, io_tensors)
        else:
            op_params = self.export_float(fp, model, io_tensors)
        fp["model"].write(op_params+"\n")
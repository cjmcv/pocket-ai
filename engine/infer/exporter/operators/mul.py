

import numpy as np
import tflite

import exporter.common as tfcom
from exporter.operators.operator import Operator

# 对于Add / Sub / Mul 算子提取公共部分代码到Operator的binary中
# 如Mean / Sum / Max 等算子则提取公共部分代码到Operator的reduce中
class Mul(Operator):
    header_quant = '#include "engine/infer/kernels/mul_quant.hpp"\n'
    header_float = '#include "engine/infer/kernels/mul.hpp"\n'

    def __init__(self, graph, op, op_id):
        super().__init__(graph, op, op_id)
        self.attr["code"] = tflite.BuiltinOperator.MUL
        
        self.attr["input_index"] = [0, 1]
        self.attr["output_index"] = [0]
    
    def export_float(self, fp, model, io_tensors):
        op_params = \
'''
MulParams mul_params_<op_id> {
    .op_id = <op_id>,
    .requires_broadcast = <requires_broadcast>,
    
    .float_activation_min = <float_activation_min>,
    .float_activation_max = <float_activation_max>,
    //
    .input_tensor = {<input_tensor_ptr>, <input_tensor_ptr1>},
    .output_tensor = <output_tensor_ptr>,
};
'''
        name_prefix = 'mul'
        self.oprun_str = "Mul(mul_params_{0});".format(str(self.id))
        op_params = op_params.replace('<op_id>', str(self.id))

        # io tensors
        op_params, input_tensors, output_tensors = self.export_io_tensors(name_prefix, op_params, io_tensors, True, fp)
        assert(len(input_tensors) == 2)
        # ref: tensorflow\lite\micro\kernels\add_common.cc: CalculateOpDataAdd
        if ((input_tensors[0].ShapeAsNumpy() == input_tensors[1].ShapeAsNumpy()).all()):
            op_params = op_params.replace('<requires_broadcast>', "false")
        else:
            op_params = op_params.replace('<requires_broadcast>', "true")
            # print(input_tensors[0].ShapeAsNumpy(), input_tensors[1].ShapeAsNumpy())
        
        self.check_and_export_const_tensor(self.attr["input_index"][0], np.float32, model, name_prefix, io_tensors, fp)
        self.check_and_export_const_tensor(self.attr["input_index"][1], np.float32, model, name_prefix, io_tensors, fp)    
        
        op_opt = self.op.BuiltinOptions()
        option = tflite.AddOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)

        assert(output_tensors[0].Type() == tflite.TensorType.FLOAT32)
        op_params = tfcom.export_activation_range_float(option, op_params)
        return op_params
    
    def export_quant(self, fp, model, io_tensors):
        # PoolParams
        op_params = \
'''
MulQuantParams mul_q_params_<op_id> {
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
        name_prefix = 'mul'
        self.oprun_str = "MulQuant(mul_q_params_{0});".format(str(self.id))
        op_params = op_params.replace('<op_id>', str(self.id))

        # io tensors
        op_params, input_tensors, output_tensors = self.export_io_tensors(name_prefix, op_params, io_tensors, True, fp)
        assert(len(input_tensors) == 2)
        # ref: tensorflow\lite\micro\kernels\add_common.cc: CalculateOpDataAdd
        if ((input_tensors[0].ShapeAsNumpy() == input_tensors[1].ShapeAsNumpy()).all()):
            op_params = op_params.replace('<requires_broadcast>', "false")
        else:
            op_params = op_params.replace('<requires_broadcast>', "true")
            # print(input_tensors[0].ShapeAsNumpy(), input_tensors[1].ShapeAsNumpy())

        self.check_and_export_const_tensor(self.attr["input_index"][0], np.int8, model, name_prefix, io_tensors, fp)
        self.check_and_export_const_tensor(self.attr["input_index"][1], np.int8, model, name_prefix, io_tensors, fp)
        
        # ref: tensorflow\lite\micro\kernels\mul_common.cc#38: CalculateOpDataMul
        op_params = op_params.replace('<input1_offset>', str(-input_tensors[0].Quantization().ZeroPoint(0))) # "-" in EvalMulQuantizedReference
        op_params = op_params.replace('<input2_offset>', str(-input_tensors[1].Quantization().ZeroPoint(0)))
        op_params = op_params.replace('<output_offset>', str(output_tensors[0].Quantization().ZeroPoint(0)))

        real_multiplier = input_tensors[0].Quantization().Scale(0) *  \
                          input_tensors[1].Quantization().Scale(0) /  \
                          output_tensors[0].Quantization().Scale(0)
        multiplier, shift = tfcom.quantize_multiplier(real_multiplier)
        op_params = op_params.replace('<output_multiplier>', str(multiplier))
        op_params = op_params.replace('<output_shift>', str(shift))
    
        op_params = tfcom.export_activation_range_quant(output_tensors[0].Type(), op_params)
        return op_params
    
    def export(self, fp, model, dynamic_buffer):
        self.scan_iotensor_lifetime(dynamic_buffer)
        if self.is_quant():
            op_params = self.export_quant(fp, model, dynamic_buffer.io_tensors)
        else:
            op_params = self.export_float(fp, model, dynamic_buffer.io_tensors)
        fp["model"].write(op_params+"\n")
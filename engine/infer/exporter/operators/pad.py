

import numpy as np
import tflite

import exporter.common as tfcom
from exporter.operators.operator import Operator

class Pad(Operator):
    header_quant = '#include "engine/infer/kernels/pad.hpp"\n'
    header_float = '#include "engine/infer/kernels/pad.hpp"\n'

    def __init__(self, graph, op, op_id):
        super().__init__(graph, op, op_id)
        self.attr["code"] = tflite.BuiltinOperator.PAD
        
        self.attr["input_index"] = [0]
        self.attr["paddings_index"] = [1]
        self.attr["constant_values_index"] = [2]
        self.attr["output_index"] = [0]

    def export(self, fp, model, dynamic_buffer):
        self.scan_iotensor_lifetime(dynamic_buffer)
        op_params = \
'''
PadParams pad_params_<op_id> = {
    .op_id = <op_id>,

    .left_padding_count = <left_padding_count>,
    .left_padding = <left_padding>,
    .right_padding_count = <right_padding_count>,
    .right_padding = <right_padding>,
    .pad_value = <pad_value>,

    .input_tensor = <input_tensor_ptr>,
    .output_tensor = <output_tensor_ptr>,
};
'''
        name_prefix = 'pad'
        
        op_params = op_params.replace('<op_id>', str(self.id))
        
         # io tensors
        op_params, input_tensor, output_tensor = self.export_io_tensors(name_prefix, op_params, dynamic_buffer.io_tensors, False, fp)
        
        if input_tensor.Type() is tflite.TensorType.FLOAT32:
            self.oprun_str = "Pad<float>(pad_params_{0});".format(str(self.id))
        elif input_tensor.Type() is tflite.TensorType.INT8:
            self.oprun_str = "Pad<int8_t>(pad_params_{0});".format(str(self.id))
        else:
            print("Error: Type {0} is not supported in Pad.\n".format(input_tensor.Type()))  
        
        # Get pad_value
        # ref: tensorflow\lite\micro\kernels\pad.cc: PadEval
        if self.op.InputsLength() == 3:
            constant_values_tensor = self.graph.Tensors(self.op.Inputs(self.attr["constant_values_index"][0]))
            constant_values_buffer = model.Buffers(constant_values_tensor.Buffer())
            if input_tensor.Type() is tflite.TensorType.FLOAT32:
                data = np.frombuffer(constant_values_buffer.DataAsNumpy(), dtype=np.float32)
                pad_value_str = '{ .fp32_value = ' + str(data[0]) + ', }'
                op_params = op_params.replace('<pad_value>', pad_value_str)
            else:
                data = np.frombuffer(constant_values_buffer.DataAsNumpy(), dtype=np.int8)
                pad_value_str = '{ .int8_value = ' + str(data[0]) + ', }'
                op_params = op_params.replace('<pad_value>', pad_value_str)                
        elif self.op.InputsLength() == 2:
            if input_tensor.Type() is tflite.TensorType.FLOAT32:
                pad_value_str = '{ .fp32_value = 0, }'
                op_params = op_params.replace('<pad_value>', pad_value_str)
            else:
                output_zero_point = output_tensor.Quantization().ZeroPoint(0)
                pad_value_str = '{ .int8_value = ' + str(output_zero_point) + ', }'
                op_params = op_params.replace('<pad_value>', pad_value_str)

        # Get others
        # ref: tensorflow\lite\micro\kernels\pad.cc: PadPrepare
        shape_as_numpy = input_tensor.ShapeAsNumpy()
        num_input_dimensions = len(shape_as_numpy)
        op_params = op_params.replace('<left_padding_count>', str(num_input_dimensions))
        op_params = op_params.replace('<right_padding_count>', str(num_input_dimensions))

        paddings_tensor = self.graph.Tensors(self.op.Inputs(self.attr["paddings_index"][0]))
        paddings_buffer = model.Buffers(paddings_tensor.Buffer())
        paddings_data = np.frombuffer(paddings_buffer.DataAsNumpy(), dtype=np.int32)
        
        left_padding = []
        right_padding = []
        for i in range(num_input_dimensions):
            left_padding.append(paddings_data[i*2])
            right_padding.append(paddings_data[i*2+1])
        
        left_padding_str = "{"
        for i in range(len(left_padding) - 1):
            left_padding_str = left_padding_str + str(left_padding[i]) + ", "
        left_padding_str = left_padding_str + str(left_padding[len(left_padding) - 1]) + "}"
        op_params = op_params.replace('<left_padding>', left_padding_str)

        right_padding_str = "{"
        for i in range(len(right_padding) - 1):
            right_padding_str = right_padding_str + str(right_padding[i]) + ", "
        right_padding_str = right_padding_str + str(right_padding[len(right_padding) - 1]) + "}"
        op_params = op_params.replace('<right_padding>', right_padding_str)
                  
        fp["model"].write(op_params+"\n")
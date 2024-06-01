

import numpy as np
import tflite

import tflite_exporter.common as tfcom
from tflite_exporter.operators.operator import Operator

class Reshape(Operator):
    header_quant = '#include "engine/infer/kernels/reshape.hpp"\n'
    header_float = '#include "engine/infer/kernels/reshape.hpp"\n'

    def __init__(self, graph, op, op_id):
        super().__init__(graph, op, op_id)
        self.attr["code"] = tflite.BuiltinOperator.RESHAPE
        
        self.attr["input_index"] = [0]
        self.attr["output_index"] = [0]

    def export(self, fp, model, io_tensors):
        op_params = \
'''
ReshapeParams reshape_params_<op_id> = {
    .op_id = <op_id>,
    .input_tensor = <input_tensor_ptr>,
    .output_tensor = <output_tensor_ptr>,
};
'''
        op_params = op_params.replace('<op_id>', str(self.id))
        
        input_tensor_id = self.op.Inputs(self.attr["input_index"][0])
        input_tensor = self.graph.Tensors(input_tensor_id)
        op_params = tfcom.write_io_tensor('reshape', 'input', self.id, input_tensor, input_tensor_id, io_tensors, op_params, fp["model"])

        output_tensor_id = self.op.Outputs(self.attr["output_index"][0])
        output_tensor = self.graph.Tensors(output_tensor_id)
        op_params = tfcom.write_io_tensor('reshape', 'output', self.id, output_tensor, output_tensor_id, io_tensors, op_params, fp["model"], input_tensor_id)
        
        self.oprun_str = "Reshape(reshape_params_{0});".format(str(self.id))
        fp["model"].write(op_params+"\n")
        
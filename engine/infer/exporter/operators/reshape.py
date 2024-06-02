

import numpy as np
import tflite

import exporter.common as tfcom
from exporter.operators.operator import Operator

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
        name_prefix = 'reshape'
        self.oprun_str = "Reshape(reshape_params_{0});".format(str(self.id))
        op_params = op_params.replace('<op_id>', str(self.id))
        
         # io tensors
        op_params, input_tensor, output_tensor = self.export_io_tensors(name_prefix, op_params, io_tensors, True, fp)

        fp["model"].write(op_params+"\n")


import numpy as np
import tflite

import exporter.common as tfcom
from exporter.operators.operator import Operator

class Tile(Operator):
    header_quant = '#include "engine/infer/kernels/tile.hpp"\n'
    header_float = '#include "engine/infer/kernels/tile.hpp"\n'

    def __init__(self, graph, op, op_id):
        super().__init__(graph, op, op_id)
        self.attr["code"] = tflite.BuiltinOperator.TILE
        
        self.attr["input_index"] = [0]
        self.attr["multipliers_index"] = 1
        self.attr["output_index"] = [0]

    def export(self, fp, model, dynamic_buffer):
        self.scan_iotensor_lifetime(dynamic_buffer)
        op_params = \
'''
TileParams tile_params_<op_id> = {
    .op_id = <op_id>,
    .multipliers = <multipliers>,
    
    .input_tensor = <input_tensor_ptr>,
    .output_tensor = <output_tensor_ptr>,
};
'''
        name_prefix = 'tile'
        op_params = op_params.replace('<op_id>', str(self.id))
        
         # io tensors
        op_params, input_tensor, output_tensor = self.export_io_tensors(name_prefix, op_params, dynamic_buffer.io_tensors, False, fp)
        
        if input_tensor.Type() is tflite.TensorType.FLOAT32:
            self.oprun_str = "Tile<float>(tile_params_{0});".format(str(self.id))
        elif input_tensor.Type() is tflite.TensorType.INT8:
            self.oprun_str = "Tile<int8_t>(tile_params_{0});".format(str(self.id))
        else:
            print("Error: Type {0} is not supported in Tile.\n".format(input_tensor.Type()))  
        
        multipliers_tensor_id = self.op.Inputs(self.attr["multipliers_index"])
        multipliers_tensor = self.graph.Tensors(multipliers_tensor_id)
        assert(multipliers_tensor.Type() == tflite.TensorType.INT32)
        multipliers_buffer = model.Buffers(multipliers_tensor.Buffer())
        multipliers_data = np.frombuffer(multipliers_buffer.DataAsNumpy(), dtype=np.int32)
        
        m_str = tfcom.format_multiplier(multipliers_data, name_prefix + "_multipliers_" + str(self.id))
        fp["params"].write(m_str)
        op_params = op_params.replace('<multipliers>', name_prefix + "_multipliers_" + str(self.id))    

        fp["model"].write(op_params+"\n")
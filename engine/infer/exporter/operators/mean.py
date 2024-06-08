

import numpy as np
import tflite

import exporter.common as tfcom
from exporter.operators.operator import Operator

# Mean / Sum / Max 等算子则提取公共部分代码到Operator的reduce中
class Mean(Operator):
    header_quant = '#include "engine/infer/kernels/mean_or_sum_quant.hpp"\n'
    header_float = '#include "engine/infer/kernels/mean_or_sum.hpp"\n'

    def __init__(self, graph, op, op_id):
        super().__init__(graph, op, op_id)
        self.attr["code"] = tflite.BuiltinOperator.ADD
        
        self.attr["input_index"] = [0]
        self.attr["axis_index"] = [1]
        self.attr["output_index"] = [0]
    
    def export_float(self, fp, model, io_tensors):
        op_params = \
'''
typedef struct {
    .op_id = <op_id>,
    
    .padding_values = <PaddingValues>,
    .stride_width = <stride_width>,
    .stride_height = <stride_height>,
    .filter_width = <filter_width>,
    .filter_height = <filter_height>,
    // uint8_t, etc, activation params.
    .float_activation_min = <float_activation_min>,
    .float_activation_max = <float_activation_max>
    //
    .input_tensor = <input_tensor_ptr>,
    .output_tensor = <output_tensor_ptr>,
} PoolQuantParams;
'''
    
    def export_quant(self, fp, model, io_tensors):
        op_params = \
'''
MeanOrSumQuantParams mean_params_<op_id> {
    .op_id = <op_id>,

    .is_compute_sum = <is_compute_sum>,

    .multiplier = <multiplier>,
    .shift = <shift>,
    .input_zero_point = <input_zero_point>, 
    .output_zero_point = <output_zero_point>,

    .num_output_elements = <num_output_elements>,
    .num_axis = <num_axis>,
    .axis = <axis>,
    
    .temp_buffer = <temp_buffer>,  // The same size as the output
    //
    .input_tensor = <input_tensor_ptr>,
    .output_tensor = <output_tensor_ptr>,
};
'''
        name_prefix = 'mean'
        self.oprun_str = "MeanOrSumQuant(mean_params_{0});".format(str(self.id))
        op_params = op_params.replace('<op_id>', str(self.id))
        op_params = op_params.replace('<is_compute_sum>', 'False')
        
        # io tensors
        op_params, input_tensor, output_tensor = self.export_io_tensors(name_prefix, op_params, io_tensors, False, fp)
        axis_tensor = self.graph.Tensors(self.op.Inputs(self.attr["axis_index"][0]))
        
        # ref: tensorflow\lite\micro\kernels\reduce_common.cc: PrepareMeanOrSumHelper
        real_multiplier = input_tensor.Quantization().Scale(0) / output_tensor.Quantization().Scale(0)
        multiplier, shift = tfcom.quantize_multiplier(real_multiplier)
        op_params = op_params.replace('<multiplier>', str(multiplier))
        op_params = op_params.replace('<shift>', str(shift))
        
        op_params = op_params.replace('<input_zero_point>', str(input_tensor.Quantization().ZeroPoint(0)))
        op_params = op_params.replace('<output_zero_point>', str(output_tensor.Quantization().ZeroPoint(0)))
        
        num_output_elements = tfcom.get_tensor_element_num(output_tensor)
        op_params = op_params.replace('<num_output_elements>', str(num_output_elements))
        num_axis = tfcom.get_tensor_element_num(axis_tensor)
        op_params = op_params.replace('<num_axis>', str(num_axis))
        
        axis_tensor_buffer = model.Buffers(axis_tensor.Buffer())
        axis_data = np.frombuffer(axis_tensor_buffer.DataAsNumpy(), dtype=np.int32)
        axis_data_str = "{"
        for i in range(len(axis_data) - 1):
            axis_data_str = axis_data_str + str(axis_data[i]) + ", "
        axis_data_str = axis_data_str + str(axis_data[len(axis_data) - 1]) + "}"
        op_params = op_params.replace('<axis>', axis_data_str)
        #
        output_size = tfcom.get_tensor_element_num(output_tensor)
        if output_size > Operator.g_scratch_bufffer_size:
            Operator.g_scratch_bufffer_size = output_size
        op_params = op_params.replace('<temp_buffer>', '(void *)' + Operator.g_scratch_bufffer_name + ', // ' + str(output_size))

        return op_params
    
    def export(self, fp, model, io_tensors):
        if self.is_quant():
            op_params = self.export_quant(fp, model, io_tensors)
        else:
            op_params = self.export_float(fp, model, io_tensors)
        fp["model"].write(op_params+"\n")
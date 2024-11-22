

import numpy as np
import tflite

import exporter.common as tfcom
from exporter.operators.operator import Operator

class Lstm(Operator):
    header_quant = '#include "engine/infer/kernels/lstm_quant.hpp"\n'
    header_float = '#include "engine/infer/kernels/lstm.hpp"\n'

    # https://zhenhuaw.me/tflite/docs/LSTMOptions.html
    def __init__(self, graph, op, op_id):
        super().__init__(graph, op, op_id)
        self.attr["code"] = tflite.BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_LSTM
        # tensorflow\lite\micro\kernels\lstm_shared.h#23
        self.attr["input_index"]       = [0]
        self.attr["weight_bias_index"] = [ 1,  2,  3,  4,  5,  6,  7,  8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        self.attr["output_index"] = [0]
    
    def export_float(self, fp, model, dynamic_buffer):
        op_params = \
'''
LstmParams lstm_params_<op_id> {
    .op_id = <op_id>,

    .input_tensor = {<input_tensor_ptr>, <other_tensor_ptr>},
    .output_tensor = <output_tensor_ptr>,
    
    // OpDataLSTM op_data;
    // LstmSizeInfo size_info;
    .op_data = {
        .size_info = {
            .time_major = <time_major>,
            .batch_size = <batch_size>,
            .time_steps = <time_steps>,
            .input_dimension = <input_dimension>,
            .state_dimension = <state_dimension>,
        },
        .cell_state_info = {
            .cell_clip = <cell_clip>,
            .quantized_cell_clip = <quantized_cell_clip>,  // not used in float
            .cell_state_scale_power = <cell_state_scale_power>, // not used in float
        },
        // TfLiteFusedActivation
        .cell_gate_nonlinear_type = <cell_gate_nonlinear_type>,
        // GateParameters: input_weight 2, input_bias 13, rec_weight 6
        .forget_gate_parameters = {
            .input_fc_activation_min = <float_activation_min>,
            .input_fc_activation_max = <float_activation_max>,
            .recurrent_fc_activation_min = <float_activation_min>,
            .recurrent_fc_activation_max = <float_activation_max>,
        },
        // GateParameters input_gate_parameters: input_weight 1, input_bias 12, rec_weight 5
        .input_gate_parameters = {
            .input_fc_activation_min = <float_activation_min>,
            .input_fc_activation_max = <float_activation_max>,
            .recurrent_fc_activation_min = <float_activation_min>,
            .recurrent_fc_activation_max = <float_activation_max>,
        },
        // GateParameters cell_gate_parameters: input_weight 3, input_bias 14, rec_weight 7
        .cell_gate_parameters = {
            .input_fc_activation_min = <float_activation_min>,
            .input_fc_activation_max = <float_activation_max>,
            .recurrent_fc_activation_min = <float_activation_min>,
            .recurrent_fc_activation_max = <float_activation_max>,
        },
        // GateParameters output_gate_parameters: input_weight 4, input_bias 15, rec_weight 8
        .output_gate_parameters = {
            .input_fc_activation_min = <float_activation_min>,
            .input_fc_activation_max = <float_activation_max>,
            .recurrent_fc_activation_min = <float_activation_min>,
            .recurrent_fc_activation_max = <float_activation_max>,
        },
        // InterGateParameters inter_gate_parameters
        .inter_gate_parameters = {
            .forget_cell_mul_params = {
                .float_activation_min = <float_activation_min>,
                .float_activation_max = <float_activation_max>,
            },
            .input_mul_params = {
                .float_activation_min = <float_activation_min>,
                .float_activation_max = <float_activation_max>,
            },
            .output_mul_params = {
                .float_activation_min = <float_activation_min>,
                .float_activation_max = <float_activation_max>,
            },
        },
        // 
        .buffer = nullptr, // <scratch_buffer_size>
    }
};
'''
        name_prefix = 'lstm'
        self.oprun_str = "Lstm(lstm_params_{0});".format(str(self.id))
        op_params = op_params.replace('<op_id>', str(self.id))

        # io tensors
        op_params, input_tensors, output_tensors = self.export_io_tensors(name_prefix, op_params, dynamic_buffer.io_tensors, False, fp)
        
        # weight
        const_num = 23
        for idx in range(const_num):
            weights_tensor_id = self.op.Inputs(self.attr["weight_bias_index"][idx])
            # print("id:", weights_tensor_id, idx)
            if (weights_tensor_id == -1):
                if idx == int(const_num - 1):
                    op_params = op_params.replace('<other_tensor_ptr>', "NULL")
                else:
                    op_params = op_params.replace('<other_tensor_ptr>', "NULL, <other_tensor_ptr>")
                continue
            weights_tensor = self.graph.Tensors(weights_tensor_id)
            weights_buffer = model.Buffers(weights_tensor.Buffer())
            
            assert(weights_tensor.Type() == tflite.TensorType.FLOAT32)
            if isinstance(weights_buffer.DataAsNumpy(), int):
                weight_data = np.zeros(tfcom.get_tensor_element_num(weights_tensor))
                # print("yes", weight_data, weights_tensor.ShapeAsNumpy())
            else:
                weight_data = np.frombuffer(weights_buffer.DataAsNumpy(), dtype=np.float32)
                # print("no", weight_data)
            name = weights_tensor.Name().decode('utf-8').replace('.', '_').replace('/', '_')
            
            weight_name = name_prefix + "_weights_" + name + "_" + str(self.id)
            weight_str, weight_var_name = tfcom.format_weight_bias(weight_data, weights_tensor.Type(), weight_name)
            fp["params"].write(weight_str)

            filter_tensor_str = tfcom.format_tensor(weights_tensor, weights_tensor_id, weight_var_name)
            tensor_name = "weight_tensor_{0}_{1}".format(str(self.attr["weight_bias_index"][idx]), name)
            fp["model"].write("Tensor {0} = {1};\n".format(tensor_name, filter_tensor_str))
            if idx == int(const_num - 1):
                op_params = op_params.replace('<other_tensor_ptr>', "&" + tensor_name)
            else:
                op_params = op_params.replace('<other_tensor_ptr>', "&" + tensor_name + ", <other_tensor_ptr>")
        
        op_opt = self.op.BuiltinOptions()
        option = tflite.LSTMOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)
        op_params = tfcom.export_activation_range_float(option, op_params) # Only fill max / min
        op_params = op_params.replace('<cell_gate_nonlinear_type>', '(TfLiteFusedActivation)'+str(option.FusedActivationFunction()))
        op_params = op_params.replace('<cell_clip>', str(option.CellClip()))
        
        # !warning: fixed parameters were used here because tflite cannot export relevant information 
        op_params = op_params.replace('<time_major>', "false")
        # tensorflow\lite\micro\kernels\lstm_eval_common.cc#26
        input_shape = input_tensors.ShapeAsNumpy()
        op_params = op_params.replace('<batch_size>', str(input_shape[0]))
        op_params = op_params.replace('<time_steps>', str(input_shape[1]))
        op_params = op_params.replace('<input_dimension>', str(input_shape[2]))
        
        hidden_state_tensor = self.graph.Tensors(self.op.Inputs(18)) # 18 = kLstmOutputStateTensor = hidden_state_tensor
        hidden_state_shape = hidden_state_tensor.ShapeAsNumpy()
        op_params = op_params.replace('<state_dimension>', str(hidden_state_shape[1]))
        
        # tensorflow\lite\micro\kernels\lstm_eval_common.cc#182
        op_params = op_params.replace('<quantized_cell_clip>', str(0))   # not used in float
        op_params = op_params.replace('<cell_state_scale_power>', str(0))
        
        # tensorflow\lite\micro\kernels\unidirectional_sequence_lstm.cc#88
        # batch_size * state_dimension * TfLiteTypeGetSize(cell_state_type)
        one_buffer_size = input_shape[0] * hidden_state_shape[1] * 4 # The 4 is sizeof(float)
        scratch_buffer_size = one_buffer_size * 4
        if scratch_buffer_size > dynamic_buffer.scratch_buffer_size:
            # print("g_scratch_buffer_size in: ", Operator.g_scratch_buffer_size, output_size * 4)
            dynamic_buffer.scratch_buffer_size = scratch_buffer_size # sizeof(int32_t)
        
        base_scratch_buffer_str = "    lstm_params_{0}.op_data.buffer[<id>] = (void *)({1}+<bias>); \n".format(str(self.id), dynamic_buffer.scratch_buffer_name)
        dynamic_buffer.scratch_buffer_allocate_info += base_scratch_buffer_str.replace('<id>', str(0)).replace('<bias>', str(0))
        dynamic_buffer.scratch_buffer_allocate_info += base_scratch_buffer_str.replace('<id>', str(1)).replace('<bias>', "1*" + str(one_buffer_size))
        dynamic_buffer.scratch_buffer_allocate_info += base_scratch_buffer_str.replace('<id>', str(2)).replace('<bias>', "2*" + str(one_buffer_size))
        dynamic_buffer.scratch_buffer_allocate_info += base_scratch_buffer_str.replace('<id>', str(3)).replace('<bias>', "3*" + str(one_buffer_size))
        op_params = op_params.replace('<scratch_buffer_size>', str(scratch_buffer_size))
        return op_params
    
    def export_quant(self, fp, model, io_tensors):
        # PoolParams
        op_params = \
'''
AddQuantParams add_q_params_<op_id> {
    .op_id = <op_id>,
    
    .requires_broadcast = <requires_broadcast>,
    // uint8_t inference params.
    .input1_offset = <input1_offset>,
    .input2_offset = <input2_offset>,
    .output_offset = <output_offset>,
    .output_multiplier = <output_multiplier>,
    .output_shift = <output_shift>,
    // Add / Sub, not Mul, uint8_t inference params.
    .left_shift = <left_shift>,
    .input1_multiplier = <input1_multiplier>,
    .input1_shift = <input1_shift>,
    .input2_multiplier = <input2_multiplier>,
    .input2_shift = <input2_shift>,

    // uint8_t, etc, activation params.
    .quantized_activation_min = <quantized_activation_min>,
    .quantized_activation_max = <quantized_activation_max>,
    
    .input_tensor = {<input_tensor_ptr>, <input_tensor_ptr1>},
    .output_tensor = <output_tensor_ptr>,
};
'''
        # TODO
        return op_params
    
    def export(self, fp, model, dynamic_buffer):
        self.scan_iotensor_lifetime(dynamic_buffer)
        if self.is_quant():
            op_params = self.export_quant(fp, model, dynamic_buffer.io_tensors)
        else:
            op_params = self.export_float(fp, model, dynamic_buffer)
        fp["model"].write(op_params+"\n")
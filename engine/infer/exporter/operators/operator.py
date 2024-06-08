
import sys
import numpy as np
import tflite
import exporter.common as tfcom

class Operator:
    g_scratch_bufffer_name = 'g_scratch_buffer'
    g_scratch_bufffer_size = 0  
    def __init__(self, graph, op, id):
        self.attr = {}
        self.graph = graph
        self.op = op
        self.id = id
    
    def op_id(self):
        return self.id
    
    def is_quant(self):
        input_tensor = self.graph.Tensors(self.op.Inputs(self.attr["input_index"][0]))
        if input_tensor.Type() == tflite.TensorType.FLOAT32:
            return False
        else:
            return True
        
    def export_io_tensor(self, prefix, tag, tensor, tensor_id, io_tensors, op_params, fp, inplace_id = -1, write_id = 0):
        if tag == 'input':
            suffix = 'input'
            if write_id == 0:
                target_ptr = '<input_tensor_ptr>'
            else:
                target_ptr = '<input_tensor_ptr{0}>'.format(write_id)
        else:
            suffix = 'output'
            if write_id == 0:
                target_ptr = '<output_tensor_ptr>'
            else:
                target_ptr = '<output_tensor_ptr{0}>'.format(write_id)
            
        if tfcom.check_value_in_dict(tensor_id, io_tensors):
            in_var_name = io_tensors[tensor_id][1] # 
            io_tensors[tensor_id].append(self.id)
            op_params = op_params.replace(target_ptr, '&'+in_var_name)
        else:
            in_var_name = prefix + '_' + str(self.id) + '_' + suffix
            
            if inplace_id != -1:  # For inplace op, Assign input to output.
                inplace_var_name = io_tensors[inplace_id][1]
                io_tensors[tensor_id] = [tensor, in_var_name, inplace_var_name, self.id]
                tensor_str = tfcom.format_tensor(tensor, tensor_id, inplace_var_name+".data")
            else:    
                io_tensors[tensor_id] = [tensor, in_var_name, tfcom.get_tensor_size(tensor), self.id]
                tensor_str = tfcom.format_tensor(tensor, tensor_id, 'NULL')
                
            op_params = op_params.replace(target_ptr, '&'+in_var_name)
            
            tensor_str = 'Tensor ' + in_var_name + ' = ' + tensor_str + ';\n'
            fp.write(tensor_str)
        
        return op_params

    def export_io_tensors(self, name_prefix, op_params, io_tensors, is_inplace, fp):
        input_tensors = []
        output_tensors = []
        
        for id in self.attr["input_index"]:
            input_tensor_id = self.op.Inputs(id)
            input_tensor = self.graph.Tensors(input_tensor_id)
            op_params = self.export_io_tensor(name_prefix, 'input', input_tensor, input_tensor_id, io_tensors, op_params, fp['model'], -1, id)
            input_tensors.append(input_tensor)
            
        inplace_id = -1
        if ((is_inplace is True) and (len(self.attr["input_index"]) == 1)):
            inplace_id = input_tensor_id
        
        for id in self.attr["output_index"]:
            output_tensor_id = self.op.Outputs(id)
            output_tensor = self.graph.Tensors(output_tensor_id)
            op_params = self.export_io_tensor(name_prefix, 'output', output_tensor, output_tensor_id, io_tensors, op_params, fp['model'], inplace_id, id)
            output_tensors.append(output_tensor)
        
        if len(input_tensors) == 1 and len(output_tensors) == 1:
            return op_params, input_tensors[0], output_tensors[0]
        return op_params, input_tensors, output_tensors
    
    def export_weight_quant(self, name_prefix, model, op_params, fp):
        weights_tensor_id = self.op.Inputs(self.attr["weight_index"])
        weights_tensor = self.graph.Tensors(weights_tensor_id)
        weights_buffer = model.Buffers(weights_tensor.Buffer())
        assert(weights_tensor.Type() == tflite.TensorType.INT8)
        
        weight_data = np.frombuffer(weights_buffer.DataAsNumpy(), dtype=np.int8)
        # weight_scale = weights_tensor.Quantization().ScaleAsNumpy() # .Scale(0)
        weight_zero_point = weights_tensor.Quantization().ZeroPointAsNumpy() #.ZeroPoint(0)
        op_params = op_params.replace('<weights_offset>', str(weight_zero_point))
            
        weight_str, weight_var_name = tfcom.format_weight_bias(weight_data, weights_tensor.Type(), name_prefix + "_weights_" + str(self.id))
        fp["params"].write(weight_str)

        filter_tensor_str = tfcom.format_tensor(weights_tensor, weights_tensor_id, weight_var_name)
        op_params = op_params.replace('<filter_tensor>', filter_tensor_str)
        
        return op_params, weights_tensor
    
    def export_bias_quant(self, name_prefix, model, op_params, fp):
        bias_tensor_id = self.op.Inputs(self.attr["bias_index"])
        bias_tensor = self.graph.Tensors(bias_tensor_id)
        bias_buffer = model.Buffers(bias_tensor.Buffer())

        assert(bias_tensor.Type() == tflite.TensorType.INT32)
        bias_data = np.frombuffer(bias_buffer.DataAsNumpy(), dtype=np.int32) 

        bias_str, bias_var_name = tfcom.format_weight_bias(bias_data, bias_tensor.Type(), name_prefix + "_bias_" + str(self.id))
        fp["params"].write(bias_str)

        bias_tensor_str = tfcom.format_tensor(bias_tensor, bias_tensor_id, bias_var_name)
        op_params = op_params.replace('<bias_tensor>', bias_tensor_str)
        
        return op_params, bias_tensor
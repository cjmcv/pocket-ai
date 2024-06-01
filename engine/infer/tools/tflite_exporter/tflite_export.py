
import os
import sys
import argparse
import numpy as np
import tflite

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(CURRENT_PATH + "/../")
print(sys.path)

#
import tflite_exporter.common as tfcom
from tflite_exporter.operators.operator import Operator
from tflite_exporter.operators.conv2d import Conv2D
from tflite_exporter.operators.max_pooling import MaxPooling
from tflite_exporter.operators.reshape import Reshape
from tflite_exporter.operators.fully_connected import FullyConnected
#

ending_debug_op = 4

class Add(Operator):
    def __init__(self, graph, op, op_id):
        super().__init__(graph, op, op_id)
        self.attr["code"] = tflite.BuiltinOperator.ADD
        
        self.attr["input_index"] = [0, 1]
        self.attr["output_index"] = [0]
        
class Split(Operator):
    def __init__(self, graph, op, op_id):
        super().__init__(op, graph, op_id)
        self.attr["axis_index"] = 0
        self.attr["input_index"] = [1]
        self.attr["output_index"] = []
        for i in range(op.OutputsLength()):
            self.attr["output_index"].append(i)

class TransposeConv(Operator):
    def __init__(self, graph, op, op_id):
        super().__init__(graph, op, op_id)
        self.attr["code"] = tflite.BuiltinOperator.TRANSPOSE_CONV
        
        self.attr["input_index"] = [2]
         
BUILDINCODE2OP = {
    tflite.BuiltinOperator.SPLIT: Split,
    tflite.BuiltinOperator.CONV_2D: Conv2D,
    tflite.BuiltinOperator.MAX_POOL_2D: MaxPooling,
    tflite.BuiltinOperator.RESHAPE: Reshape,
    tflite.BuiltinOperator.FULLY_CONNECTED: FullyConnected,
    tflite.BuiltinOperator.TRANSPOSE_CONV: TransposeConv,
}

class TfliteExporter:
    
    def create_lifetime(self, index, size):
        # [lower, upper), lifetime of the tensor, marked by operator index
        tensor_lifetime = {
                    'index': index,
                    'lower': -1,
                    'upper': -1,
                    'size': size,
                }
        return tensor_lifetime
    
    def tensor_list_update_start(self, all_tensors, index, i):
        for tensor in all_tensors:
            if tensor['index'] == index:
                tensor['lower'] = i     # for [

    def tensor_list_update_end(self, all_tensors, index, i):
        for tensor in all_tensors:
            if tensor['index'] == index:
                tensor['upper'] = i + 1 # for )

    def code2op_exporter(self, graph, code, op, op_id):
        return BUILDINCODE2OP[code](graph, op, op_id)
        
    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            buf = f.read()
            self.model = tflite.Model.GetRootAsModel(buf, 0)
            
            assert(self.model.SubgraphsLength() == 1)
            subgraph = self.model.Subgraphs(0)
            
            # [tensor, tensor_name, tensor_size, 0/1 is allocate memory, op_id0, op_id1, op_id2...]
            self.io_tensors = {}
            for gin_id in range(subgraph.InputsLength()):
                tensor_id = subgraph.Inputs(gin_id)
                tensor = subgraph.Tensors(tensor_id)
                in_var_name = "graph_input_" + str(gin_id)
                self.io_tensors[tensor_id] = [tensor, in_var_name, tfcom.get_tensor_size(tensor)]
                
            for gout_id in range(subgraph.OutputsLength()):
                tensor_id = subgraph.Outputs(gout_id)
                tensor = subgraph.Tensors(tensor_id)
                in_var_name = "graph_output_" + str(gout_id)
                self.io_tensors[tensor_id] = [tensor, in_var_name, tfcom.get_tensor_size(tensor)]
                
            self.op_exporters = []    
            for i in range(subgraph.OperatorsLength()):
                operator = subgraph.Operators(i)
                op_code = self.model.OperatorCodes(operator.OpcodeIndex())
                op_exporter = self.code2op_exporter(subgraph, op_code.BuiltinCode(), operator, i)
                self.op_exporters.append(op_exporter)
                
    def print_tensor_info(self, graph, tensor_id):
        tensor = graph.Tensors(tensor_id)
        print("    ", tensor_id, tensor.Name().decode('utf-8'), " -> ", tfcom.get_tensor_type_name(tensor.Type()), tensor.ShapeAsNumpy())
        
    def print_model_info(self):
        for graph_id in range(self.model.SubgraphsLength()):
            subgraph = self.model.Subgraphs(graph_id)
            print("Subgraph Input tensors:")
            for gin_id in range(subgraph.InputsLength()):
                self.print_tensor_info(subgraph, subgraph.Inputs(gin_id))
            print("Subgraph Output tensors:")
            
            for gout_id in range(subgraph.OutputsLength()):
                self.print_tensor_info(subgraph, subgraph.Outputs(gout_id))    
                
            print("\n#################################")
            for op_id in range(subgraph.OperatorsLength()):
                op = subgraph.Operators(op_id)
                op_code = self.model.OperatorCodes(op.OpcodeIndex())
                print("\n", op_id, tflite.opcode2name(op_code.BuiltinCode()), "[", op.InputsLength(), op.OutputsLength(), "]")
                print("  Input tensors:")
                for i in range(op.InputsLength()):
                    self.print_tensor_info(subgraph, op.Inputs(i))
                print("  Output tensors:")
                for i in range(op.OutputsLength()):
                    self.print_tensor_info(subgraph, op.Outputs(i))
            print("\n#################################\n")
        
    def gen_tensor_lifetime(self):
        for graph_id in range(self.model.SubgraphsLength()):
            subgraph = self.model.Subgraphs(graph_id)
        
            tensors_lifetime = []
            for i in range(subgraph.OperatorsLength()):
                operator = subgraph.Operators(i)
                
                op_code = self.model.OperatorCodes(operator.OpcodeIndex())
                op = self.code2op_exporter(subgraph, op_code.BuiltinCode(), operator, i)
                print("get", tflite.opcode2name(op.attr["code"]))
                
                # 作为该节点的输出，则该节点作为其生命周期的起点（目前认为所有的output tensor都是输出）
                output_idx = op.attr["output_index"] # if isinstance(output_idx, list):
                for idx in output_idx:
                    output_tensor_size = tfcom.get_tensor_size(subgraph.Tensors(operator.Outputs(idx)))
                    tensor_lifetime = self.create_lifetime(operator.Outputs(idx), output_tensor_size)
                    tensors_lifetime.append(tensor_lifetime)
                    self.tensor_list_update_start(tensors_lifetime, operator.Outputs(idx), i)

                # 作为该节点的输入，则生命周期需包含该节点, 标记为结束点，后续后更晚的再继续更新
                input_idx = op.attr["input_index"]
                for idx in input_idx:
                    self.tensor_list_update_end(tensors_lifetime, operator.Inputs(idx), i) 
                    
            ####  
            tensors_lifetime_selected = []
            for tensor_lifetime in tensors_lifetime:
                if tensor_lifetime['upper'] != 1:
                    tensors_lifetime_selected.append(tensor_lifetime)

            for tensor_lifetime in tensors_lifetime_selected:
                print(tensor_lifetime)

            # Save to csv
            file = open("./tensor_lifetime.csv", "w")
            file.write("id, lower, upper, size\n")
            for tensor in tensors_lifetime_selected:
                tensor_result = str(tensor["index"]) + ',' + str(tensor['lower']) + ',' + \
                                str(tensor['upper']) + ',' + str(tensor['size']) + '\n'
                file.write(tensor_result)
            file.close()

    def include_op_header(self, fp):
        assert(self.model.SubgraphsLength() == 1)
        selected_header = []
        for op in self.op_exporters:
            if op.is_quant():
                header = op.header_quant
            else:
                header = op.header_float
                
            if header not in selected_header:
                selected_header.append(header)
            if op.op_id() == ending_debug_op:
                break
        for h in selected_header:
            fp.write(h)
            
    def export_model(self, output_path, model_file = "exported_model"):
        model_tag = model_file
        model_params_file = output_path + model_file + "_params.h"
        model_file = output_path + model_file + ".h"
        
        fp = {}
        # Header head
        fp["model"] = open(model_file, "w")
        fp["model"].write('#ifndef POCKET_AI_ENGINE_INFERENCE_{0}_STRUCT_HPP_\n'.format(model_tag.upper()))
        fp["model"].write('#define POCKET_AI_ENGINE_INFERENCE_{0}_STRUCT_HPP_\n\n'.format(model_tag.upper()))
        fp["model"].write('#include "engine/infer/types.hpp"\n')
        fp["model"].write('#include "engine/infer/common.hpp"\n')
        fp["model"].write('#include \"{0}\"\n\n'.format(model_params_file))
        self.include_op_header(fp["model"])
        fp["model"].write('\nnamespace pai {\n')
        fp["model"].write('namespace infer {\n')
        fp["model"].write('namespace {0} {{\n\n'.format(model_tag))
        
        fp["params"] = open(model_params_file, "w")
        fp["params"].write('#ifndef POCKET_AI_ENGINE_INFERENCE_{0}_PARAMS_HPP_\n'.format(model_tag.upper()))
        fp["params"].write('#define POCKET_AI_ENGINE_INFERENCE_{0}_PARAMS_HPP_\n\n'.format(model_tag.upper()))
        fp["params"].write('#include <stdint.h>\n\n')
        fp["params"].write('namespace pai {\n')
        fp["params"].write('namespace infer {\n\n')
        fp["params"].write('namespace {0} {{\n\n'.format(model_tag))
        
        # Graph input/output tensors
        # The inputs and outputs of the graph are taken out in the loadmodel function to self.io_tensors
        fp["model"].write('// graph io tensor\n')
        for id in self.io_tensors:
            tensor = self.io_tensors[id][0]
            tensor_name =  self.io_tensors[id][1]   
            tensor_str = tfcom.format_tensor(tensor, id, 'NULL')
            tensor_str = 'Tensor ' + tensor_name + ' = ' + tensor_str + ';\n'
            fp["model"].write(tensor_str)
            
            tensor_str = 'uint32_t ' + tensor_name + '_size = ' + str(tfcom.get_tensor_size(tensor)) + ';\n'
            fp["model"].write(tensor_str)
        fp["model"].write('\n')
        
        # Export params of each op
        for op in self.op_exporters:
            op.export(fp, self.model, self.io_tensors)
            if op.op_id() == ending_debug_op:
                break
        
        
        for id in self.io_tensors:
            print(id, end=": ")
            for op_id in self.io_tensors[id]:
                print(op_id, end=", ")
            print()
            
        # Set Init
        is_malloc = True # True / False
        fp["model"].write('void Init() {\n')
        for id in self.io_tensors:
            tensor_name = self.io_tensors[id][1]
            tensor_size = self.io_tensors[id][2]
            
            # If type(tensor_size) is str, tensor_size will be the src tensor of inplace op
            if type(tensor_size) is str:
                # fp["model"].write("    // Reshape inplace: {0} <- {1}\n".format(tensor_name, tensor_size))
                fp["model"].write("    {0}.data = {1}.data; // Inplace\n".format(tensor_name, tensor_size))
                continue
                
            if is_malloc is True:
                tensor_str = "    {0}.data = (void *)malloc({1});".format(tensor_name, str(tensor_size)) + "\n"
                tensor_str = tensor_str + "    memset({0}.data, 0, {1});".format(tensor_name, str(tensor_size)) + "\n"
            else:
                base = 0x200000
                offset = 100
                tensor_str = "    {0}.data = (void *)({1}+{2});".format(tensor_name, str(base), str(offset)) + "\n"
            fp["model"].write(tensor_str)
        fp["model"].write('}\n')

        # Set DeInit
        fp["model"].write('void Deinit() {\n')
        for id in self.io_tensors:
            tensor_name = self.io_tensors[id][1]
            tensor_size = self.io_tensors[id][2]
            if type(tensor_size) is str:
                fp["model"].write("    // Reshape inplace: {0} <- {1}\n".format(tensor_name, tensor_size))
                continue
            
            tensor_str = ";"
            if is_malloc is True:
                tensor_str = "    free({}.data);".format(tensor_name) + "\n"
            fp["model"].write(tensor_str)
        fp["model"].write('}\n')
                        
        # Set Run. The smaller the ID, the earlier it is executed
        fp["model"].write('void Run() {\n')
        for op in self.op_exporters:
            fp["model"].write("    " + op.oprun_str + "\n")
            if op.op_id() == ending_debug_op:
                break
        fp["model"].write('}\n')
        
        # Header tail
        fp["params"].write('\n} // namespace model_tag')
        fp["params"].write('\n} // namespace pai')
        fp["params"].write('\n} // namespace infer')
        fp["params"].write('\n\n#endif // POCKET_AI_ENGINE_INFERENCE_{0}_PARAMS_HPP_\n'.format(model_tag.upper()))
        fp["params"].close()
        
        fp["model"].write('\n} // namespace model_tag')
        fp["model"].write('\n} // namespace pai')
        fp["model"].write('\n} // namespace infer')
        fp["model"].write('\n\n#endif // POCKET_AI_ENGINE_INFERENCE_{0}_STRUCT_HPP_\n'.format(model_tag.upper()))
        fp["model"].close()
        
        print("\nExport completed.\n")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--model_path", type=str, default="../models/tf_micro_conv_test_model.int8.tflite")
    parser.add_argument("--output_path", type=str, default="./")
    parser.add_argument("--model_tag", type=str, default="exported_model")
    args = parser.parse_args()

    exporter = TfliteExporter()
    exporter.load_model(args.model_path)
    exporter.print_model_info()
    exporter.gen_tensor_lifetime()
    exporter.export_model(args.output_path, args.model_tag)
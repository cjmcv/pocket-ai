
import os
import sys
import argparse
import numpy as np
from functools import reduce
import math
import tflite

#
from operators.operator import Operator
from operators.conv2d import Conv2D
#

# CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
# sys.path.insert(1, os.path.jooin(CURRENT_PATH, "py_infer/"))

BUILTIN_TENSORTYPE2NAME = {
    tflite.TensorType.UINT8: 'UINT8',
    tflite.TensorType.INT8: 'INT8',
    tflite.TensorType.UINT16: 'UINT16',
    tflite.TensorType.INT16: 'INT16',
    tflite.TensorType.UINT32: 'UINT32',
    tflite.TensorType.INT32: 'INT32',
    tflite.TensorType.FLOAT32: 'FLOAT32',
}

BUILTIN_TENSORTYPE2SIZE = {
    tflite.TensorType.UINT8: 1,
    tflite.TensorType.INT8: 1,
    tflite.TensorType.UINT16: 2,
    tflite.TensorType.INT16: 2,
    tflite.TensorType.UINT32: 4,
    tflite.TensorType.INT32: 4,
    tflite.TensorType.FLOAT32: 4,
}

class Add(Operator):
    def __init__(self, op, op_id):
        super().__init__(op, op_id)
        self.attr["code"] = tflite.BuiltinOperator.ADD
        
        self.attr["input_index"] = [0, 1]
        self.attr["output_index"] = [0]
        
class Split(Operator):
    def __init__(self, op, op_id):
        super().__init__(op, op_id)
        self.attr["axis_index"] = 0
        self.attr["input_index"] = [1]
        self.attr["output_index"] = []
        for i in range(op.OutputsLength()):
            self.attr["output_index"].append(i)
        
class MaxPool2D(Operator):
    def __init__(self, graph, op, op_id):
        super().__init__(graph, op, op_id)
        self.attr["code"] = tflite.BuiltinOperator.MAX_POOL_2D
        
        self.attr["input_index"] = [0]
        self.attr["output_index"] = [0]
        
class Reshape(Operator):
    def __init__(self, graph, op, op_id):
        super().__init__(graph, op, op_id)
        self.attr["code"] = tflite.BuiltinOperator.RESHAPE
        
        self.attr["input_index"] = [0]
        self.attr["output_index"] = [0]
    
class FullyConnected(Operator):
    def __init__(self, graph, op, op_id):
        super().__init__(graph, op, op_id)
        self.attr["code"] = tflite.BuiltinOperator.FULLY_CONNECTED
        
        self.attr["input_index"] = [0]
        self.attr["weight_index"] = 1
        self.attr["bias_index"] = 2
        
        self.attr["output_index"] = [0]

class TransposeConv(Operator):
    def __init__(self, graph, op, op_id):
        super().__init__(graph, op, op_id)
        self.attr["code"] = tflite.BuiltinOperator.TRANSPOSE_CONV
        
        self.attr["input_index"] = [2]
         
BUILDINCODE2OP = {
    tflite.BuiltinOperator.SPLIT: Split,
    tflite.BuiltinOperator.CONV_2D: Conv2D,
    tflite.BuiltinOperator.MAX_POOL_2D: MaxPool2D,
    tflite.BuiltinOperator.RESHAPE: Reshape,
    tflite.BuiltinOperator.FULLY_CONNECTED: FullyConnected,
    tflite.BuiltinOperator.TRANSPOSE_CONV: TransposeConv,
}

class TfliteExporter:

    def get_output_tensor_size(self, tensor):
        size = reduce(math.mul, tensor.ShapeAsNumpy(), 1)
        size *= BUILTIN_TENSORTYPE2SIZE[tensor.Type()]
        return size

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

    def tensortype2name(self, tensor_type):
        return BUILTIN_TENSORTYPE2NAME[tensor_type]

    def code2op_exporter(self, graph, op_code, op, op_id):
        return BUILDINCODE2OP[op_code.BuiltinCode()](graph, op, op_id)
        
    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            buf = f.read()
            self.model = tflite.Model.GetRootAsModel(buf, 0)
            
            self.op_exporters = []
            assert(self.model.SubgraphsLength() == 1)
            subgraph = self.model.Subgraphs(0)
            for i in range(subgraph.OperatorsLength()):
                operator = subgraph.Operators(i)
                op_code = self.model.OperatorCodes(operator.OpcodeIndex())
                op_exporter = self.code2op_exporter(subgraph, op_code, operator, i)
                self.op_exporters.append(op_exporter)
                
    def print_tensor_info(self, graph, tensor_id):
        tensor = graph.Tensors(tensor_id)
        print("    ", tensor_id, tensor.Name().decode('utf-8'), " -> ", self.tensortype2name(tensor.Type()), tensor.ShapeAsNumpy())
        
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
                op = self.code2op_exporter(op_code, operator, i)
                print("get", tflite.opcode2name(op.attr["code"]))
                
                # 作为该节点的输出，则该节点作为其生命周期的起点（目前认为所有的output tensor都是输出）
                output_idx = op.attr["output_index"] # if isinstance(output_idx, list):
                for idx in output_idx:
                    output_tensor_size = self.get_output_tensor_size(subgraph.Tensors(operator.Outputs(idx)))
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
            if op.op_id == 1:
                break
        for h in selected_header:
            fp.write(h)
            
    def export_model(self, output_path, model_file = "exported_model"):
        model_tag = model_file
        model_params_file = output_path + model_file + "_params.h"
        model_file = output_path + model_file + ".h"
        
        fp = {}
        fp["model"] = open(model_file, "w")
        fp["model"].write('#ifndef POCKET_AI_ENGINE_INFERENCE_{0}_STRUCT_HPP_\n'.format(model_tag.upper()))
        fp["model"].write('#define POCKET_AI_ENGINE_INFERENCE_{0}_STRUCT_HPP_\n\n'.format(model_tag.upper()))
        fp["model"].write('#include "engine/infer/types.hpp"\n')
        fp["model"].write('#include "engine/infer/common.hpp"\n')
        fp["model"].write('#include \"{0}\"\n\n'.format(model_params_file))
        self.include_op_header(fp["model"])
        fp["model"].write('\nnamespace pai {\n')
        fp["model"].write('namespace infer {\n\n')
        
        fp["params"] = open(model_params_file, "w")
        fp["params"].write('#ifndef POCKET_AI_ENGINE_INFERENCE_{0}_PARAMS_HPP_\n'.format(model_tag.upper()))
        fp["params"].write('#define POCKET_AI_ENGINE_INFERENCE_{0}_PARAMS_HPP_\n\n'.format(model_tag.upper()))
        fp["params"].write('#include <stdint.h>\n\n')
        fp["params"].write('namespace pai {\n')
        fp["params"].write('namespace infer {\n\n')
        
        for op in self.op_exporters:
            op.export(fp, self.model)
            if op.op_id == 1:
                break
            
        fp["params"].write('\n} // namespace pai')
        fp["params"].write('\n} // namespace infer')
        fp["params"].write('\n\n#endif // POCKET_AI_ENGINE_INFERENCE_{0}_PARAMS_HPP_\n'.format(model_tag.upper()))
        fp["params"].close()
        
        fp["model"].write('\n} // namespace pai')
        fp["model"].write('\n} // namespace infer')
        fp["model"].write('\n\n#endif // POCKET_AI_ENGINE_INFERENCE_{0}_STRUCT_HPP_\n'.format(model_tag.upper()))
        fp["model"].close()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--model_path", type=str, default="../models/tf_micro_conv_test_model.int8.tflite")
    parser.add_argument("--output_path", type=str, default="./")
    parser.add_argument("--model_tag", type=str, default="exported_model")
    args = parser.parse_args()

    exporter = TfliteExporter()
    exporter.load_model(args.model_path)
    # exporter.print_model_info()
    # exporter.gen_tensor_lifetime()
    exporter.export_model(args.output_path, args.model_tag)
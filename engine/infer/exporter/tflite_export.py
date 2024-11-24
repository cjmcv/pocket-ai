
import os
import sys
import argparse
import numpy as np
import tflite

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(CURRENT_PATH + "/../")
print(sys.path)

# https://tensorflow.google.cn/lite/guide/op_select_allowlist?hl=zh-cn
import exporter.common as tfcom
from exporter.operators.fusion.fusion_manager import FusionManager
from exporter.operators.operator import Operator
from exporter.operators.add import Add
from exporter.operators.conv2d import Conv2D
from exporter.operators.depthwise_conv2d import DepthwiseConv2D
from exporter.operators.dequantize import Dequantize
from exporter.operators.div import Div
from exporter.operators.fully_connected import FullyConnected
from exporter.operators.hard_swish import HardSwish
from exporter.operators.lstm import Lstm
from exporter.operators.max_pooling import MaxPooling
from exporter.operators.mean import Mean
from exporter.operators.mul import Mul
from exporter.operators.pad import Pad
from exporter.operators.quantize import Quantize
from exporter.operators.reshape import Reshape
from exporter.operators.softmax import Softmax
from exporter.operators.tile import Tile
from exporter.operators.transpose_conv import TransposeConv
#

ending_debug_op = 1000
        
class Split(Operator):
    def __init__(self, graph, op, op_id):
        super().__init__(op, graph, op_id)
        self.attr["axis_index"] = 0
        self.attr["input_index"] = [1]
        self.attr["output_index"] = []
        for i in range(op.OutputsLength()):
            self.attr["output_index"].append(i)

# class TransposeConv(Operator):
#     def __init__(self, graph, op, op_id):
#         super().__init__(graph, op, op_id)
#         self.attr["code"] = tflite.BuiltinOperator.TRANSPOSE_CONV
        
#         self.attr["input_index"] = [2]
         
BUILDINCODE2OP = {
    tflite.BuiltinOperator.ADD: Add,
    tflite.BuiltinOperator.CONV_2D: Conv2D,
    tflite.BuiltinOperator.DEPTHWISE_CONV_2D: DepthwiseConv2D,
    tflite.BuiltinOperator.DEQUANTIZE: Dequantize,
    tflite.BuiltinOperator.DIV: Div,
    tflite.BuiltinOperator.FULLY_CONNECTED: FullyConnected,
    tflite.BuiltinOperator.HARD_SWISH: HardSwish,
    tflite.BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_LSTM: Lstm,
    tflite.BuiltinOperator.MAX_POOL_2D: MaxPooling,
    tflite.BuiltinOperator.MEAN: Mean,
    tflite.BuiltinOperator.MUL: Mul,
    tflite.BuiltinOperator.PAD: Pad,
    tflite.BuiltinOperator.PADV2: Pad,
    tflite.BuiltinOperator.QUANTIZE: Quantize,
    tflite.BuiltinOperator.RESHAPE: Reshape,
    tflite.BuiltinOperator.SOFTMAX: Softmax,
    tflite.BuiltinOperator.SPLIT: Split,
    tflite.BuiltinOperator.TILE: Tile,
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
        
    # Get io_tensors and op_exporters.
    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            buf = f.read()
            self.model = tflite.Model.GetRootAsModel(buf, 0)
            
            assert(self.model.SubgraphsLength() == 1)
            subgraph = self.model.Subgraphs(0)
            
            self.dynamic_buffer = tfcom.DynamicBuffer()
            # [tensor, tensor_name, tensor_size, 0/1 is allocate memory, op_id0, op_id1, op_id2...]
            for gin_id in range(subgraph.InputsLength()):
                tensor_id = subgraph.Inputs(gin_id)
                tensor = subgraph.Tensors(tensor_id)
                in_var_name = "graph_input_" + str(gin_id)
                self.dynamic_buffer.io_tensors.append(tensor_id, tensor, tfcom.get_tensor_size(tensor), in_var_name) 
                
            for gout_id in range(subgraph.OutputsLength()):
                tensor_id = subgraph.Outputs(gout_id)
                tensor = subgraph.Tensors(tensor_id)
                in_var_name = "graph_output_" + str(gout_id)
                self.dynamic_buffer.io_tensors.append(tensor_id, tensor, tfcom.get_tensor_size(tensor), in_var_name) 
                
            self.op_exporters = []    
            for i in range(subgraph.OperatorsLength()):
                operator = subgraph.Operators(i)
                op_code = self.model.OperatorCodes(operator.OpcodeIndex())
                print('Getting exporter {0}: {1}'.format(i, tflite.opcode2name(op_code.BuiltinCode())))
                op_exporter = self.code2op_exporter(subgraph, op_code.BuiltinCode(), operator, i)
                self.op_exporters.append(op_exporter)
                
                if i == ending_debug_op:
                    break
                
    def print_tensor_info(self, graph, tensor_id):
        tensor = graph.Tensors(tensor_id)
        if (tensor_id != -1):
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
        lifetime_file = output_path + model_file + "_lifetime.csv"
        model_params_file = output_path + model_file + "_params.h"
        model_file = output_path + model_file + ".h"

        fp = {}
        # Header head
        fp["model"] = open(model_file, "w")
        fp["model"].write('#ifndef POCKET_AI_ENGINE_INFERENCE_{0}_STRUCT_HPP_\n'.format(model_tag.upper()))
        fp["model"].write('#define POCKET_AI_ENGINE_INFERENCE_{0}_STRUCT_HPP_\n\n'.format(model_tag.upper()))
        fp["model"].write('#include <string.h>\n')
        fp["model"].write('#include <float.h>\n')
        fp["model"].write('#include "engine/infer/types.hpp"\n')
        fp["model"].write('#include "engine/infer/common.hpp"\n')
        fp["model"].write('#include \"{0}\"\n\n'.format(model_params_file))
        self.include_op_header(fp["model"])
        fp["model"].write('\nnamespace pai {\n')
        fp["model"].write('namespace infer {\n')
        fp["model"].write('namespace {0} {{\n\n'.format(model_tag))
        fp["model"].write('char *{0} = nullptr;\n'.format(self.dynamic_buffer.scratch_buffer_name))
        
        fp["params"] = open(model_params_file, "w")
        fp["params"].write('#ifndef POCKET_AI_ENGINE_INFERENCE_{0}_PARAMS_HPP_\n'.format(model_tag.upper()))
        fp["params"].write('#define POCKET_AI_ENGINE_INFERENCE_{0}_PARAMS_HPP_\n\n'.format(model_tag.upper()))
        fp["params"].write('#include <stdint.h>\n\n')
        fp["params"].write('#include <math.h>\n\n')
        fp["params"].write('namespace pai {\n')
        fp["params"].write('namespace infer {\n\n')
        fp["params"].write('namespace {0} {{\n\n'.format(model_tag))
        
        # Graph input/output tensors
        # The inputs and outputs of the graph are taken out in the loadmodel function to self.dynamic_buffer.io_tensors
        fp["model"].write('// graph io tensor\n')
        for io_tensor in self.dynamic_buffer.io_tensors.ins:
            id = io_tensor['id']
            tensor = io_tensor['obj']
            tensor_name = io_tensor['name'] 
            tensor_str = tfcom.format_tensor(tensor, id, 'NULL')
            tensor_str = 'Tensor ' + tensor_name + ' = ' + tensor_str + ';\n'
            fp["model"].write(tensor_str)
            
            tensor_str = 'uint32_t ' + tensor_name + '_size = ' + str(tfcom.get_tensor_size(tensor)) + ';\n'
            fp["model"].write(tensor_str)
        fp["model"].write('\n')
        
        # Detect 
        fm = FusionManager()
        fm.detect_fusible_ops(self.model, self.op_exporters)
        
        # Export params of each op
        for op in self.op_exporters:
            op.export(fp, self.model, self.dynamic_buffer)
            if op.op_id() == ending_debug_op:
                break
        
        self.dynamic_buffer.show_all_io_tensors()

        # Set Init
        is_malloc = True # True / False
        fp["model"].write('void Init() {\n')
        if self.dynamic_buffer.scratch_buffer_size != 0:
            if is_malloc is True:
                fp["model"].write('    {0} = (char *)malloc({1});\n'.format(self.dynamic_buffer.scratch_buffer_name, str(self.dynamic_buffer.scratch_buffer_size)))
            else:
                fp["model"].write('    {0} = {Please manually set the memory address};\n'.format(self.dynamic_buffer.scratch_buffer_name))
        
        fp["model"].write(self.dynamic_buffer.scratch_buffer_allocate_info)

        # TODO: 先遍历lifetime，过滤不符合要求的inplace，去掉inplace部分，存文件，生成新的lifetime表。按新表划分空间。
        fp["lifetime"] = open(lifetime_file, "w")
        fp["lifetime"].write('id, lower, upper, size, alignment, name\n')

        for io_tensor in self.dynamic_buffer.io_tensors.ins:
            tensor_name = io_tensor['name']
            tensor_size = io_tensor['size'] 
            tensor_inplace_name = io_tensor['inplace_name']
            tensor_const_name = io_tensor['const_name']
            if tensor_inplace_name != 'NULL':
                fp["model"].write("    {0}.data = {1}.data; // Inplace\n".format(tensor_name, tensor_inplace_name))
                continue
            elif tensor_const_name != 'NULL':
                fp["model"].write("    {0}.data = {1}; // constant \n".format(tensor_name, tensor_const_name))
                continue

            # Else it will be the size of tensor.
            # fp["lifetime"].write()
                    # Check and export lifetime.
            is_found = False
            for lt in self.dynamic_buffer.lifetime:
                if lt['id'] == id:
                    lt_str = "{0},{1},{2},{3},{4},{5}".format(str(lt['id']), str(lt['start']), str(lt['end']), str(lt['size']), '16', tensor_name)
                    fp["lifetime"].write(lt_str + '\n')
                    is_found = True
            if is_found == False:
                print("Warning: Lifetime calculation error, target tensor not found.\n")

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
        if self.dynamic_buffer.scratch_buffer_size != 0 and is_malloc is True:
            fp["model"].write('    free({0});\n'.format(self.dynamic_buffer.scratch_buffer_name))
        for io_tensor in self.dynamic_buffer.io_tensors.ins:
            tensor_name = io_tensor['name']
            tensor_size = io_tensor['size'] 
            tensor_inplace_name = io_tensor['inplace_name']
            tensor_const_name = io_tensor['const_name']
            if tensor_inplace_name != 'NULL':
                fp["model"].write("    // Reshape inplace: {0} <- {1}\n".format(tensor_name, tensor_inplace_name))
                continue
            elif tensor_const_name != 'NULL':
                fp["model"].write("    // {0}.data = {1}; // constant \n".format(tensor_name, tensor_const_name))
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
    args = parser.parse_args()

    (filepath, filename) = os.path.split(args.model_path)
    model_tag = os.path.splitext(filename)[0]+"_model"
    model_tag = model_tag.replace('.', '_')

    exporter = TfliteExporter()
    exporter.load_model(args.model_path) # Get io_tensors and op_exporters.
    exporter.print_model_info()          # Show model's information
    exporter.export_model(args.output_path, model_tag) # Loop and execute each op_exporter
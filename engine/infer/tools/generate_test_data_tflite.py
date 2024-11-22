import os
import argparse
import csv
import numpy as np
import tensorflow as tf

class TfliteInference:
    def hello(self):
        print("hello TfliteInference.")

    def load_model(self, model_path, model_tag):
        self.model_tag = model_tag
        if os.path.exists(model_path) is False:
            print("Error: Can not find", model_path)
        self.interpreter = tf.lite.Interpreter(model_path, experimental_preserve_all_tensors=True)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        # self.tensor_details = self.interpreter.get_tensor_details()
        # print("tensor_details: ", self.tensor_details)
        self.interpreter.allocate_tensors()
        self.inputs = {}
        self.outputs = {}
        for i in range(len(self.input_details)):
            print("in<{0}>: {1}, id: {2}, shape: {3}".format(i, self.input_details[i]['name'], self.input_details[i]['index'], self.input_details[i]['shape']))
        for i in range(len(self.output_details)):
            print("out<{0}>: {1}, id: {2}, shape: {3}".format(i, self.output_details[i]['name'], self.output_details[i]['index'], self.output_details[i]['shape']))

    def fill_one2inputs(self, fp):
        for i in range(len(self.input_details)):
            if(self.input_details[i]['dtype'] is np.float32):
                data = np.ones(self.input_details[i]['shape'], np.float32)
                data_c = ",".join(map(str, data.flatten()))
                fp.write("const float test_graph_in{0}_{1}[] __attribute__((aligned(16))) = {{{2}}};\n\n".format(str(i), str(self.input_details[i]['index']), data_c))
            else:
                data = np.ones(self.input_details[i]['shape'], np.int8)
                data_c = ",".join(map(str, data.flatten()))
                fp.write("const int8_t test_graph_in{0}_{1}[] __attribute__((aligned(16))) = {{{2}}};\n\n".format(str(i), str(self.input_details[i]['index']), data_c))
            self.inputs[i] = data
        return self.inputs
        
    def fill_random_inputs(self, fp):
        for i in range(len(self.input_details)):
            if(self.input_details[i]['dtype'] is np.float32):
                data = np.random.rand(*self.input_details[i]['shape'])
                data = data.astype(np.float32)
                data_c = ",".join(map(str, data.flatten()))
                fp.write("const float test_graph_in{0}_{1}[] __attribute__((aligned(16))) = {{{2}}};\n\n".format(str(i), str(self.input_details[i]['index']), data_c))
            else:
                data = np.random.randint(0, 255, self.input_details[i]['shape'])
                data = data.astype(np.int8)
                data_c = ",".join(map(str, data.flatten()))
                fp.write("const int8_t test_graph_in{0}_{1}[] __attribute__((aligned(16))) = {{{2}}};\n\n".format(str(i), str(self.input_details[i]['index']), data_c))
            self.inputs[i] = data
        return self.inputs
    
    def export_output(self, fp):
        for i in range(len(self.output_details)):     
            tensor = self.outputs[i]
            if self.output_details[i]["dtype"] is np.float32:
                data_c = ", ".join(map(str, [f"{x:.7f}" for x in tensor.flatten()]))
                fp.write("const float test_graph_out{0}_{1}[] __attribute__((aligned(16))) = {{{2}}};\n\n".format(str(i), str(self.output_details[i]['index']), data_c))
            elif self.output_details[i]["dtype"] is np.int32:
                data_c = ", ".join(map(str, tensor.flatten()))
                fp.write("const int32_t test_graph_out{0}_{1}[] __attribute__((aligned(16))) = {{{2}}};\n\n".format(str(i), str(self.output_details[i]['index']), data_c))
            elif self.output_details[i]["dtype"] is np.int8:
                data_c = ", ".join(map(str, tensor.flatten()))
                fp.write("const int8_t test_graph_out{0}_{1}[] __attribute__((aligned(16))) = {{{2}}};\n\n".format(str(i), str(self.output_details[i]['index']), data_c))

    def export_all_output_tensor(self, fp, fp_lifetime):
        next(fp_lifetime) # Skip the first row
        lines = csv.reader(fp_lifetime)
        test_data_map = {}
        for line in lines:
            i = int(line[0])
            name = line[5]

            # Skip the io tensors
            is_io_tensor = False
            for j in range(len(self.output_details)):
                if i == self.output_details[j]['index']:
                    is_io_tensor = True
            for j in range(len(self.input_details)):
                if i == self.input_details[j]['index']:
                    is_io_tensor = True
            
            if is_io_tensor == False:
                tensor = self.interpreter.get_tensor(i)
                if tensor.dtype == np.float32:
                    data_c = ", ".join(map(str, [f"{x:.7f}" for x in tensor.flatten()]))
                    fp.write("const float test_tensor_data{0}[] __attribute__((aligned(16))) = {{{1}}};\n\n".format(str(i), data_c))
                elif tensor.dtype == np.int32:
                    data_c = ", ".join(map(str, tensor.flatten()))
                    fp.write("const int32_t test_tensor_data{0}[] __attribute__((aligned(16))) = {{{1}}};\n\n".format(str(i), data_c))
                elif tensor.dtype == np.int8:
                    data_c = ", ".join(map(str, tensor.flatten()))
                    fp.write("const int8_t test_tensor_data{0}[] __attribute__((aligned(16))) = {{{1}}};\n\n".format(str(i), data_c))

                test_data_map[name] = "test_tensor_data{0}".format(str(i))

        # for key in test_data_map:
        #     print(key, test_data_map[key])
        return test_data_map

    def run(self):
        for i in range(len(self.input_details)):
            self.interpreter.set_tensor(self.input_details[i]['index'], self.inputs[i])
        self.interpreter.invoke()
        for i in range(len(self.output_details)):
            self.outputs[i] = self.interpreter.get_tensor(self.output_details[i]['index'])

    def export_test_function(self, fp, test_data_map):
        fp.write("void TEST_{0}() {{\n".format(self.model_tag.upper()))
        fp.write("    using namespace pai::infer;\n")
        fp.write("    namespace model = {0};\n".format(self.model_tag))
        fp.write("    model::Init();\n")
        for i in range(len(self.input_details)):
            if(self.input_details[i]['dtype'] is np.float32):
                type_str = "float"
            elif(self.input_details[i]['dtype'] is np.int8):
                type_str = "int8_t"
            fp.write("    memcpy(model::graph_input_{0}.data, model::test_graph_in{0}_{1}, sizeof({2}) * GetShapeFlatSize(model::graph_input_{0}.shape));\n".format(str(i), str(self.input_details[i]['index']), type_str))
        fp.write("    model::Run();\n")
        for i in range(len(self.output_details)):
            fp.write("    CheckTensor(model::graph_output_{0}, (void*)model::test_graph_out{0}_{1});\n".format(str(i), str(self.output_details[i]['index'])))

        for key in test_data_map:
            fp.write("    CheckTensor(model::{0}, (void*)model::{1});\n".format(key, test_data_map[key]))

        fp.write("    model::Deinit();\n")
        fp.write("}\n")

    ################
    
    def print_tensor(self, tensor_id):
        print("Tensor id: ", tensor_id, self.interpreter.get_tensor(tensor_id).shape)
        print("     data: ", self.interpreter.get_tensor(tensor_id))
    
    ################

    def get_npy_data(self, npy_data_file):
        data = np.load(npy_data_file)
        print(data.shape)
        print(data)
        return data
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="./models/tf_micro_conv_test_model.int8.tflite")
    parser.add_argument('--input_one', action="store_true", help="Fix all input data to 1. Else fill random inputs")
    parser.add_argument('--debug_all_tensors', action="store_true", help="Export all tensors.")
    args = parser.parse_args()
    
    (filepath, filename) = os.path.split(args.model_path)
    model_tag = os.path.splitext(filename)[0]+"_model"
    model_tag = model_tag.replace('.', '_')
    
    target_file = "./" + str(model_tag) + "_test_data.h"
    if os.path.exists(target_file):
        os.remove(target_file)

    infer = TfliteInference()
    infer.load_model(args.model_path, model_tag)
    with open(target_file, "a") as fp:

        fp.write('#include "{0}.h"\n'.format(model_tag))
        fp.write('namespace pai {\n')
        fp.write('namespace infer {\n')
        fp.write('namespace {0} {{\n\n'.format(model_tag))
    
        if args.input_one is True:
            infer.fill_one2inputs(fp)
        else:
            infer.fill_random_inputs(fp)
        
        infer.run()
        
        infer.export_output(fp)

        test_data_map = None
        if args.debug_all_tensors:
            lifetime_file = "./" + str(model_tag) + "_lifetime.csv"
            with open(lifetime_file, "r") as fp_lifetime:
                test_data_map = infer.export_all_output_tensor(fp, fp_lifetime)

        fp.write('\n}} // namespace {0}'.format(model_tag))
        fp.write('\n} // namespace pai')
        fp.write('\n} // namespace infer')

        # Generate test function
        fp.write('\n\n\n')
        infer.export_test_function(fp, test_data_map)

        # np.set_printoptions(suppress=True, precision=7)
        # print(infer.print_tensor(2))

            
import os
import argparse
import numpy as np
import tensorflow as tf

class TfliteInference:
    def hello(self):
        print("hello TfliteInference.")

    def load_model(self, model_path):
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
    
    def get_io_num(self):
        return len(self.input_details), len(self.output_details)
    
    def get_inputs_detail(self, index):
        # print(self.input_details[index])
        if(self.input_details[index]['dtype'] is np.float32):
            type_size = 4
        else:
            type_size = 1
        return type_size, self.input_details[index]['name'], self.input_details[index]['shape']
    
    def get_outputs_detail(self, index):
        # print(self.output_details[index])
        if(self.output_details[index]['dtype'] is np.float32):
            type_size = 4
        else:
            type_size = 1
        return type_size, self.output_details[index]['name'], self.output_details[index]['shape']
    
    def set_input(self, index, data):
        self.inputs[index] = data
    
    def get_output(self, index):
        return self.outputs[index]
    
    def run(self):
        for i in range(len(self.input_details)):
            self.interpreter.set_tensor(self.input_details[i]['index'], self.inputs[i])

        self.interpreter.invoke()

        for i in range(len(self.output_details)):
            self.outputs[i] = self.interpreter.get_tensor(self.output_details[i]['index'])

        # print("get_tensor shape", self.interpreter.get_tensor(10).shape)
        # print("get_tensor data", self.interpreter.get_tensor(10).reshape(-1).tolist())
    ################
    
    def print_tensor(self, tensor_id):
        print("Tensor id: ", tensor_id, self.interpreter.get_tensor(tensor_id).shape)
        print("     data: ", self.interpreter.get_tensor(tensor_id))

    def get_tensor(self, tensor_id):
        return self.interpreter.get_tensor(tensor_id)
    
    def fill_one2inputs(self):
        for i in range(len(self.input_details)):
            if(self.input_details[i]['dtype'] is np.float32):
                data = np.ones(self.input_details[i]['shape'], np.float32)
            else:
                data = np.ones(self.input_details[i]['shape'], np.int8)
            self.inputs[i] = data
        return self.inputs
        
    def fill_random_inputs(self):
        for i in range(len(self.input_details)):
            if(self.input_details[i]['dtype'] is np.float32):
                data = np.random.rand(*self.input_details[i]['shape'])
                data = data.astype(np.float32)
            else:
                data = np.random.randint(0, 255, self.input_details[i]['shape'])
                data = data.astype(np.int8)
            self.inputs[i] = data
        return self.inputs
    
    ################

    def get_npy_data(self, npy_data_file):
        data = np.load(npy_data_file)
        print(data.shape)
        print(data)
        return data
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="./models/tf_micro_conv_test_model.int8.tflite")
    parser.add_argument('--input_one', action="store_true", default=False)
    parser.add_argument('--tensor_id', type=int, default=-1)
    args = parser.parse_args()
    
    infer = TfliteInference()
    infer.load_model(args.model)
    if args.input_one is True:
        input = infer.fill_one2inputs()
        # print("input:", input)
    else:
        input = infer.fill_random_inputs()
        print("input:", input)
    
    infer.run()
    output = infer.get_output(0)

    if (args.tensor_id == -1):
        print("output:", output)
    else:
        print(infer.print_tensor(args.tensor_id))

            
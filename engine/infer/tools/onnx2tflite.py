import os
import numpy as np
from onnx_tf.backend import prepare
import onnx
import tensorflow as tf

# https://github.com/onnx/onnx-tensorflow
# pip install onnx
# pip install onnx-tf
# pip install tensorflow_probability
# pip install tensorflow
# https://github.com/onnx/models

if __name__ == "__main__":
    model_dir = '/home/shared_dir/models/'
    model_name = 'mobilenetv2-12-int8'
    
    onnx_model = model_dir + model_name + '.onnx'
    tf_model = model_dir + 'gen/' + model_name + '_tfpb'
    tflite_model = model_dir + 'gen/' + model_name + '.tflite'
    
    model = onnx.load(onnx_model)
    tf_rep = prepare(model)
    tf_rep.export_graph(tf_model)
    
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tf_lite_model = converter.convert()
    with open(tflite_model, 'wb') as f:
        f.write(tf_lite_model)

            
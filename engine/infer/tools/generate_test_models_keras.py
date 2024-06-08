
import os
import numpy as np
import tensorflow as tf

# https://keras.io/api/layers/
def generate_allop_model(filename="conv_test_model_int8.tflite"):
    """Creates a basic Keras model and converts to tflite.

    This model does not make any relevant classifications. It only exists to
    generate a model that is designed to run on embedded devices.
    """
    np.random.seed(0)
    input_shape = (16, 16, 1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation="relu", input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")) 
    model.add(tf.keras.layers.DepthwiseConv2D(kernel_size=3, depth_multiplier=2, activation="relu")) 
    model.add(tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, activation="relu")) # filters, kernel_size,
    model.add(tf.keras.layers.MaxPooling2D(2))
    model.add(tf.keras.layers.AveragePooling2D(2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10))
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    # Test with random data
    data_x = np.random.rand(12, 16, 16, 1)
    data_y = np.random.randint(2, size=(12, 10))
    model.fit(data_x, data_y, epochs=5)

    def representative_dataset_gen():
        np.random.seed(0)
        for _ in range(12):
            yield [np.random.rand(16, 16).reshape(1, 16, 16, 1).astype(np.float32)]

    # Now convert to a TFLite model with full int8 quantization:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.representative_dataset = representative_dataset_gen

    tflite_model = converter.convert()
    
    output_dir = './gen/keras/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    open(output_dir + filename, "wb").write(tflite_model)

    return tflite_model

if __name__ == "__main__":
    generate_allop_model()

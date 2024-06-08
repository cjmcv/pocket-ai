import os
import sys
import argparse

import torch
import torch.nn
# https://pytorch.org/vision/stable/models.html
import torchvision.models as tm
import torchvision.models.quantization as tmq

# Using TinyNeuralNetwork
# https://github.com/alibaba/TinyNeuralNetwork
CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(1, os.path.join('/home/shared_dir/TinyNeuralNetwork'))

from tinynn.converter import TFLiteConverter
from tinynn.graph.tracer import model_tracer, trace
# PTQ
from tinynn.graph.quantization.quantizer import PostQuantizer
from tinynn.util.cifar10 import get_dataloader, calibrate
from tinynn.util.train_util import DLContext, get_device

def save_model(model, dummy_input, out_dir):
    model_name = type(model).__name__.lower()
    
    print(model)
    model.cpu()
    model.eval()
    # When converting quantized models, please ensure the quantization backend is set.
    torch.backends.quantized.engine = 'qnnpack'
    # The code section below is used to convert the model to the TFLite format
    # If you want perform dynamic quantization on the float models,
    # you may refer to `dynamic.py`, which is in the same folder.
    # As for static quantization (e.g. quantization-aware training and post-training quantization),
    # please refer to the code examples in the `examples/quantization` folder.
    converter = TFLiteConverter(model, dummy_input, out_dir + model_name + '.tflite')
    converter.convert()
    
    with model_tracer():
        # After tracing the model, we will get a TraceGraph object
        graph = trace(model, dummy_input)
        # We can use it to generate the code for the original model
        # But be careful that it has to be in the with-block.
        graph.generate_code(out_dir + model_name + '.py',  out_dir + model_name + '.pth', model_name.title())

def post_quant(model, dummy_input, out_dir):
    model_name = type(model).__name__.lower()
    with model_tracer():
        # TinyNeuralNetwork provides a PostQuantizer class that may rewrite the graph for and perform model fusion for
        # quantization. The model returned by the `quantize` function is ready for quantization calibration.
        # By default, the rewritten model (in the format of a single file) will be generated in the working directory.
        # You may also pass some custom configuration items through the argument `config` in the following line. For
        # example, if you have a quantization-rewrite-ready model (e.g models in torchvision.models.quantization),
        # then you may use the following line.
        #   quantizer = PostQuantizer(model, dummy_input, work_dir='out', config={'rewrite_graph': False})
        # Alternatively, if you have modified the generated model description file and want the quantizer to load it
        # instead, then use the code below.
        #     quantizer = PostQuantizer(
        #         model, dummy_input, work_dir='out', config={'force_overwrite': False, 'is_input_quantized': None}
        #     )
        # The `is_input_quantized` in the previous line is a flag on the input tensors whether they are quantized or
        # not, which can be None (False for all inputs) or a list of booleans that corresponds to the inputs.
        # Also, we support multiple qschemes for quantization preparation. There are several common choices.
        #   a. Asymmetric uint8. (default) config={'asymmetric': True, 'per_tensor': True}
        #      The is the most common choice and also conforms to the legacy TFLite quantization spec.
        #   b. Asymmetric int8. config={'asymmetric': True, 'per_tensor': False}
        #      The conforms to the new TFLite quantization spec. In legacy TF versions, this is usually used in post
        #      quantization. Compared with (a), it has support for per-channel quantization in supported kernels
        #      (e.g Conv), while (a) does not.
        #   c. Symmetric int8. config={'asymmetric': False, 'per_tensor': False}
        #      The is same to (b) with no offsets, which may be used on some low-end embedded chips.
        #   d. Symmetric uint8. config={'asymmetric': False, 'per_tensor': True}
        #      The is same to (a) with no offsets. But it is rarely used, which just serves as a placeholder here.
        # In addition, we support additional ptq algorithms including kl-divergence, the usage is shown as below:
        #       quantizer = PostQuantizer(model, dummy_input, work_dir='out', config={'algorithm': 'kl'})

        quantizer = PostQuantizer(model, dummy_input, work_dir=out_dir)
        ptq_model = quantizer.quantize()

    print(ptq_model)

    # Move model to the appropriate device
    device = get_device()
    ptq_model.to(device=device)

    context = DLContext()
    context.device = device
    context.train_loader, context.val_loader = get_dataloader(data_path="./datasets/cifar10", img_size=224, batch_size=8, worker=1, download=False)
    context.max_iteration = 100

    # Post quantization calibration
    calibrate(ptq_model, context)

    with torch.no_grad():
        ptq_model.eval()
        ptq_model.cpu()

        # The step below converts the model to an actual quantized model, which uses the quantized kernels.
        ptq_model = quantizer.convert(ptq_model)

        # When converting quantized models, please ensure the quantization backend is set.
        torch.backends.quantized.engine = quantizer.backend

        # The code section below is used to convert the model to the TFLite format
        # If you need a quantized model with a specific data type (e.g. int8)
        # you may specify `quantize_target_type='int8'` in the following line.
        # If you need a quantized model with strict symmetric quantization check (with pre-defined zero points),
        # you may specify `strict_symmetric_check=True` in the following line.
        converter = TFLiteConverter(ptq_model, dummy_input, tflite_path=out_dir + model_name + '_q.tflite', quantize_target_type='int8')
        converter.convert()
            
def main_worker(args):
    model = tm.resnet18(weights=None)
    # model = tmq.resnet18(weights=None)
    # model = tmq.resnet18(weights=ResNet18_QuantizedWeights, quantize=True)
    # model = VisionTransformer(image_size=32, patch_size=2, num_layers = 20, num_heads = 1, hidden_dim = 5, mlp_dim = 10)
    # model.load_state_dict(torch.load(DEFAULT_STATE_DICT))
    dummy_input = torch.rand((1, 3, 32, 32))

    out_dir = 'gen/tinynn/'
    print(args.quant)
    if args.quant is False:
        save_model(model, dummy_input, out_dir)
    else:
        post_quant(model, dummy_input, out_dir)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quant', action="store_true", default=False)
    args = parser.parse_args()
    
    main_worker(args)

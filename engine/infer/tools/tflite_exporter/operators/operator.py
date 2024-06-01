
import sys
import numpy as np
import tflite

class Operator:
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
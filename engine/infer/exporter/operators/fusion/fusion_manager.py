
import sys
import numpy as np
import tflite
import exporter.common as tfcom

class FusionManager:
    def __init__(self):
        self.attr = {}
    
    def detect_fusible_ops(self, model, op_exporters):
        for xpt in op_exporters:
            op_code = model.OperatorCodes(xpt.get_op().OpcodeIndex())
            if tflite.BuiltinOperator.RESHAPE == op_code.BuiltinCode():
                print('reshape')
            # print(xpt.op_id())
    
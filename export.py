import argparse

import openvino as ov
import torch

from models.experimental import attempt_load
from models.yolo import Detect, DualDDetect
from utils.general import yaml_save

parser = argparse.ArgumentParser(description='YOLO export')
parser.add_argument('-m', '--model', type=str, help='model name for .pt', default='yolov9c')
parser.add_argument('-b', '--batch', type=int, help='batch number', default=1)
parser.add_argument('-q', '--quantization', type=str, help='when export, fp32 fp16 int8', default='fp16')
args = parser.parse_args()

model_name = args.model
ov_model_path = f"{model_name}_openvino_model/{model_name}.xml"

model = attempt_load(model_name + '.pt', device="cpu", inplace=True, fuse=True)
metadata = {"stride": int(max(model.stride)), "names": model.names}

model.eval()
for k, m in model.named_modules():
    if isinstance(m, (Detect, DualDDetect)):
        m.inplace = False
        m.dynamic = True
        m.export = True

example_input = torch.zeros((args.batch, 3, 640, 640))
model(example_input)

ov_model = ov.convert_model(model, example_input=example_input)

# specify input and output names for compatibility with yolov9 repo interface
ov_model.outputs[0].get_tensor().set_names({"output0"})
ov_model.inputs[0].get_tensor().set_names({"images"})
ov.save_model(ov_model, ov_model_path)
# save metadata
yaml_save(f"{model_name}_openvino_model/metadata.yaml", metadata)

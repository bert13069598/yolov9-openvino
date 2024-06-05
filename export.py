from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(description='YOLO export')
parser.add_argument('-m', '--model', type=str, help='model name for .pt', default='yolov9c')
parser.add_argument('-b', '--batch', type=int, help='batch number', default=1)
parser.add_argument('-q', '--quantization', type=str, help='when export, fp32 fp16 int8', default='fp32')
args = parser.parse_args()

model = YOLO(f"{args.model}.pt")
model.export(format="openvino",
             imgsz=640,
             dynamic=True,
             half=args.quantization == 'fp16',
             batch=args.batch)

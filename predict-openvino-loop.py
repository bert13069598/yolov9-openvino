import argparse
import re
from glob import glob
from pathlib import Path
from time import time

import cv2
import numpy as np
import openvino as ov
import pyopencl as cl
import torch
import torchvision

from utils.color import colormap
from utils.labels import yolo_labels

if cl and hasattr(cl, 'get_platforms'):
    platforms = cl.get_platforms()
    for platform in platforms:
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        print('Device:', devices[0].name)
    DEVICE = 'GPU'
else:
    DEVICE = 'CPU'

core = ov.Core()

parser = argparse.ArgumentParser(description='YOLO export')
parser.add_argument('-m', '--model', type=str, help='model name for .pt', default='yolov9c')
parser.add_argument('-b', '--batch', type=int, help='batch number', default=1)
args = parser.parse_args()


def get_warpAffineM(W, H, dst_width=640, dst_height=640):
    scale = min((dst_width / W, dst_height / H))
    ox = (dst_width - scale * W) / 2
    oy = (dst_height - scale * H) / 2
    M = np.array([
        [scale, 0, ox],
        [0, scale, oy]
    ], dtype=np.float32)
    IM = cv2.invertAffineTransform(M)
    return M, IM


def preprocess_warpAffine(image, M, dst_width=640, dst_height=640):
    img_pre = cv2.warpAffine(image, M,
                             (dst_width, dst_height),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(114, 114, 114))
    return img_pre


def xywh2xyxy(x):
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y


class YOLOv9_OPENVINO:
    def __init__(self, ov_model_name):
        batch = int(re.search(r'-b(\d+)', ov_model_name).group(1))
        src_shape = (3840, 2160)
        self.dst_shape = (640, 640)
        self.M, self.IM = get_warpAffineM(*src_shape, *self.dst_shape)

        OV_MODEL_PATH = Path(f"{ov_model_name}_openvino_model/{ov_model_name}.xml")
        ov_model = core.read_model(OV_MODEL_PATH)
        ov_config = {}
        if DEVICE != "CPU":
            ov_model.reshape({0: [batch, 3, 1024, 1024]})
        if DEVICE == "GPU":
            ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}

        t1 = time()
        self.compiled_ov_model = core.compile_model(ov_model, DEVICE, ov_config)
        t2 = time()
        print('compiled ', t2 - t1)

    def preprocess(self, *imgs):
        img_pre_batch = np.stack([preprocess_warpAffine(img, self.M, *self.dst_shape) for img in imgs])
        img_pre_batch = (img_pre_batch[..., ::-1]).transpose(0, 3, 1, 2)
        img_pre_batch = np.ascontiguousarray(img_pre_batch)
        img_pre_batch = torch.from_numpy(img_pre_batch).to('cpu').float()
        img_pre_batch /= 255
        return img_pre_batch

    def infer(self, img_pre_batch):
        return torch.from_numpy(self.compiled_ov_model(img_pre_batch)[0])

    def postprocess(self, *imgs, results, conf_thres=0.25, iou_thres=0.45):
        bs = results.shape[0]  # batch size
        nc = results.shape[1] - 4  # num of cls
        xc = results[:, 4:].amax(1) > conf_thres

        results = results.transpose(-1, -2)
        results = torch.cat((xywh2xyxy(results[..., :4]), results[..., 4:]), dim=-1)

        outputs = [torch.zeros((0, 6), device=results.device)] * bs
        for xi, x in enumerate(results):
            x = x[xc[xi]]
            if not x.shape[0]:
                continue

            box, cls = x.split((4, nc), dim=1)

            conf, label = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, label.float()), dim=1)[conf.view(-1) > conf_thres]

            if not x.shape[0]:
                continue

            x[:, 0:4:2] = self.IM[0][0] * x[:, 0:4:2] + self.IM[0][2]
            x[:, 1:4:2] = self.IM[1][1] * x[:, 1:4:2] + self.IM[1][2]

            # Batched NMS
            scores = x[:, 4]  # scores
            boxes = x[:, :4]  # boxes (offset by class)
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            outputs[xi] = x[i]

        for img, output in zip(imgs, outputs):
            boxes, confs, classes = output.split((4, 1, 1), dim=1)
            confs = confs.squeeze(1).cpu()
            classes = classes.squeeze(1).cpu()

            for i, box in enumerate(boxes):
                confidence = confs[i]
                label = int(classes[i])
                x1, y1, x2, y2 = map(int, box)
                r, g, b = map(int, colormap[label])
                caption = f"{yolo_labels[label]} {confidence:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (r, g, b), 2)
                cv2.putText(img, caption, (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (r, g, b), 2)

        return imgs


if __name__ == "__main__":
    path = ""
    paths1 = sorted(glob(path + "/202007171220_60m_45도_2_image/*.jpg"))
    paths2 = sorted(glob(path + "/202007171454_60m_45도_2_image/*.jpg"))
    paths3 = sorted(glob(path + "/202007171527_80m_45도_2_image/*.jpg"))
    paths4 = sorted(glob(path + "/202007201035_60m_45도_2_image/*.jpg"))
    pathl = [paths1, paths2, paths3, paths4]

    yolov9_ov = YOLOv9_OPENVINO("{}-b{}".format(args.model, args.batch))
    while True:
        for paths in zip(*pathl[:args.batch]):
            imgs = [cv2.imread(path) for path in paths]

            t1 = time()
            pre_batch = yolov9_ov.preprocess(*imgs)
            t2 = time()
            results = yolov9_ov.infer(pre_batch)
            t3 = time()
            post_batch = yolov9_ov.postprocess(*imgs, results=results)
            t4 = time()

            print('{:.5f} + {:.5f} + {:.5f} = {:.5f}'.format(t2 - t1, t3 - t2, t4 - t3, t4 - t1))

            for i in range(len(paths)):
                cv2.imshow(f"Drone {i + 1}", cv2.resize(post_batch[i], (3840 // 4, 2160 // 4)))
            cv2.waitKey(1)

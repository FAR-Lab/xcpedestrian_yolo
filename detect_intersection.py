import random
import time
import copy
from pathlib import Path

import cv2
import fire
import torch

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import (set_logging, check_img_size, non_max_suppression,
                           scale_coords, xyxy2xywh, increment_path)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
from ultralytics import YOLO


class Pipeline(object):
    def __init__(self, source,
                 weights="yolov7-e6e.pt",
                 view_img=True,
                 save_txt=False,
                 img_size=640,
                 no_save=True,
                 no_trace=True,
                 device="",
                 iou_thres=0.45,
                 conf_thres=0.25,
                 save_conf=False,
                 augment=False,
                 agnostic_nms=False,
                 project="runs/detect",
                 exist_ok=True,
                 name="exp"):
        self.source = source
        self.save_img = not no_save and not source.endswith('.txt')
        self.webcam = source.isnumeric() or \
            source.endswith('.txt') or \
            source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        self.weights = weights
        self.view_img = view_img
        self.save_txt = save_txt
        self.save_conf = save_conf
        self.img_size = img_size
        self.augment = augment
        self.agnostic_nms = agnostic_nms
        self.no_trace = not no_trace
        self.device = device
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres

        # Directories
        self.save_dir = Path(increment_path(Path(project) /
                                            name, exist_ok=exist_ok))  # increment run
        (self.save_dir / 'labels' if save_txt else self.save_dir).mkdir(parents=True,
                                                                        exist_ok=True)  # make dir

        # Initialize
        set_logging()
        self.device = select_device(self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.names = []
        self.colors = []

        # Load default model
        self.load_default_model()
        self.stride = int(self.default_model.stride.max())  # model stride
        self.img_size = check_img_size(
            self.img_size, s=self.stride)  # check img_size
        # Get names and colors
        self.names.append(self.default_model.module.names if hasattr(
            self.default_model, 'module') else self.default_model.names)
        self.colors.append([[random.randint(0, 255)
                            for _ in range(3)] for _ in self.default_model.names])

        # load crosswalk model
        self.crosswalk_model = YOLO("crosswalk.pt")
        self.names.append(["crosswalk"])
        self.colors.append([[random.randint(0, 255) for _ in range(3)]])

        # load dataset
        self.dataset = self.load_dataset()

        # Run inference
        if self.device.type != 'cpu':
            self.run_inference()

        # predict
        for c, (path, img, im0s, vid_cap) in enumerate(self.dataset):
            preds = []
            preds.append(self.predict_elements(c, img, im0s))
            preds.append(self.predict_crosswalk(c, img, im0s))
            self.save_result(preds, path, im0s, vid_cap)

    def load_dataset(self):
        dataset = LoadImages(self.source, img_size=self.img_size)
        return dataset

    def load_default_model(self):
        self.default_model = attempt_load(
            self.weights, map_location=self.device)  # load FP32 model

    def run_inference(self):
        self.default_model(torch.zeros(1, 3, self.img_size, self.img_size).to(
            self.device).type_as(next(self.default_model.parameters())))  # run once

    def remove_shadow(self, img):
        return img

    def img_preprocess(self, img):
        img = torch.from_numpy(img).to(self.device)
        img = self.remove_shadow(img)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def predict_elements(self, cc, img, im0s):
        img = copy.deepcopy(img)
        img = self.img_preprocess(img)

        # Inference
        t1 = time_synchronized()
        pred = self.default_model(img, augment=self.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, classes=[0, 1, 2, 3, 5, 9, 11], agnostic=self.agnostic_nms)
        t3 = time_synchronized()

        s = ""
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                im0 = im0s[i] if self.webcam else im0s
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {self.names[0][int(c)]}{'s' * (n > 1)}, "
            else:
                s = "No elements detected,"

        # Print time (inference + NMS)
        print(
            f'[{cc}/{len(self.dataset)}] {s}Predict elements done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        return pred

    def predict_crosswalk(self, cc, img, im0s):
        img = copy.deepcopy(img)

        # Inference
        t1 = time_synchronized()
        results = self.crosswalk_model(img.transpose(1, 2, 0))[0]
        t2 = time_synchronized()

        # Convert to numpy
        pred = []
        for box in results.boxes:
            if box.conf < self.conf_thres:
                continue
            pred.append([box.xyxy[0].tolist() + [float(box.conf), 0]])
        pred = torch.tensor(pred)

        # Apply NMS
        # pred = non_max_suppression(
        #   pred, self.conf_thres, self.iou_thres, classes=None, agnostic=self.agnostic_nms)
        t3 = time_synchronized()

        s = ""
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                im0 = im0s[i] if self.webcam else im0s
                det[:, :4] = scale_coords(
                    img.shape[1:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {self.names[1][int(c)]}{'s' * (n > 1)}, "
            else:
                "No crosswalk detected."

        # Print time (inference + NMS)
        print(
            f'[{cc}/{len(self.dataset)}]  {s}Predict elements done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        return pred

    def save_result(self, preds, path, im0s, vid_cap):
        for pp, pred in enumerate(preds):
            for i, det in enumerate(pred):
                if self.webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), self.dataset.count
                else:
                    p, im0, frame = path, im0s, getattr(
                        self.dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(self.save_dir / p.name)  # img.jpg
                txt_path = str(self.save_dir / 'labels' / p.stem) + \
                    ('' if self.dataset.mode ==
                        'image' else f'_{frame}')  # img.txt

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                # Write results
                # normalization gain whwh
                for *xyxy, conf, cls in reversed(det):
                    if self.save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(
                            1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # label format
                        line = (
                            cls, *xywh, conf) if self.save_conf else (cls, *xywh)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() %
                                    line + '\n')

                    if self.save_img or self.view_img:  # Add bbox to image
                        label = f'{self.names[pp][int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label,
                                     color=self.colors[pp][int(cls)], line_thickness=1)

                # Stream results
                if self.view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if self.save_img:
                    if self.dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                        print(
                            f" The image with the result is saved in: {save_path}")
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(
                                save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

        if self.save_txt or self.save_img:
            s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if self.save_txt else ''
            # print(f"Results saved to {save_dir}{s}")


if __name__ == '__main__':
    fire.Fire(Pipeline)

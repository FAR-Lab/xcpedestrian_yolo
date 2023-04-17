import random
import time
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

        # Load model
        self.load_model()
        self.stride = int(self.model.stride.max())  # model stride
        self.img_size = check_img_size(
            self.img_size, s=self.stride)  # check img_size

        # Get names and colors
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255)
                        for _ in range(3)] for _ in self.names]

        # load dataset
        self.dataset = self.load_dataset()
        self.img = list(self.dataset)[0]

        # Run inference
        if self.device.type != 'cpu':
            self.run_inference()

        # predict
        self.predict_elements()

    def load_dataset(self):
        dataset = LoadImages(self.source, img_size=self.img_size)
        return dataset

    def load_model(self):
        self.model = attempt_load(
            self.weights, map_location=self.device)  # load FP32 model

    def run_inference(self):
        self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(
            self.device).type_as(next(self.model.parameters())))  # run once

        # Warmup
        old_img_w = old_img_h = self.img_size
        old_img_b = 1
        if old_img_b != self.img.shape[0] or old_img_h != self.img.shape[2] or old_img_w != self.img.shape[3]:
            for _ in range(3):
                self.model(self.img, augment=self.augment)[0]

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

    def predict_elements(self):
        t0 = time.time()
        for path, img, im0s, vid_cap in self.dataset:
            img = self.img_preprocess(img)

            # Inference
            t1 = time_synchronized()
            pred = self.model(img, augment=self.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(
                pred, self.conf_thres, self.iou_thres, classes=[0, 1, 2, 3, 5, 9, 11], agnostic=self.agnostic_nms)
            t3 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if self.webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(
                    ), self.dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(
                        self.dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(self.save_dir / p.name)  # img.jpg
                txt_path = str(self.save_dir / 'labels' / p.stem) + \
                    ('' if self.dataset.mode ==
                     'image' else f'_{frame}')  # img.txt
                # normalization gain whwh
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        # add to string
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "

                    # Write results
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
                            label = f'{self.names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label,
                                         color=self.colors[int(cls)], line_thickness=1)

                # Print time (inference + NMS)
                print(
                    f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

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

        print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    fire.Fire(Pipeline)

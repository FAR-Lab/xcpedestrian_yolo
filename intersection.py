import random
import copy
import math
from pathlib import Path

import cv2
import fire
import torch
import tqdm
import pandas as pd
# from ultralytics import YOLO


from models.experimental import attempt_load
from utils.datasets import LoadImages, letterbox
from utils.general import (set_logging, check_img_size, non_max_suppression,
                           scale_coords, xyxy2xywh, increment_path, xywh2xyxy)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized


class ExtractVideo(object):
    def __init__(self, source, format=".png", video_stride='fps', size=None, img_stride=32):
        # set up output path
        self.source = source
        p = Path(source).absolute()  # os-agnostic absolute path
        if format.endswith('.mp4') or format.endswith('.avi'):
            self.mode = "video"
            self.output_path = p.parent / f"{p.stem}{format}"  # video
            print("Video path:", self.output_path)
        elif format.endswith(".png") or format.endswith(".jpg"):
            self.mode = "image"
            output_dir = p.parent / "output" / p.stem / "images"
            output_dir.mkdir(parents=True, exist_ok=True)  # make dir
            self.output_path = str(output_dir / f"%05d{format}")  # image
            print("Image path:", output_dir)
        else:
            raise Exception(f"ERROR: format {format} not supported")

        # set up video capture
        self.frame = 0
        self.cap = cv2.VideoCapture(source)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # set up output params
        self.video_stride = self.fps if video_stride == 'fps' else video_stride
        self.image_stride = img_stride
        self.size = size

    def __iter__(self):
        return self

    def __next__(self):
        while self.frame < self.nframes:
            ret_val, img0 = self.cap.read()
            self.frame += 1
            if ret_val and (self.frame - 1) % self.video_stride == 0:
                if self.size:
                    img0 = letterbox(
                        img0, self.size, stride=self.image_stride)[0]
                return img0
        else:
            self.cap.release()
            raise StopIteration

    def extract(self):
        if self.mode == "video":
            encoder = cv2.VideoWriter_fourcc(
                *"mjpg" if self.output_path.endswith(".avi") else "mp4v")
            writer = cv2.VideoWriter(self.output_path,
                                     encoder,
                                     self.fps // self.video_stride,
                                     frameSize=(self.width, self.height))
        for img0 in tqdm.tqdm(self, total=self.nframes // self.video_stride):
            if self.mode == "image":
                cv2.imwrite(self.output_path % self.frame, img0)
            else:
                writer.write(img0)


class MergeImage(object):
    def __init__(self, source, format=".mp4", fps=1):
        # set up output path
        self.source = source
        p = Path(source).absolute()  # os-agnostic absolute path

        self.dataset = list(p.glob('*.png')) + \
            list(p.glob('*.jpg')) + list(p.glob('*.jpeg'))
        self.nframes = len(self.dataset)
        self.height, self.width = cv2.imread(
            str(self.dataset[0])).shape[:2]

        self.output_path = p.parent / f"{p.stem}{format}"  # video
        self.fps = fps

    def __iter__(self):
        return self.__next__()

    def __next__(self):
        for path in sorted(self.dataset, key=lambda x: int(x.stem)):
            img = cv2.imread(str(path))
            yield img

    def __len__(self):
        return len(self.dataset)

    def merge(self):
        encoder = cv2.VideoWriter_fourcc(
            *"mjpg" if format == ".avi" else "mp4v")

        writer = cv2.VideoWriter(str(self.output_path),
                                 encoder,
                                 fps=self.fps,
                                 frameSize=(self.width, self.height))
        for img0 in tqdm.tqdm(self, total=self.nframes):
            writer.write(img0)
        writer.release()
        print("Video saved to", self.output_path)


class Preprocessing(object):
    def __init__(self, source, output_path=None):
        self.source = source
        self.image = cv2.imread(source)
        if output_path is None:
            p = Path(self.source)
            self.output_path = p.parent / 'output' / 'images' / p.name
        else:
            self.output_path = output_path

    def crop(self, width, height, x=0, y=0):
        self.image = self.image[y:y+height, x:x+width]

    def remove_shadow(self):
        pass

    def remove_reflection(self):
        pass


class YoloModel(object):
    def __init__(self, weights="yolov7-e6e.pt", img_size=640, device="cpu") -> None:
        self.model = attempt_load(weights, map_location=device)
        self.stride = int(self.model.stride.max())
        self.img_size = img_size
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255)
                        for _ in range(3)] for _ in self.names]
        if device != "cpu":
            self.run_inference(device)

        self.count = -1

    def run_inference(self, device):
        with torch.no_grad():
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(
                device).type_as(next(self.model.parameters())))  # run once

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class CrosswalkModel(object):
    def __init__(self, weights) -> None:
        self.model = YOLO(weights)
        self.names = ['crosswalk']
        self.colors = [[random.randint(0, 255) for _ in range(3)]]

        self.count = -1

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def is_webcam(source):
    return source.isnumeric() or \
        source.endswith('.txt') or \
        source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))


class Detection(object):
    def __init__(self,
                 source,
                 detect_yolo=True,
                 detect_crosswalk=False,
                 view_img=True,
                 save_txt=False,
                 no_save=True,
                 device="",
                 img_size=640,
                 stride=32,
                 iou_thres=0.45,
                 conf_thres=0.25,
                 augment=False,
                 agnostic_nms=False,
                 project="runs/detect",
                 name="exp") -> None:
        self.source = source
        self.dataset = LoadImages(source)
        self.augment = augment
        self.agnostic_nms = agnostic_nms
        self.device = device
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.project = project
        self.name = name
        self.img_size = check_img_size(img_size, s=stride)
        self.stride = stride

        # Directories
        self.save_dir = Path(Path(source).parent / increment_path(Path(project) /
                                                                  name, exist_ok=True))  # increment run
        (self.save_dir / 'labels' if save_txt else self.save_dir).mkdir(parents=True,
                                                                        exist_ok=True)  # make dir

        set_logging()
        self.device = select_device(self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        self.yolo_model = None
        self.crosswalk_model = None

        save_img = not no_save and not source.endswith('.txt')
        self.colors = []
        self.names = []

        self.imgs = iter(self.dataset)
        self.c = -1
        with tqdm.tqdm(total=len(self.dataset)) as pbar:
            while True:
                self.load_next_image()
                # TODO: tmp solution for skipping detected images
                txt_path = self.save_dir / 'labels' / \
                    (Path(self.path).stem + '.txt')
                if not txt_path.exists():
                    if detect_yolo:
                        self.detect_yolo_process()
                    if detect_crosswalk:
                        self.detect_crosswalk_process()
                    self.save_results(
                        view_img=view_img,
                        save_txt=save_txt,
                        save_img=save_img
                    )
                pbar.update(1)

    def load_next_image(self):
        self.path, self.img, self.im0s, self.vid_cap = next(self.imgs)
        self.preds = []
        self.c += 1

    def detect_yolo_process(self, weights="yolov7-e6e.pt"):
        # initialize
        if self.yolo_model is None:
            self.yolo_model = YoloModel(
                weights,
                img_size=self.img_size,
                device=self.device
            )
            self.colors.append(self.yolo_model.colors)
            self.names.append(self.yolo_model.names)

        # img preprocessing
        img = self.img
        # img = copy.deepcopy(self.img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():
            pred = self.yolo_model(img, augment=self.augment)[0]
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
                im0 = self.im0s[i] if is_webcam(self.source) else self.im0s
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
            f'[{self.c}/{len(self.dataset)}] {s}Predict elements done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        self.preds.append(pred)

    def detect_crosswalk_process(self, weights="crosswalk.pt"):
        if self.crosswalk_model is None:
            self.crosswalk_model = CrosswalkModel(weights)
            self.colors.append(self.crosswalk_model.colors)
            self.names.append(self.crosswalk_model.names)

        img = copy.deepcopy(self.img)

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
                im0 = self.m0s[i] if is_webcam(self.source) else self.im0s
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
            f'[{self.c}/{len(self.dataset)}]  {s}Predict elements done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        self.preds.append(pred)

    def save_results(self, view_img, save_txt, save_img):
        for pp, pred in enumerate(self.preds):
            for i, det in enumerate(pred):
                if is_webcam(self.source):  # batch_size >= 1
                    p, im0, frame = self.path[i], self.im0s[i].copy(
                    ), self.dataset.count
                else:
                    p, im0, frame = self.path, self.im0s, getattr(
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
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(
                            1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # label format
                        line = *xywh, conf
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(("_".join(self.names[pp][int(cls)].split(" ")) +
                                     ' %g' * len(line)).rstrip() % (line) +
                                    '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{self.names[pp][int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label,
                                     color=self.colors[pp][int(cls)], line_thickness=1)

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond
                    cv2.destroyAllWindows()

                # Save results (image with detections)
                if save_img:
                    if self.dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                        print(
                            f" The image with the result is saved in: {save_path}")
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if self.vid_cap:  # video
                                fps = self.vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(self.vid_cap.get(
                                    cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(self.vid_cap.get(
                                    cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(
                                save_path,
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                fps,
                                (w, h)
                            )
                        vid_writer.write(im0)

        if save_txt or save_img:
            s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {self.save_dir}{s}")


def sigmoid(x):
    "Numerically-stable sigmoid function."
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)


class Intersection:
    def __init__(self, yolo_label_dir, crosswalk_label_dir) -> None:
        self.yolo_label_dir = Path(yolo_label_dir)
        self.cross_label_dir = Path(crosswalk_label_dir)

        # read labels
        self.yolo_df = self.get_labels(self.yolo_label_dir)
        self.crosswalk_df = self.get_labels(self.cross_label_dir)

        self.df = pd.concat(
            [self.yolo_df.copy(), self.crosswalk_df.copy()], axis=0)

    def read_label(self, path):
        labels = []
        if path.exists():
            with open(path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    words = line.strip("\n").split(" ")
                    if len(words) == 7:
                        labels.append([words[0] + words[1]] + words[2:])
                    else:
                        labels.append(words)
        return labels

    def get_labels(self,  dir):
        dic = []
        for p in dir.glob("*"):
            label = self.read_label(dir / f"{p.stem}.txt")
            for cls, x, y, w, h, conf in label:
                dic.append({
                    "filename": p.name,
                    "label": cls if cls != "Zebra_Cross" else "Crosswalk",
                    "conf": conf
                })
        df_label = pd.DataFrame(dic)
        df_label["conf"] = df_label["conf"].astype(float)
        return df_label

    def predict(self, thres=0.8, seconds=3, buffer=0, fps=1):
        df = self.df.groupby(["filename", "label"]).agg(
            {"conf": "sum"}).unstack().fillna(0).astype(float)
        df = df.droplevel(0, axis=1).reset_index()

        w = {
            "person": 0.008687173316285312,
            "Crosswalk": 3.3470720555129185,
            "car": 0.9916819493912497,
            "bicycle": 0.4772582269590735,
            "bus": 1.9100436819742963,
            "motorcycle": 0.2392703359735101,
            "traffic_light": 1.8459140771948266,
            "R_Signal": 0.12295418615641256,
            "stop_sign": -0.45393678624340095
        }
        b = -4.909474292254606

        def logistic_regression(x):
            z = b
            for label, w_i in w.items():
                z += w_i * x[label] if label in x else 0.0
            return sigmoid(z)

        df["prob"] = df.apply(logistic_regression, axis=1)

        # add buffer and smoothing to prediction
        # using moving average
        if buffer > 0 or seconds > 1:
            frames_to_buffer = fps * buffer
            frames_range = fps * seconds
            frames_thres = frames_range * thres
            probs = df["prob"].values
            intervals = []
            moving_sum = 0
            for i in range(len(df)):
                # actively seaking first pred > thres and count > frames_threshold
                moving_sum += probs[i]

                if i - frames_range >= 0:
                    moving_sum -= probs[i - frames_range]

                if moving_sum >= frames_thres:
                    end = min(i + frames_to_buffer, len(df) - 1)
                    if len(intervals) == 0 or intervals[-1][-1] < i - frames_to_buffer:
                        intervals.append(
                            [max(i - frames_range - frames_to_buffer, 0), end])
                    else:
                        intervals[-1][-1] = end

            preds = [0] * len(df)
            for start, end in intervals:
                preds[start:end + 1] = [1] * (end - start + 1)
            df["pred"] = preds
        else:
            df["pred"] = df["prob"] >= thres

        self.df = df
        return self

    def save(self, image_dir, save_img=False, view_img=False):
        image_dir = Path(image_dir)
        output_dir = self.cross_label_dir.parent.parent / "intersection"
        output_label_dir = output_dir / "labels"
        output_label_dir.mkdir(parents=True, exist_ok=True)

        for filename, prob, pred in zip(self.df["filename"], self.df["prob"], self.df["pred"]):
            p = Path(filename)
            with open(output_label_dir / f"{p.stem}.txt", "w") as f:
                f.write(f"{prob}, {int(pred)}")

        if save_img or view_img:
            for filename, prob, pred in tqdm.tqdm(zip(self.df["filename"], self.df["prob"], self.df["pred"]),
                                                  total=len(self.df)):
                p = Path(filename)

                # read image
                im = cv2.imread(str(image_dir / (p.stem + ".png")))

                # read label
                yolo_label = self.read_label(
                    self.yolo_label_dir / f"{p.stem}.txt")
                crosswalk_label = self.read_label(
                    self.cross_label_dir / f"{p.stem}.txt")

                # draw label
                colors = {name: [random.randint(0, 255) for _ in range(
                    3)] for name in self.crosswalk_df["label"].unique().tolist() + self.yolo_df["label"].unique().tolist()}
                gn = torch.tensor(im.shape)[[1, 0, 1, 0]]
                line_thickness = 1 if im.shape[0] < 1000 else 2
                for cls, x, y, w, h, conf in yolo_label + crosswalk_label:
                    cls = cls if cls != "Zebra_Cross" else "Crosswalk"
                    xywh = torch.Tensor(
                        list(map(float, [x, y, w, h]))).view(1, 4)
                    xyxy = xywh2xyxy(xywh) * gn
                    label = f'{cls} {float(conf):.2f}'
                    plot_one_box(xyxy.reshape(-1), im, label=label,
                                 color=colors[cls], line_thickness=line_thickness)
                    if pred == 1:
                        cv2.putText(im, f"{'Intersection'}({prob:.2f})", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), line_thickness + 1)
                    else:
                        cv2.putText(im, f"{'Other'}({prob:.2f})", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), line_thickness + 1)

                if view_img:
                    cv2.imshow(str(p), im)
                    cv2.waitKey(0) & 0xFF == ord('q')
                    cv2.destroyAllWindows()

                if save_img:
                    cv2.imwrite(str(output_dir / (p.stem + '.png')), im)


if __name__ == "__main__":
    fire.Fire({
        "extract": ExtractVideo,
        "preprocess": Preprocessing,
        "detect": Detection,
        "intersection": Intersection,
        "merge": MergeImage
    })

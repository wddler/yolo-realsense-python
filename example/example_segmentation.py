"""Yolo v8 segmentation with RealSense D435i Distance Measurement"""

import sys
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import torch

sys.path.append('../')
from yolo_realsense.scripts.yolov8_realsense import annotate_distance

W = 1280 # 640
H = 720 # 480

torch.cuda.set_device(0) # Set to your desired GPU number

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')

config = rs.config() # pyright: ignore
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30) # pyright: ignore
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30) # pyright: ignore

pipeline = rs.pipeline() # pyright: ignore
profile = pipeline.start(config)
align_to = rs.stream.color # pyright: ignore
align = rs.align(align_to) # pyright: ignore

# Load a model
model = YOLO('yolov8m-seg.pt')  # load an official model
# model_directory = './models/yolov8m-seg.pt'
# model = YOLO(model_directory)

while True:
    frames = pipeline.wait_for_frames()

    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08),
                                       cv2.COLORMAP_JET)

    results = model(color_image, verbose=False)

    color_image = annotate_distance(color_image, depth_image, results)

    # add a class name manually, because it's a part of a bounding box which we don't visualize.
    for r in results:
        boxes = r.boxes
        for box in boxes:
            class_name=box.cls.item()
            cv2.putText(img=color_image,
                        text=model.names[class_name], #pylint: disable=E1136
                        org=(int(box.xyxy[0][0].item()), int(box.xyxy[0][1].item())),
                        fontFace=0,
                        fontScale=1,
                        color=(255, 255, 255),
                        thickness=2
                        )

    annotated_frame = results[0].plot(boxes=False)

    cv2.imshow("color_image", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

"""Yolo v8 with RealSense D435i Distance Measurement"""

import cv2
import numpy as np
from ultralytics.engine.results import Boxes, Results

def get_distance(bounding_box: Boxes, depth_image: np.ndarray) -> float:
    """Returns distance in millimeters to the center of the bounding_box"""
    x = int(bounding_box.xyxy[0][0].item())
    y = int(bounding_box.xyxy[0][1].item())
    x2 = int(bounding_box.xyxy[0][2].item())
    y2 = int(bounding_box.xyxy[0][3].item())
    cx = (x + x2) // 2
    cy = (y + y2) // 2
    distance_mm = depth_image[cy, cx]
    return distance_mm

def annotate_distance(image: np.ndarray, depth_image: np.ndarray, res: Results) -> np.ndarray:
    """returns color_image annotated with distances"""
    for r in res:
        boxes = r.boxes
        for box in boxes:
            distance = get_distance(box, depth_image)
            cv2.putText(img=image,
                        text=f"{distance / 10} cm",
                        org=(int(box.xyxy[0][0].item())+5, int(box.xyxy[0][1].item())+30), # x, y
                        fontFace=0,
                        fontScale=0.99,
                        color=(255, 255, 255),
                        thickness=2
                        )
    return image

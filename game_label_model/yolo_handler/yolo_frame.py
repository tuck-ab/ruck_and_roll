import numpy as np

from .bounding_box_store import BoundingBoxStore
from .YOLOv7 import YOLOv7

class YOLORunner:
    def __init__(self, model_path: str):
        self._yolo_detector = YOLOv7(model_path, conf_thres=0.5, iou_thres=0.5)
    
    def run(self, frame: np.ndarray) -> BoundingBoxStore:
        boxes, scores, class_ids = self._yolo_detector(frame)
        
        return BoundingBoxStore(boxes, scores, class_ids)
        
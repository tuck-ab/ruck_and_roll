import numpy as np

from .bounding_box_store import BoundingBoxStore
from .YOLOv7 import YOLOv7

class YOLORunner:
    def __init__(self, model_path: str):
        self._yolo_detector = YOLOv7(model_path, conf_thres=0.5, iou_thres=0.5)
    
    def run(self, frame: np.ndarray) -> BoundingBoxStore:
        """Runs YOLO on a given frame

        Args:
            frame (np.ndarray): The frame to run YOLO on

        Returns:
            BoundingBoxStore: The result of running YOLO
        """
        results = self._yolo_detector(frame)
        
        return BoundingBoxStore(*results)
    
    def get_detections(self, frame: np.ndarray) -> np.ndarray:
        """Gets the annotated frame from running YOLO on a given frame

        Args:
            frame (np.ndarray): The frame to run YOLO on

        Returns:
            np.ndarray: The annotated frame
        """
        boxes, scores, class_ids = self._yolo_detector(frame)
        
        return self._yolo_detector.draw_detections(frame)

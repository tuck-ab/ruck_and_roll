from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from ..video_handler import VideoHandler
from ..yolo_handler import YOLORunner

class BoundingBoxCNN:
    def __init__(self, vid_path: str, model_path: str):
        self._vid_handler = VideoHandler().load_video(vid_path)
        self._yolo_handler = YOLORunner(model_path)

    def run_frame(self):
        frame, valid = self._vid_handler.get_next_frame()

        if not valid:
            print("NOT VALID FRAME IN BoundingBoxCNN.run_frame")
            return
            
        result = self._yolo_handler.run(frame)

        for bb in result.get_store():
            x_start, x_end = bb.x, bb.x+bb.w
            y_start, y_end = bb.y, bb.y+bb.h

            bb_im = frame[y_start: y_end, x_start: x_end, :]
            
    def get_num_bbs_per_frame(self, save_to_file: Optional[str] = None):
        bb_per_frame = []
        
        frame, is_new = self._vid_handler.get_next_frame()
        
        while is_new:
            result = self._yolo_handler.run(frame)
            
            bb_per_frame.append(len(result.get_store()))
            
            frame, is_new = self._vid_handler.get_next_frame()
            
        bb_per_frame = np.array(bb_per_frame)
        
        if save_to_file:
            np.savetxt(save_to_file, bb_per_frame)

        return bb_per_frame

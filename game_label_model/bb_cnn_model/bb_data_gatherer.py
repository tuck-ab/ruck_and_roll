import json
from typing import Optional

import numpy as np

from ..video_handler import VideoHandler
from ..yolo_handler import YOLORunner

class BoundingBoxDataGatherer:
    def __init__(self, vid_path: str, model_path: str):
        self._vid_handler = VideoHandler().load_video(vid_path)
        self._yolo_handler = YOLORunner(model_path)

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


    def get_bbs_data_per_frame(self, save_to_file: Optional[str] = None):
        data_per_frame = []

        frame, is_new = self._vid_handler.get_next_frame()
        
        while is_new:
            result = self._yolo_handler.run(frame)

            data = []
            for bb in result.get_store():
                data.append(bb.get_JSON_dict())

            data_per_frame.append({
                "bbs": data,
                "count": len(data)
            })

            frame, is_new = self._vid_handler.get_next_frame()

        if save_to_file:
            with open(save_to_file, "w") as out:
                json.dump(data_per_frame, out)

        return data_per_frame

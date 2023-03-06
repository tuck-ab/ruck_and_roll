import os
import shutil
from typing import Tuple

import numpy as np

from ..video_handler import ErrorPlayingVideoException, VideoHandler
from ..yolo_handler import YOLORunner
from .hyperparameters import VALID_YOLO_LABELS, NUM_CNNS, BB_SIZE

def generate(vid_dir: str, vid_name: str, yolo_dir: str, yolo_name: str,
             data_dir: str, bb_size: Tuple[int, int]=BB_SIZE):
    vid_path = os.path.join(vid_dir, vid_name)
    yolo_path = os.path.join(yolo_dir, yolo_name)

    vid_handler = VideoHandler().load_video(vid_path)
    yolo_handler = YOLORunner(yolo_path)

    frame, valid = vid_handler.get_next_frame()

    if not valid:
        raise ErrorPlayingVideoException(
            f"Could not play video\n\tPath: {vid_path}"
            )
    
    save_dir = os.path.join(data_dir, f"{vid_name}-{yolo_name}")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    while valid:
        result = yolo_handler.run(frame)

        bb_ims = []

        for bb in result.get_store():
            if bb.class_name in VALID_YOLO_LABELS:
                x_start, x_end = bb.x, bb.x+bb.w
                y_start, y_end = bb.y, bb.y+bb.h

                bb_im = frame[y_start: y_end, x_start: x_end, :]
                bb_ims.append(bb_im)

        ## Choose which bounding boxes to use
        ## TODO Make this a better choice
        bb_ims = bb_ims[:NUM_CNNS]

        temp = [pad_image(im, bb_size) for im in bb_ims]
        bb_ims = temp

        while len(bb_ims) < 10:
            bb_ims.append(np.zeros((*bb_size, 3)))

        bb_ims = np.array(bb_ims)        

        ## Save the bb_ims_to_size
        f_name = f"{vid_handler.current_frame_num}.npy"
        with open(os.path.join(save_dir, f_name), "wb") as f:
            np.save(f, bb_ims)

        frame, valid = vid_handler.get_next_frame()

def pad_image(im, target_size):
    new_im = np.zeros((*target_size, 3)).astype(np.uint8)
    new_im[:im.shape[0], :im.shape[1]] = im[:new_im.shape[0], :new_im.shape[1]]

    return new_im

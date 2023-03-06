import os
from typing import Tuple

import numpy as np
from tensorflow.keras.utils import Sequence

from .generate_yolo_data import generate
from .hyperparameters import BB_SIZE


class YOLOSequence(Sequence):
    def __init__(self, labels_dir: str, labels_name:str,
                 vid_dir: str, vid_name: str,
                 yolo_dir: str, yolo_name: str,
                 temp_path: str, bb_size: Tuple[int, int]=BB_SIZE,
                 batch_size: int=32, generate_data: bool=True):
        self.labels = []

        self.batch_size = batch_size

        self.data_path = os.path.join(temp_path, "yolo_batch_files")

        if generate_data:
            if not os.path.isdir(self.data_path):
                os.mkdir(self.data_path)

            generate(vid_dir, vid_name, yolo_dir, yolo_name,
                     self.data_path, bb_size=bb_size)

    def __len__(self):
        return int(np.floor(len(self.labels) / self.batch_size))
    
    def __getitem__(self, index):
        pass
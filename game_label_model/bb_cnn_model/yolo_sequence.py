import os
from typing import Tuple

import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical

from ..hyperparameters import BB_SIZE, NUM_CNNS
from ..labels import NUM_CLASSES
from ..labels import load_from_file as load_labels_from_file
from .generate_yolo_data import generate


class YOLOSequence(Sequence):
    def __init__(self, labels_dir: str, labels_name:str,
                 vid_dir: str, vid_name: str,
                 yolo_dir: str, yolo_name: str,
                 temp_path: str, bb_size: Tuple[int, int]=BB_SIZE,
                 batch_size: int=32, generate_data: bool=True):
        
        labels_path = os.path.join(labels_dir, labels_name)
        self.labels = load_labels_from_file(labels_path)

        self.batch_size = batch_size

        self.data_path = os.path.join(temp_path, "yolo_batch_files")
        self.dir_name = f"{vid_name}-{yolo_name}"

        if generate_data:
            if not os.path.isdir(self.data_path):
                os.mkdir(self.data_path)

            generate(vid_dir, vid_name, yolo_dir, yolo_name,
                     self.data_path, bb_size=bb_size)

    def __len__(self):
        return int(np.floor(len(self.labels) / self.batch_size))
    
    def __getitem__(self, index):
        frames = list(range(index*self.batch_size, (index+1)*self.batch_size))

        batch_xs = [[] for _ in range(0, NUM_CNNS)]
        batch_y = []

        for frame in frames:
            yolo_result_path = os.path.join(self.data_path, self.dir_name,
                                            f"{frame}.npy")
            
            for im, batch_x in zip(np.load(yolo_result_path), batch_xs):
                batch_x.append(im)

            batch_y.append(self.labels[frame].value - 1)

        batch_x_final = [np.array(xs) for xs in batch_xs]

        return batch_x_final, to_categorical(batch_y, num_classes=NUM_CLASSES)

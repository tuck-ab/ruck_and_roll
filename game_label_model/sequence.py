import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow.image import resize
from tensorflow.keras.utils import Sequence, to_categorical

from .hyperparameters import (BATCH_SIZE, CLIP_SIZE, NUM_CNNS,
                              THREED_CNN_INPUT_SHAPE)
from .labels import NUM_CLASSES
from .labels import load_from_file as load_labels_from_file
from .video_handler import VideoHandler


class CustomSequence(Sequence):
    def __init__(self, vid_path, labels, int_dir, batch_size, clip_size=CLIP_SIZE,
                 image_size=THREED_CNN_INPUT_SHAPE[1:3]):
        self.labels = labels

        self.batch_size = batch_size
        self.clip_size = clip_size
        self.image_size = image_size

        self.video_handler = VideoHandler().load_video(vid_path)
        self.intermidiate_dir = int_dir

    def __len__(self):
        return int(np.floor(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        indicies = list(range(index*self.batch_size, (index+1)*self.batch_size))

        ## Add the yolo bounding boxes from pregenned
        batch_bb_xs = [[] for _ in range(0, NUM_CNNS)]
        batch_y = []

        clips = []

        for i in indicies:
            frame = self.labels.iloc[i]["frame"]
            bbs_path = os.path.join(self.intermidiate_dir, f"yolo-{frame}.npy")

            for bb, batch_x in zip(np.load(bbs_path), batch_bb_xs):
                batch_x.append(bb)

            batch_y.append(self.labels.iloc[i]["label"].value - 1)

            clip = self.video_handler.get_clip(frame-(self.clip_size//2), self.clip_size)
            clip = resize(np.array(clip), self.image_size).numpy()
            clips.append(clip)

        ## Convert to np arrays
        batch_x_final = [np.array(clips)]
        for xs in batch_bb_xs:
            batch_x_final.append(np.array(xs))

        return batch_x_final, to_categorical(batch_y, num_classes=NUM_CLASSES)

    def on_epoch_end(self):
        ## Shuffle the rows
        self.labels = self.labels.sample(frac=1).reset_index(drop=True)


def get_train_test_val(video_path, int_data_dir, labels_path, clip_size=CLIP_SIZE, limit=None):
    labels = pd.DataFrame(load_labels_from_file(labels_path), columns=["label"])
    labels["frame"] = labels.index

    if limit:
        labels = labels[:limit]

    buff_size = (clip_size // 2) + 1
    labels = labels.iloc[buff_size:-buff_size]

    ## Split into train, test, validation
    skf = StratifiedKFold(n_splits=5)
    
    splits = [test_index for _, test_index in skf.split(labels, labels["label"].apply(lambda x: x.value))]

    train_indecies = np.concatenate([splits[i] for i in [0, 1, 2]])

    train_sequence = CustomSequence(video_path, labels.iloc[train_indecies], int_data_dir, BATCH_SIZE)
    validation_sequence = CustomSequence(video_path, labels.iloc[splits[3]], int_data_dir, BATCH_SIZE)
    test_sequence = CustomSequence(video_path, labels.iloc[splits[4]], int_data_dir, BATCH_SIZE)

    return train_sequence, validation_sequence, test_sequence
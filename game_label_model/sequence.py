import os

from tensorflow.keras.utils import Sequence, to_categorical
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from .labels import NUM_CLASSES
from .labels import load_from_file as load_labels_from_file
from .video_handler import VideoHandler
from .hyperparameters import NUM_CNNS

class CustomSequence(Sequence):
    def __init__(self, vid_path, labels, int_dir, batch_size):
        self.labels = labels

        self.batch_size = batch_size

        self.video_handler = VideoHandler().load_video(vid_path)
        self.intermidiate_dir = int_dir

    def __len__(self):
        return int(np.floor(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        frames = list(range(index*self.batch_size, (index+1)*self.batch_size))

        ## Add the yolo bounding boxes from pregenned
        batch_xs = [[] for _ in range(0, NUM_CNNS)]
        batch_y = []

        for frame in frames:
            bbs_path = os.path.join(self.intermidiate_dir, f"yolo-{frame}.npy")

            for bb, batch_x in zip(np.load(bbs_path), batch_xs):
                batch_x.append(bb)

            batch_y.append(self.labels.iloc[frame]["label"].value - 1)

        ## Convert to np arrays
        batch_x_final = [np.array(xs) for xs in batch_xs]

        return batch_x_final, to_categorical(batch_y, num_classes=NUM_CLASSES)

    def on_epoch_end(self):
        ## Shuffle the rows
        self.labels = self.labels.sample(frac=1).reset_index(drop=True)


def get_train_test_val(video_path, int_data_dir, labels_path):
    labels = pd.DataFrame(load_labels_from_file(labels_path), columns=["label"])
    labels["frame"] = labels.index

    ## Split into train, test, validation
    skf = StratifiedKFold(n_splits=5)
    
    splits = [test_index for _, test_index in skf.split(labels, labels["label"])]

    train_indecies = np.concatenate([splits[i] for i in [0, 1, 2]])

    train_sequence = CustomSequence(video_path, labels.iloc[train_indecies], int_data_dir, 32)
    validation_sequence = CustomSequence(video_path, labels.iloc[splits[3]], int_data_dir, 32)
    test_sequence = CustomSequence(video_path, labels.iloc[splits[4]], int_data_dir, 32)

    return train_sequence, validation_sequence, test_sequence
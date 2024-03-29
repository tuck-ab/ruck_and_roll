import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow.data import TFRecordDataset
from tensorflow.image import resize
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.models import load_model

from .data_gen import get_bb_ims
from .graph_model import (GraphGenerator, convert_label, create_graph_tensor,
                          decode_fn, label_to_numeric, write_tensors)
from .hyperparameters import (BATCH_SIZE, CLIP_SIZE, NUM_CNNS,
                              THREED_CNN_INPUT_SHAPE)
from .labels import LABELS, NUM_CLASSES, Label
from .labels import load_from_file as load_labels_from_file
from .video_handler import VideoHandler
from .yolo_handler import YOLORunner

# YOLO_MODEL = os.path.join(pathlib.Path(__file__).parent, "yolov7_480x640.onnx")
# TFRECORD_FILEPATH = os.path.join(pathlib.Path(__file__).parent, "graph.tfrecords")

class CustomSequence(Sequence):
    def __init__(self, vid_path, yolo_path, gnn_path, 
                 labels, int_dir, batch_size, clip_size=CLIP_SIZE,
                 image_size=THREED_CNN_INPUT_SHAPE[1:3]):
        self.labels = labels

        self.batch_size = batch_size
        self.clip_size = clip_size
        self.image_size = image_size

        self.video_handler = VideoHandler().load_video(vid_path)
        self.intermidiate_dir = int_dir
        self.yolo_handler = YOLORunner(yolo_path)
        self.tfrecord_path = os.path.join(gnn_path, "graph.tfrecords")
        self.gnn_model = os.path.join(gnn_path, "22")

    def __len__(self):
        return int(np.floor(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        indicies = list(range(index*self.batch_size, (index+1)*self.batch_size))

        ## Add the yolo bounding boxes from pregenned
        batch_bb_xs = [[] for _ in range(0, NUM_CNNS)]
        batch_y = []

        clips = []
        graph_predictions = []

        gnn_model = load_model(self.gnn_model, compile=True)

        for i in indicies:
            frame = self.labels.iloc[i]["frame"]
            # bbs_path = os.path.join(self.intermidiate_dir, f"yolo-{frame}.npy")

            # for bb, batch_x in zip(np.load(bbs_path), batch_bb_xs):
            #     batch_x.append(bb)

            ## BB generation

            # GNN Preprocessing

            image = self.video_handler.get_frame(frame)
            for bb, batch_x in zip(get_bb_ims(self.yolo_handler, image), batch_bb_xs):
                batch_x.append(bb)
            results = self.yolo_handler.run(image)

            graph_gen = GraphGenerator(results)
            m, nodes = graph_gen.get_graph().get_edge_matrix()

            ourLabel = convert_label(self.labels.iloc[i]["label"].name, LABELS)
            ourNumericLabel = label_to_numeric(ourLabel, LABELS)

            g = create_graph_tensor(nodes, m, ourNumericLabel)

            write_tensors([g], self.tfrecord_path)

            graph_ds = TFRecordDataset([self.tfrecord_path]).map(decode_fn)

            batch_size = 32
            graph_ds_batched = graph_ds.batch(batch_size=batch_size)

            prediction = gnn_model.predict(graph_ds_batched, verbose=0)

            graph_predictions.append(prediction[0])

            # End of GNN Preprocessing

            batch_y.append(self.labels.iloc[i]["label"].value - 1)

            clip = self.video_handler.get_clip(frame-(self.clip_size//2), self.clip_size)
            clip = resize(np.array(clip), self.image_size).numpy()
            clips.append(clip)

        ## Convert to np arrays
        batch_x_final = [np.array(clips)]
        for xs in batch_bb_xs:
            batch_x_final.append(np.array(xs))
        # for pred in graph_predictions:
        
        batch_x_final.append(np.array(graph_predictions))

        return batch_x_final, to_categorical(batch_y, num_classes=NUM_CLASSES)

    def on_epoch_end(self):
        ## Shuffle the rows
        self.labels = self.labels.sample(frac=1).reset_index(drop=True)

def balance_labels(indicies, labels, limit):
    print("\nPerforming Label Balancing")
    total = np.zeros(NUM_CLASSES - 1, np.float32)
    for i in range(0, len(indicies)):
        total[labels.iloc[indicies[i]]["label"].value - 2] += 1

    numFrames = np.sum(total)
    numPerClass = limit / (NUM_CLASSES - 1)
    print("Total Frames: {}".format(numFrames))
    print(numPerClass, NUM_CLASSES)
    print("Breakdown: {}".format(total))

    new_indicies = []
    new_totals = np.zeros(NUM_CLASSES - 1, np.float32)
    for i in range(0, len(indicies)):
        label_val = labels.iloc[indicies[i]]["label"].value - 2
        prob = numPerClass / total[label_val]
        if random.uniform(0, 1) <= prob:
            new_indicies.append(indicies[i])
            new_totals[label_val] += 1

    print("New Breakdown: {}".format(new_totals))
    print("Total frames: {}\n".format(np.sum(new_totals)))

    return new_indicies

def get_train_test_val(video_path, yolo_path, graph_path, int_data_dir, labels_path, clip_size=CLIP_SIZE, limit=3000):
    labels = pd.DataFrame(load_labels_from_file(labels_path), columns=["label"])
    labels["frame"] = labels.index

    labels = labels[labels["label"].apply(lambda x:x.value != Label.NOTHING.value)].reset_index(drop=True)

    #if limit:
    #    labels = labels[:limit]

    buff_size = (clip_size // 2) + 1
    labels = labels.iloc[buff_size:-buff_size]

    ## Split into train, test, validation
    skf = StratifiedKFold(n_splits=5)
    
    splits = [test_index for _, test_index in skf.split(labels, labels["label"].apply(lambda x: x.value))]

    train_indecies = np.concatenate([splits[i] for i in [0, 1, 2]])
    
    # Label balancing
    train_indecies = balance_labels(train_indecies, labels, limit)
    splits[3] = balance_labels(splits[3], labels, limit / 3)
    splits[4] = balance_labels(splits[4], labels, limit / 3)

    train_sequence = CustomSequence(video_path, yolo_path, graph_path, labels.iloc[train_indecies], int_data_dir, BATCH_SIZE)
    validation_sequence = CustomSequence(video_path, yolo_path, graph_path, labels.iloc[splits[3]], int_data_dir, BATCH_SIZE)
    test_sequence = CustomSequence(video_path, yolo_path, graph_path, labels.iloc[splits[4]], int_data_dir, BATCH_SIZE)

    return train_sequence, validation_sequence, test_sequence

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

from .graph_gen import GraphGenerator
from .bounding_box_store import BoundingBoxStore
from .yolo_handler.yolo_frame import YOLORunner

# Graph Neural Network Integration

import tensorflow as tf
import tensorflow_gnn as tfgnn
import pathlib

YOLO_MODEL = "yolov7_480x640.onnx"
TFRECORD_FILEPATH = os.path.join(pathlib.Path(__file__).parent, "graph.tfrecords")

# GNN Labels
LABELS_TO_PREDICT = [
    "CARRY",
    "PASS",
    "KICK",
    "RUCK",
    "TACKLE",
    "LINEOUT",
    "SCRUM",
    "MAUL"    
]

# Convert the labels into our format for predicts
def convert_label(label, to_predict):
    for lab in to_predict:
        if lab in label:
            return lab
    return "NOTHING"

# Convert labels to a numeric value
def label_to_numeric(label, to_predict):
    return to_predict.index(label)

# Graph spec for the model
graph_tensor_spec = tfgnn.GraphTensorSpec.from_piece_specs(
    context_spec=tfgnn.ContextSpec.from_field_specs(features_spec={
                  'label': tf.TensorSpec(shape=(len(LABELS_TO_PREDICT),), dtype=tf.int32)
    }),
    node_sets_spec={
        'players':
            tfgnn.NodeSetSpec.from_field_specs(
                features_spec={
                    tfgnn.HIDDEN_STATE:
                        tf.TensorSpec((None, 2), tf.float32)
                },
                sizes_spec=tf.TensorSpec((1,), tf.int32))
    },
    edge_sets_spec={
        'distances':
            tfgnn.EdgeSetSpec.from_field_specs(
                features_spec={
                    tfgnn.HIDDEN_STATE:
                        tf.TensorSpec((None, 1), tf.float32)
                },
                sizes_spec=tf.TensorSpec((1,), tf.int32),
                adjacency_spec=tfgnn.AdjacencySpec.from_incident_node_sets(
                    'players', 'players'))
    })

# Convert the nodes, edges, weights, and label for a graph into the correct format. Stored as a list
def correct_format(nodes_old, edges_old, label_old):
    nodes = np.arange(len(nodes_old))
    edge_df = pd.DataFrame(columns = ['source', 'dest', 'weight'], dtype='float64')
    for i in range(0, len(edges_old)):
        for j in range(0, len(edges_old[0])):
            temp_df = pd.DataFrame([[i, j, edges_old[i][j]]],columns = ['source', 'dest', 'weight'], dtype='float64')
            edge_df = pd.concat([edge_df, temp_df], ignore_index=True)
    label_df = pd.DataFrame(columns = ['label'], dtype = 'int32', index=[0])
    label_df['label'] = label_old
    arr = []
    arr.append(nodes)
    arr.append(edge_df)
    arr.append(label_df)
    return arr

# Create the graph tensor for a given graph
# Inputs given as the outputs to the correct_format function
def create_graph_tensor(nodes, edges, labelID):
    arr = correct_format(nodes, edges, labelID)
    edges_df = arr[1]
    sources = np.array(edges_df['source'], dtype='int32')
    dests = np.array(edges_df['dest'], dtype='int32')
    weight = np.array(edges_df['weight'], dtype='float32')
    node_feats = []
    for i in range(0, len(nodes)):
        if i < len(nodes) - 1:
            node_feats.append([0.0,1.0])
        else:
            node_feats.append([1.0,0.0])
    label_matrix = np.zeros(len(LABELS_TO_PREDICT))
    label_matrix[labelID] = 1
    label_matrix = label_matrix.astype('int64')
    g = tfgnn.GraphTensor.from_pieces(
    node_sets = {
      'players': tfgnn.NodeSet.from_fields(sizes=[len(nodes)], features={'hidden_state': node_feats})},
    edge_sets = {
      'distances': tfgnn.EdgeSet.from_fields(
         sizes=[len(nodes) * len(nodes)],
         features={'hidden_state' : weight},
         adjacency=tfgnn.Adjacency.from_indices(
           source=('players', sources),
           target=('players', dests)))},
    context =
        tfgnn.Context.from_fields(
            features={'label': label_matrix}, sizes=[len(label_matrix)]
        ))
    
    return g

# Save the graph tensors into one file
def write_tensors(tensors, filename):
    with tf.io.TFRecordWriter(filename) as writer:
        for i in range(0, len(tensors)):
            graph = tensors[i]
            example = tfgnn.write_example(graph)
            writer.write(example.SerializeToString())

# Function to reload the data back in in the correct format
def decode_fn(record_bytes):
  graph = tfgnn.parse_single_example(
      graph_tensor_spec, record_bytes, validate=True)

  # extract label from context and remove from input graph
  context_features = graph.context.get_features_dict()
  print(context_features)
  label = context_features.pop('label')
  print(context_features)
  new_graph = graph.replace_features(context=context_features)

  return new_graph, label


class CustomSequence(Sequence):
    def __init__(self, vid_path, labels, int_dir, batch_size, clip_size=CLIP_SIZE,
                 image_size=THREED_CNN_INPUT_SHAPE[1:3]):
        self.labels = labels

        self.batch_size = batch_size
        self.clip_size = clip_size
        self.image_size = image_size

        self.video_handler = VideoHandler().load_video(vid_path)
        self.intermidiate_dir = int_dir
        self.yolo_handler = YOLORunner(YOLO_MODEL)

    def __len__(self):
        return int(np.floor(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        indicies = list(range(index*self.batch_size, (index+1)*self.batch_size))

        ## Add the yolo bounding boxes from pregenned
        batch_bb_xs = [[] for _ in range(0, NUM_CNNS)]
        batch_y = []

        clips = []
        graph_tensor_datasets = []

        for i in indicies:
            frame = self.labels.iloc[i]["frame"]
            bbs_path = os.path.join(self.intermidiate_dir, f"yolo-{frame}.npy")

            for bb, batch_x in zip(np.load(bbs_path), batch_bb_xs):
                batch_x.append(bb)

            # GNN Preprocessing

            image = self.video_handler.get_frame(frame)
            results = self.yolo_handler.run(image)

            graph_gen = GraphGenerator(*results)
            m, nodes = graph_gen.get_graph().get_edge_matrix()

            ourLabel = convert_label(self.labels.iloc[i]["label"].name, LABELS_TO_PREDICT)
            ourNumericLabel = label_to_numeric(ourLabel, LABELS_TO_PREDICT)

            g = create_graph_tensor(nodes, m, ourNumericLabel)

            write_tensors([g], TFRECORD_FILEPATH)

            graph_ds = tf.data.TFRecordDataset([TFRECORD_FILEPATH]).map(decode_fn)

            batch_size = 32
            graph_ds_batched = graph_ds.batch(batch_size=batch_size)

            graph_tensor_datasets.append(graph_ds_batched)

            # End of GNN Preprocessing

            batch_y.append(self.labels.iloc[i]["label"].value - 1)

            clip = self.video_handler.get_clip(frame-(self.clip_size//2), self.clip_size)
            clip = resize(np.array(clip), self.image_size).numpy()
            clips.append(clip)

        ## Convert to np arrays
        batch_x_final = [np.array(clips)]
        for xs in batch_bb_xs:
            batch_x_final.append(np.array(xs))
        for graph_tensor in graph_tensor_datasets:
            batch_x_final.append(graph_tensor)

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
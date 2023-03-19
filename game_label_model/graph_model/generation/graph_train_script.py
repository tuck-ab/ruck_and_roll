import numpy as np
import math
import argparse
import pathlib
import os
import cv2
import time
import re
import random


def get_label(index, run, label_tuples, to_predict):
    tuple = label_tuples[index]
    end = False
    if tuple[1] <= run:
        index += 1
        run = 0
        try:
            tuple = label_tuples[index]
        except:
            end = True
    return index, run + 1, "NOTHING" if end else convert_label(tuple[0], to_predict)

def convert_label(label, to_predict):
    for lab in to_predict:
        if label in lab:
            return lab
    return "NOTHING"


FILE_DIR = pathlib.Path(__file__).parent
LABEL_DIR = os.path.join(FILE_DIR, "labels")
OUT_DIR = os.path.join(FILE_DIR, "outputs")

game_names = ["galleinvorLabels", "221015ramsvcambridge-bip", "2022101cambridgevsale-fc-bip", "20220903cambridgevplymouthnational-1-bip", "cambridge-v-cinderford-bip", "20220917cambridgevtauntonnational-1-bip", "20220910bishops-stortfordvcambridgenational-1-bi", "20220924dmpvcambridge"]
# game_names = ["220611galleivnor_2_movie-001from0to48000", "220611galleivnor_2_movie-001from48001for15000frames", "220611galleivnor_2_movie-001from63001for 22965 frames", "220611galleivnor_2_movie-001from85966 for 51450 frames", "220611galleivnor_2_movie-001from137417"]
exists = []
raw_lists = []
num_frames = []

for name in game_names:
    try:
        labelLocation = os.path.join(LABEL_DIR, name + ".lbl")
        with open(labelLocation) as f:
            lines = f.readlines()
        exists.append(True)
    except:
        exists.append(False)
        continue

    label_tuples = list()
    tot = 0
    for line in lines:
        label_match = re.search(r'(?<=Label.)(.*)(?=:)',line)
        frame_match = re.search(r'(?<=:)(.*)(?=\n)',line)
        if frame_match is None:
            frame_match = re.search(r'(?<=:)(.*)',line)
        label_tuples.append((label_match.group(),int(frame_match.group())))
        tot += int(frame_match.group())
    raw_lists.append(label_tuples)
    num_frames.append(tot)


LABELS = [
    "NOTHING",
    "CARRY",
    "PASS_L",
    "PASS_R",
    "KICK_L",
    "KICK_R",
    "RUCK",
    "TACKLE_S_D", ## Tackle, Single, Dominant
    "TACKLE_S", ## Tackle, Single
    "TACKLE_D_D", ## Tackle, Double, Dominant
    "TACKLE_D", ## Tackle, Double
    "TACKLE_M", ## Tackle, Missed
    "LINEOUT",
    "SCRUM",
    "MAUL"    
]

LABELS_TO_PREDICT = [
    "CARRY",
    "PASS",
    "KICK",
    "RUCK",
    "TACKLE", ## Tackle, Single, Dominant
    "LINEOUT",
    "SCRUM",
    "MAUL"    
]

split_percent = 0.7
for i in range(0, len(raw_lists)):
    label_tuples = raw_lists[i]
    train_list_raw = []
    test_list_raw = []
    validation_list_raw = []
    validation_test_list_raw = []
    index = 0
    run = 0
    for j in range(0, num_frames[i]):
        train = random.randint(1,10) <= split_percent * 10
        _, _, next_label = get_label(index, run + 1, label_tuples, LABELS_TO_PREDICT)
        index, run, label = get_label(index, run, label_tuples, LABELS_TO_PREDICT)
        if "NOTHING" not in label:
            if train:
                train_list_raw.append(j)
                test_list_raw.append(label)
            else:
                validation_list_raw.append(j)
                validation_test_list_raw.append(label)


######## Graph Model Training

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import tensorflow_gnn as tfgnn
# import tensorflow_datasets as tfds

from tensorflow_gnn import runner
from tensorflow_gnn.models import gat_v2

def create_graph_tensor(nodes, edges):
    newNodes =[]
    for i in range(0, len(nodes)):
        newNodes.append(i)
    players = tfgnn.NodeSet.from_fields(features={'players': newNodes}, sizes=tf.constant([len(newNodes)]))
    shape = edges.shape
    sources, targets = build_adjacency(nodes, edges)
    player_adjacency = tfgnn.Adjacency.from_indices(source=('player', tf.cast(sources, dtype=tf.int32)),
                                                  target=('player', tf.cast(targets, dtype=tf.int32)))
    edge_set = tfgnn.EdgeSet.from_fields(features = {
                    'weight': edges.reshape(len(edges) ** 2,1)}, adjacency = player_adjacency, sizes = tf.constant([shape[0], shape[1]]))
    # edge_set = tfgnn.EdgeSet.from_fields(adjacency=edges, sizes=tf.constant([shape[0], shape[1]]))
    
    return tfgnn.GraphTensor.from_pieces(node_sets={'players': players}, edge_sets={'weight': edge_set})

def build_adjacency(nodes, edges):
    sources = []
    targets = []
    for i in range(0, len(edges)):
        for j in range(0, len(edges[i])):
            sources.append(i)
            targets.append(j)
    return sources, targets

edges_1 = np.loadtxt("1.txt")
nodes_1 = np.loadtxt("1_nodes.txt", dtype=str)
edges_2 = np.loadtxt("2.txt")
nodes_2 = np.loadtxt("2_nodes.txt", dtype=str)

tensor_1 = create_graph_tensor(nodes_1, edges_1)
tensor_2 = create_graph_tensor(nodes_2, edges_2)

graphs = [tensor_1, tensor_2]
labels = [LABELS_TO_PREDICT[0], LABELS_TO_PREDICT[1]]

# print(tensor_1.edge_sets)
# print(tensor_2)

# tf.print(tensor_1)
# tf.print(tensor_1, [tensor_1], message="This is a: ")


# I believe this is the key to solving the issues currently held
# The problem is that it relies on each node having a label to be predicted. Currently only the entire graph has a label associated
# This requires modifications to the create graph tensor function...
def edge_batch_merge(graph):
    graph = graph.merge_batch_to_components()
    node_features = graph.node_sets['schools'].get_features_dict()
    edge_features = graph.edge_sets['games'].get_features_dict()
    
    _ = node_features.pop('conference')
    label = edge_features.pop('conference_game')
    
    new_graph = graph.replace_features(
        node_sets={'schools':node_features},
        edge_sets={'games':edge_features})
    return new_graph, label

# def create_dataset(graphs,function):
#     dataset = tf.data.Dataset.from_tensors(graphs)
#     dataset = dataset.batch(32)
#     return dataset.map(function)

def create_dataset(graph, label):
    #takes the tensor and coverts to dataset tensor
    dataset = tf.data.Dataset.from_tensors(graph)
    #here is where batch splitting and mapping over functions to split target labels from training data
    # tf.data.Dataset.
    # dataset = tf.data.Dataset.zip((graphs, labels))
    # return dataset
    return dataset.zip(dataset, label)

newGraphs = []
for graph in graphs:
    newGraphs.append(graph.merge_batch_to_components())

train_ds = []
for i in range(0, len(graphs)):
    train_dataset = create_dataset(graphs[i], labels[i])
    train_ds.append(train_dataset)

model_input_graph_spec, label_spec = train_dataset.element_spec
input_graph = tf.keras.layers.Input(type_spec=model_input_graph_spec)

def set_initial_node_state(node_set, node_set_name):
    return tfgnn.keras.layers.MakeEmptyFeature()(node_set)


def set_initial_edge_state(edge_set, edge_set_name):
    features = [
        tf.keras.layers.Dense(32,activation="relu")(edge_set['weight']),
    ]
    return tf.keras.layers.Concatenate()(features)


graph_nn = tfgnn.keras.layers.MapFeatures(
    node_sets_fn=set_initial_node_state,
    edge_sets_fn=set_initial_edge_state)(input_graph)



graph_updates = 5 #hyper parameter
activations = ['relu', 'gelu', 'sigmoid', 'sigmoid', 'softmax']
for x in range(graph_updates):
    graph = tfgnn.keras.layers.GraphUpdate(
        edge_sets = {'weight': tfgnn.keras.layers.EdgeSetUpdate(
            next_state = tfgnn.keras.layers.NextStateFromConcat(
                dense_layer(64,activation=activations[x])))},
        node_sets = {
            'Players': tfgnn.keras.layers.NodeSetUpdate({
                'weight': tfgnn.keras.layers.SimpleConv(
                    message_fn = dense_layer(32),
                    reduce_type="mean",
                    receiver_tag=tfgnn.TARGET)},
                tfgnn.keras.layers.NextStateFromConcat(
                    dense_layer(64)))})(graph)
    
#not entirely sure how to state this layer but we want a softmax layer with number of nodes equal to labels to be taken from
#another way to do this would be to pool node/edge states to classify the whole graph
logits = tf.keras.layers.Dense(len(LABELS_TO_PREDICT) ,activation='softmax')(LABELS_TO_PREDICT[tfgnn.HIDDEN_STATE])

our_model = tf.keras.Model(input_graph, logits)

our_model.compile(
    tf.keras.optimizers.Adam(learning_rate=0.01),
    loss = 'categorical_crossentropy',
    metrics = ['categorical_accuracy']
)

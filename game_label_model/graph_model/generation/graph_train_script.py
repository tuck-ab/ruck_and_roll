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

def label_to_numeric(label, to_predict):
    return to_predict.index(label)
    


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
import pandas as pd

def create_graph_tensor(nodes, edges, labelID):
    newNodes =[]
    for i in range(0, len(nodes)):
        newNodes.append(i)
    players = tfgnn.NodeSet.from_fields(features={'hidden': newNodes}, sizes=[len(newNodes)])

    shape = edges.shape
    sources, targets = build_adjacency(nodes, edges)
    player_adjacency = tfgnn.Adjacency.from_indices(source=('players', tf.cast(sources, dtype=tf.int32)),
                                                  target=('players', tf.cast(targets, dtype=tf.int32)))
    edge_set = tfgnn.EdgeSet.from_fields(features = {
                    'edge_weight': edges.reshape(len(edges) ** 2,1)}, 
                    adjacency = tfgnn.HyperAdjacency.from_indices(
                        indices={
                            tfgnn.SOURCE: ('players', sources),
                            tfgnn.TARGET: ('players', targets)
                        }
                    ), 
                    sizes = [len(edges) ** 2]
                    )
    # edge_set = tfgnn.EdgeSet.from_fields(adjacency=edges, sizes=tf.constant([shape[0], shape[1]]))
    test_df = pd.DataFrame([labelID], columns = ['label'], dtype = 'int32', index=[0])
    placeholder_df = pd.DataFrame([42], columns=['placeholder'], dtype = 'int32', index = [0])
    return tfgnn.GraphTensor.from_pieces(context=tfgnn.Context.from_fields(
                    features={'action_label': test_df, 'placeholder' : placeholder_df}, sizes=[len(test_df)]), 
                    node_sets={'players': tfgnn.NodeSet.from_fields(
                        features={'hidden': tf.constant(newNodes)}, 
                        sizes=[len(newNodes)])
                        }, 
                    edge_sets={'edge': tfgnn.EdgeSet.from_fields(features = {
                        'edge_weight': tf.constant(edges.reshape(len(edges) ** 2,1))}, 
                            adjacency = tfgnn.HyperAdjacency.from_indices(
                                indices={
                                    tfgnn.SOURCE: ('players', sources),
                                    tfgnn.TARGET: ('players', targets)
                                }
                            ), 
                            sizes = [len(edges) ** 2]
                            )}
                    )

# def create_graph_tensor(nodes, edges, labelID):
#     newNodes =[]
#     for i in range(0, len(nodes)):
#         newNodes.append(i)
#     players = tfgnn.NodeSet.from_fields(features={'players': newNodes}, sizes=tf.constant([len(newNodes)]))
#     shape = edges.shape
#     sources, targets = build_adjacency(nodes, edges)
#     player_adjacency = tfgnn.Adjacency.from_indices(source=('players', tf.cast(sources, dtype=tf.int32)),
#                                                   target=('players', tf.cast(targets, dtype=tf.int32)))
#     edge_set = tfgnn.EdgeSet.from_fields(features = {
#                     'weight': edges.reshape(len(edges) ** 2,1)}, adjacency = player_adjacency, sizes = tf.constant([shape[0], shape[1]]))
#     # edge_set = tfgnn.EdgeSet.from_fields(adjacency=edges, sizes=tf.constant([shape[0], shape[1]]))
    
#     return tfgnn.GraphTensor.from_pieces(context=tfgnn.Context.from_fields(
#                     features={'action_label': [labelID]}), node_sets={'players': players}, edge_sets={'weight': edge_set})

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

labels = [LABELS_TO_PREDICT[0], LABELS_TO_PREDICT[1]]

tensor_1 = create_graph_tensor(nodes_1, edges_1, label_to_numeric(labels[0], LABELS_TO_PREDICT))
tensor_2 = create_graph_tensor(nodes_2, edges_2, label_to_numeric(labels[1], LABELS_TO_PREDICT))

graphs = [tensor_1, tensor_2]

# print(tensor_1.edge_sets)
# print(tensor_2)

# tf.print(tensor_1)
# tf.print(tensor_1, [tensor_1], message="This is a: ")


# I believe this is the key to solving the issues currently held
# The problem is that it relies on each node having a label to be predicted. Currently only the entire graph has a label associated
# This requires modifications to the create graph tensor function...
# def edge_batch_merge(graph):
#     graph = graph.merge_batch_to_components()
#     node_features = graph.node_sets['schools'].get_features_dict()
#     edge_features = graph.edge_sets['games'].get_features_dict()
    
#     _ = node_features.pop('conference')
#     label = edge_features.pop('conference_game')
    
#     new_graph = graph.replace_features(
#         node_sets={'schools':node_features},
#         edge_sets={'games':edge_features})
#     return new_graph, label

# def create_dataset(graphs,function):
#     dataset = tf.data.Dataset.from_tensors(graphs)
#     dataset = dataset.batch(32)
#     return dataset.map(function)

# def create_dataset(graph, label):
#     #takes the tensor and coverts to dataset tensor
#     dataset = tf.data.Dataset.from_tensors([graph])
#     #here is where batch splitting and mapping over functions to split target labels from training data
#     # tf.data.Dataset.
#     # dataset = tf.data.Dataset.zip((graphs, labels))
#     # return dataset
#     # return dataset.zip(dataset, label)
#     return dataset

BATCH_SIZE = 2
#I think this just takes a batch of graphs and makes them into 1? idk im still confused by tensorflow datasets
def batch_merge(graph):
    graph = graph.merge_batch_to_components()
    node_features = graph.node_sets['players'].get_features_dict()
    edge_features = graph.edge_sets['edge'].get_features_dict()
    context_features = graph.context.get_features_dict()
    
    label = context_features.pop('action_label')
    new_graph = graph.replace_features(
        node_sets={'players':node_features},
        edge_sets={'edge':edge_features},
        context = context_features)#{'action':np.zeros(BATCH_SIZE)})
    return new_graph, label

def create_dataset(graphs, func):
    print("creating datasets")
    print(len(graphs))
    datasets= []
    for graph in graphs:
        dataset = tf.data.Dataset.from_tensors(graph)
        dataset = dataset.batch(BATCH_SIZE)
        datasets.append(dataset.map(func))

    
    return datasets

def dense_layer(self,units=64,l2_reg=0.1,dropout=0.25,activation='relu'):
    regularizer = tf.keras.regularizers.l2(l2_reg)
    return tf.keras.Sequential([
        tf.keras.layers.Dense(units,
                              kernel_regularizer=regularizer,
                              bias_regularizer=regularizer),
        tf.keras.layers.Dropout(dropout)])

newGraphs = []
for graph in graphs:
    newGraphs.append(graph.merge_batch_to_components())

train_set = create_dataset(graphs, batch_merge)

# train_ds = []
# for i in range(0, len(graphs)):
#     train_dataset = create_dataset(graphs[i], labels[i])
#     train_ds.append(train_dataset)

model_input_graph_spec, label_spec = train_set[0].element_spec
input_graph = tf.keras.layers.Input(type_spec=model_input_graph_spec)

def set_initial_node_state(node_set, node_set_name):
    return tfgnn.keras.layers.MakeEmptyFeature()(node_set)


def set_initial_edge_state(edge_set, edge_set_name):
    features = [
        tf.keras.layers.Dense(32,activation="relu")(edge_set['edge_weight']),
    ]
    return tf.keras.layers.Concatenate()(features)


graph_nn = tfgnn.keras.layers.MapFeatures(
    node_sets_fn=set_initial_node_state,
    edge_sets_fn=set_initial_edge_state)(input_graph)



graph_updates = 5 #hyper parameter
activations = ['relu', 'gelu', 'sigmoid', 'sigmoid', 'softmax']

from typing import Tuple
from tensorflow_gnn.graph.graph_constants import FieldsNest



print("\n"*5)



def build_graph_tensor(
    num_edges: int = 10, num_nodes: int = 10, num_features: int = 5
) -> tfgnn.GraphTensor:
    adjacency = tf.random.stateless_binomial(
        shape=[num_edges, 2], counts=[0, 1], probs=[0.5, 0.5], seed=[0, 0]
    )
    # adjacency = [[0,2], [3,6], [2,9], [1,3], [5,7], [7,8], [9,9], [3,5], [5,6], [4,2]]
    print("Adjacency :,0 and :,1")
    print(adjacency)
    print(adjacency[:, 0])
    print(adjacency[:, 1])
    print("Edges features")
    print(tf.random.normal(shape=[num_features]))
    print([num_edges])
    edges = tfgnn.EdgeSet.from_fields(
        features={"edge_weight": tf.random.normal(shape=[num_features])},
        sizes=[num_edges],
        adjacency=tfgnn.HyperAdjacency.from_indices(
            indices={
                tfgnn.SOURCE: ("node", adjacency[:, 0]),
                tfgnn.TARGET: ("node", adjacency[:, 1]),
            }
        ),
    )
    print("Edges")
    print(edges)
    print("Nodes hidden state (the nodes?)")
    print(tf.random.normal(shape=[num_nodes, num_features]))
    print([num_nodes])
    nodes = tfgnn.NodeSet.from_fields(
        features={"hidden_state": tf.random.normal(shape=[num_nodes, num_features])},
        sizes=[num_nodes],
    )
    print("Nodes")
    print(nodes)

    return tfgnn.GraphTensor.from_pieces(
        context=tfgnn.Context.from_fields(
                    features={'action_label': [1]}, sizes= [1], shape=np.array(1).shape), edge_sets={"edge": edges}, node_sets={"node": nodes}
    )


class MultiplyNodeEdge(tf.keras.layers.Layer):
    def __init__(self, edge_feature: str, node_feature: str = tfgnn.SOURCE) -> None:
        super().__init__()
        self.node_feature = node_feature
        self.edge_feature = edge_feature

    def call(self, inputs: Tuple[FieldsNest, FieldsNest, FieldsNest]) -> tf.Tensor:
        edge_inputs, node_inputs, _ = inputs
        return tf.multiply(
            edge_inputs[self.edge_feature], node_inputs[self.node_feature]
        )


graph_tensor = build_graph_tensor()
spec = graph_tensor.spec

input = tf.keras.layers.Input(type_spec=spec)
update = tfgnn.keras.layers.GraphUpdate(
    edge_sets={
        "edge": tfgnn.keras.layers.EdgeSetUpdate(
            next_state=MultiplyNodeEdge(edge_feature="edge_weight"),
            edge_input_feature=["edge_weight"],
        )
    },
    node_sets={
        "node": tfgnn.keras.layers.NodeSetUpdate(
            edge_set_inputs={"edge": tfgnn.keras.layers.Pool(tfgnn.TARGET, "sum")},
            next_state=tfgnn.keras.layers.NextStateFromConcat(
                tf.keras.layers.Dense(16)
            ),
        )
    },
)
graph = update(input)
hidden = tfgnn.keras.layers.Readout(node_set_name="node")(graph)
output = tf.keras.layers.Dense(1, activation="sigmoid")(hidden)
model = tf.keras.Model(input, output)

print(graph_tensor)
y = model(graph_tensor)
print(y)

model.compile(
    tf.keras.optimizers.Adam(learning_rate=0.01),
    loss = 'categorical_crossentropy',
    metrics = ['categorical_accuracy']
)

print("\n"*5)
print("End of working")





class MultiplyNodeEdge(tf.keras.layers.Layer):
    def __init__(self, edge_feature: str, node_feature: str = tfgnn.SOURCE) -> None:
        super().__init__()
        self.node_feature = node_feature
        self.edge_feature = edge_feature

    def call(self, inputs: Tuple[FieldsNest, FieldsNest, FieldsNest]) -> tf.Tensor:
        edge_inputs, node_inputs, _ = inputs
        return tf.multiply(
            edge_inputs[self.edge_feature], node_inputs[self.node_feature]
        )

graph_tensor = graphs[0]
print(graph_tensor)
spec = graph_tensor.spec

model_input_graph_spec, label_spec = train_set[0].element_spec
input_graph = tf.keras.layers.Input(type_spec=model_input_graph_spec)

# input = tf.keras.layers.Input(type_spec=spec)
# update = tfgnn.keras.layers.GraphUpdate(
#     edge_sets={
#         "edge": tfgnn.keras.layers.EdgeSetUpdate(
#             next_state=MultiplyNodeEdge(edge_feature="edge_weight"),
#             edge_input_feature=["edge_weight"],
#         )
#     },
#     node_sets={
#         "players": tfgnn.keras.layers.NodeSetUpdate(
#             edge_set_inputs={"edge": tfgnn.keras.layers.Pool(tfgnn.TARGET, "sum")},
#             next_state=tfgnn.keras.layers.NextStateFromConcat(
#                 tf.keras.layers.Dense(16)
#             ),
#         )
#     },
# )
# graph = update(input)
# hidden = tfgnn.keras.layers.Readout(node_set_name="players")(graph)
# output = tf.keras.layers.Dense(1, activation="sigmoid")(hidden)
# model = tf.keras.Model(input, output)


graph = input_graph
for x in range(graph_updates):
    graph = tfgnn.keras.layers.GraphUpdate(
        edge_sets = {'weight': tfgnn.keras.layers.EdgeSetUpdate(
            next_state = tfgnn.keras.layers.NextStateFromConcat(
                tf.keras.layers.Dense(64,activation=activations[x])))},
        node_sets = {
            'players': tfgnn.keras.layers.NodeSetUpdate(  # For node set "author".
                {"weight": tfgnn.keras.layers.SimpleConv(
                    tf.keras.layers.Dense(128, "relu"), "mean",
                    receiver_tag=tfgnn.SOURCE)},
                tfgnn.keras.layers.NextStateFromConcat(tf.keras.layers.Dense(128)))})(graph)



# for x in range(graph_updates):
#     graph = tfgnn.keras.layers.GraphUpdate(
#         edge_sets = {'weight': tfgnn.keras.layers.EdgeSetUpdate(
#             next_state = tfgnn.keras.layers.NextStateFromConcat(
#                 tf.keras.layers.Dense(64,activation=activations[x])))},
#         node_sets = {
#             'players': tfgnn.keras.layers.NodeSetUpdate(edge_set_inputs = {
#                 'weight': tfgnn.keras.layers.SimpleConv(
#                     message_fn = tf.keras.layers.Dense(32, activations[x]),
#                     reduce_type="mean",
#                     receiver_tag=tfgnn.SOURCE)},
#                 next_state = tfgnn.keras.layers.NextStateFromConcat(
#                     tf.keras.layers.Dense(64, activations[x])))})(graph)
    
# graph_updates = 5 #hyper parameter
# activations = ['relu', 'gelu', 'sigmoid', 'sigmoid', 'softmax']
# for x in range(graph_updates):
#     graph = tfgnn.keras.layers.GraphUpdate(
#         edge_sets = {'weight': tfgnn.keras.layers.EdgeSetUpdate(
#             next_state = tfgnn.keras.layers.NextStateFromConcat(
#                 dense_layer(64,activation=activations[x])))},
#         node_sets = {
#             'players': tfgnn.keras.layers.NodeSetUpdate({
#                 'weight': tfgnn.keras.layers.SimpleConv(
#                     message_fn = dense_layer(32),
#                     reduce_type="mean",
#                     receiver_tag=tfgnn.TARGET)},
#                 tfgnn.keras.layers.NextStateFromConcat(
#                     dense_layer(64)))})(graph)
    
#not entirely sure how to state this layer but we want a softmax layer with number of nodes equal to labels to be taken from
#another way to do this would be to pool node/edge states to classify the whole graph
logits = tf.keras.layers.Dense(len(LABELS_TO_PREDICT) ,activation='softmax')(LABELS_TO_PREDICT[tfgnn.HIDDEN_STATE])

our_model = tf.keras.Model(input_graph, logits)

our_model.compile(
    tf.keras.optimizers.Adam(learning_rate=0.01),
    loss = 'categorical_crossentropy',
    metrics = ['categorical_accuracy']
)

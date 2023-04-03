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

import os

# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
import pandas as pd

graph_tensor_spec = tfgnn.GraphTensorSpec.from_piece_specs(
    context_spec=tfgnn.ContextSpec.from_field_specs(features_spec={
                  'label': tf.TensorSpec(shape=(1,), dtype=tf.int32)
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
                        tf.TensorSpec((None, 50), tf.float32)
                },
                sizes_spec=tf.TensorSpec((1,), tf.int32),
                adjacency_spec=tfgnn.AdjacencySpec.from_incident_node_sets(
                    'players', 'players'))
    })

def build_adjacency(nodes, edges):
    sources = []
    targets = []
    for i in range(0, len(edges)):
        for j in range(0, len(edges[i])):
            sources.append(i)
            targets.append(j)
    return sources, targets

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
    return tfgnn.GraphTensor.from_pieces(context=tfgnn.Context.from_fields(
                    features={'label': test_df}, sizes=[len(test_df)]), 
                    node_sets={'players': tfgnn.NodeSet.from_fields(
                        features={'hidden': tf.constant(newNodes)}, 
                        sizes=[len(newNodes)])
                        }, 
                    edge_sets={'distances': tfgnn.EdgeSet.from_fields(features = {
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

def batch_merge(graph):
    graph = graph.merge_batch_to_components()
    node_features = graph.node_sets['players'].get_features_dict()
    edge_features = graph.edge_sets['distances'].get_features_dict()
    
    context_features = graph.context.get_features_dict()
    label = context_features.pop('label')
    new_graph = graph.replace_features(context=context_features)
    return new_graph, label

def decode_fn(record_bytes):
#   graph = tfgnn.parse_single_example(
#       graph_tensor_spec, record_bytes, validate=True)

  # extract label from context and remove from input graph
  context_features = graph.context.get_features_dict()
  label = context_features.pop('label')
  new_graph = graph.replace_features(context=context_features)

  return new_graph, label



###### Running gnn training ######

# Load data
edges_1 = np.loadtxt("1.txt")
nodes_1 = np.loadtxt("1_nodes.txt", dtype=str)
edges_2 = np.loadtxt("2.txt")
nodes_2 = np.loadtxt("2_nodes.txt", dtype=str)

labels = [LABELS_TO_PREDICT[0], LABELS_TO_PREDICT[1]]

tensor_1 = create_graph_tensor(nodes_1, edges_1, label_to_numeric(labels[0], LABELS_TO_PREDICT))
tensor_2 = create_graph_tensor(nodes_2, edges_2, label_to_numeric(labels[1], LABELS_TO_PREDICT))

graphs_old = [tensor_1, tensor_2]
graphs = []

# Map graphs to correct format
for graph in graphs_old:
    g, l = batch_merge(graph)
    graphs.append((g, l))

# Create Datasets
BATCH_SIZE = 32
def create_dataset(graphs, func):
    print("creating datasets")
    print(len(graphs))
    datasets= []
    for graph in graphs:
        dataset = tf.data.Dataset.from_tensors(graph)
        dataset = dataset.batch(BATCH_SIZE).repeat()
        datasets.append(dataset.map(func))
    return datasets

train_ds = create_dataset(graphs_old, decode_fn)

g, y = train_ds.take(1).get_single_element()
print(g)
print(y)


# Batch the datasets
# batch_size = 32
# train_ds_batched = train_ds.batch(batch_size=batch_size).repeat()
# val_ds_batched = val_ds.batch(batch_size=batch_size)


# Build the GNN model
def _build_model(
    graph_tensor_spec,
    # Dimensions of initial states.
    node_dim=16,
    edge_dim=16,
    # Dimensions for message passing.
    message_dim=64,
    next_state_dim=64,
    # Dimension for the logits.
    num_classes=2,
    # Number of message passing steps.
    num_message_passing=3,
    # Other hyperparameters.
    l2_regularization=5e-4,
    dropout_rate=0.5,
):
  # Model building with Keras's Functional API starts with an input object
  # (a placeholder for the eventual inputs). Here is how it works for
  # GraphTensors:
  input_graph = tf.keras.layers.Input(type_spec=graph_tensor_spec)

  # IMPORTANT: All TF-GNN modeling code assumes a GraphTensor of shape []
  # in which the graphs of the input batch have been merged to components of
  # one contiguously indexed graph. (There are no edges between components,
  # so no information flows between them.)
  graph = input_graph.merge_batch_to_components()

  # Nodes and edges have one-hot encoded input features. Sending them through
  # a Dense layer effectively does a lookup in a trainable embedding table.
  def set_initial_node_state(node_set, *, node_set_name):
    # Since we only have one node set, we can ignore node_set_name.
    return tf.keras.layers.Dense(node_dim)(node_set[tfgnn.HIDDEN_STATE])
  def set_initial_edge_state(edge_set, *, edge_set_name):
    return tf.keras.layers.Dense(edge_dim)(edge_set[tfgnn.HIDDEN_STATE])
  graph = tfgnn.keras.layers.MapFeatures(
      node_sets_fn=set_initial_node_state, edge_sets_fn=set_initial_edge_state)(
          graph)

  # This helper function is just a short-hand for the code below.
  def dense(units, activation="relu"):
    """A Dense layer with regularization (L2 and Dropout)."""
    regularizer = tf.keras.regularizers.l2(l2_regularization)
    return tf.keras.Sequential([
        tf.keras.layers.Dense(
            units,
            activation=activation,
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer),
        tf.keras.layers.Dropout(dropout_rate)
    ])

  # The GNN core of the model does `num_message_passing` many updates of node
  # states conditioned on their neighbors and the edges connecting to them.
  # More precisely:
  #  - Each edge computes a message by applying a dense layer `message_fn`
  #    to the concatenation of node states of both endpoints (by default)
  #    and the edge's own unchanging feature embedding.
  #  - Messages are summed up at the common TARGET nodes of edges.
  #  - At each node, a dense layer is applied to the concatenation of the old
  #    node state with the summed edge inputs to compute the new node state.
  # Each iteration of the for-loop creates new Keras Layer objects, so each
  # round of updates gets its own trainable variables.
  for i in range(num_message_passing):
    graph = tfgnn.keras.layers.GraphUpdate(
        node_sets={
            "atoms": tfgnn.keras.layers.NodeSetUpdate(
                {"bonds": tfgnn.keras.layers.SimpleConv(
                     sender_edge_feature=tfgnn.HIDDEN_STATE,
                     message_fn=dense(message_dim),
                     reduce_type="sum",
                     receiver_tag=tfgnn.TARGET)},
                tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim)))}
    )(graph)

  # After the GNN has computed a context-aware representation of the "atoms",
  # the model reads out a representation for the graph as a whole by averaging
  # (pooling) nde states into the graph context. The context is global to each
  # input graph of the batch, so the first dimension of the result corresponds
  # to the batch dimension of the inputs (same as the labels).
  readout_features = tfgnn.keras.layers.Pool(
      tfgnn.CONTEXT, "mean", node_set_name="atoms")(graph)

  # Put a linear classifier on top (not followed by dropout).
  logits = tf.keras.layers.Dense(1)(readout_features)

  # Build a Keras Model for the transformation from input_graph to logits.
  return tf.keras.Model(inputs=[input_graph], outputs=[logits])

# Define loss metrics
model_input_graph_spec, label_spec = train_ds.element_spec
del label_spec # Unused.
model = _build_model(model_input_graph_spec)

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.BinaryAccuracy(threshold=0.),
            tf.keras.metrics.BinaryCrossentropy(from_logits=True)]

# Compile model
model.compile(tf.keras.optimizers.Adam(), loss=loss, metrics=metrics)
print(model.summary())

# Train model
# history = model.fit(train_ds_batched,
#                     steps_per_epoch=10,
#                     epochs=200,
#                     validation_data=val_ds_batched)
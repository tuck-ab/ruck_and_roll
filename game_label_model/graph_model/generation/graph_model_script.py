import numpy as np
import math
import argparse
import pathlib
import os
import cv2
import time
import re
import random

script_start = time.time()


# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--number', type=int, default='', help='Model number to train')
opt = parser.parse_args()

if opt.number == 0:
    activations = ["relu", "relu", "linear"]
    dimensions = [128, 128, 64]
    epochs = 100
elif opt.number == 1:
    activations = ["relu", "linear", "linear"]
    dimensions = [128, 128, 64]
    epochs = 100
elif opt.number == 2:
    activations = ["linear", "linear", "linear"]
    dimensions = [128, 128, 64]
    epochs = 100
elif opt.number == 3:
    activations = ["relu", "sigmoid", "elu"]
    dimensions = [128, 128, 64]
    epochs = 100
elif opt.number == 4:
    activations = ["relu", "sigmoid", "swish"]
    dimensions = [128, 128, 64]
    epochs = 100

print("Beginning Script {}".format(opt.number))

# Get a series of labels
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

# Convert the labels into our format for predicts
def convert_label(label, to_predict):
    for lab in to_predict:
        if label in lab:
            return lab
    return "NOTHING"

# Convert labels to a numeric value
def label_to_numeric(label, to_predict):
    return to_predict.index(label)
    
# File directories
FILE_DIR = pathlib.Path(__file__).parent
LABEL_DIR = os.path.join(FILE_DIR, "labels")
OUT_DIR = os.path.join(FILE_DIR, "outputs")

# Set of different game names
game_names = ["galleinvorLabels", "221015ramsvcambridge-bip", "2022101cambridgevsale-fc-bip", "20220903cambridgevplymouthnational-1-bip", "cambridge-v-cinderford-bip", "20220917cambridgevtauntonnational-1-bip", "20220910bishops-stortfordvcambridgenational-1-bi", "20220924dmpvcambridge"]
# game_names = ["220611galleivnor_2_movie-001from0to48000", "220611galleivnor_2_movie-001from48001for15000frames", "220611galleivnor_2_movie-001from63001for 22965 frames", "220611galleivnor_2_movie-001from85966 for 51450 frames", "220611galleivnor_2_movie-001from137417"]
exists = []
raw_lists = []
num_frames = []
names = []

print("Loading labels")
# Sort out regex of labels
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
    if name == game_names[0]:
        names.append("220611galleivnor_2_movie-001")
    else:
        names.append(name)
print("Labels Loaded")

# Original labels
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

# Labels the model is predicting
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

print("Loading node and edge files. Creating Test-Train Split")
# Custom test-train split ignores "nothing" labels
split_percent = 0.7
edges_train = []
nodes_train = []
labels_train = []
edges_val = []
nodes_val = []
labels_val = []
for i in range(0, len(raw_lists)):
    label_tuples = raw_lists[i]
    index = 0
    run = 0
    for j in range(0, num_frames[i]):
        train = random.randint(1,10) <= split_percent * 10
        _, _, next_label = get_label(index, run + 1, label_tuples, LABELS_TO_PREDICT)
        index, run, label = get_label(index, run, label_tuples, LABELS_TO_PREDICT)
        if "NOTHING" not in label:
            if train:
                labels_train.append(label)
                edges_train.append(np.loadtxt(os.path.join(OUT_DIR, names[i], "edges", str(j) + ".txt")))
                nodes_train.append(np.loadtxt(os.path.join(OUT_DIR, names[i], "nodes", str(j) + ".txt"), dtype=str))
            else:
                labels_val.append(label)
                edges_val.append(np.loadtxt(os.path.join(OUT_DIR, names[i], "edges", str(j) + ".txt")))
                nodes_val.append(np.loadtxt(os.path.join(OUT_DIR, names[i], "nodes", str(j) + ".txt"), dtype=str))
print("Test-Train Split Made")

######## Graph Model Training

# Necessary imports
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
import pandas as pd

# Graph spec for the model
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
                        tf.TensorSpec((None, 1), tf.float32)
                },
                sizes_spec=tf.TensorSpec((1,), tf.int32),
                adjacency_spec=tfgnn.AdjacencySpec.from_incident_node_sets(
                    'players', 'players'))
    })

# Build a set illustrating adjacencies as two lists
def build_adjacency(nodes, edges):
    sources = []
    targets = []
    for i in range(0, len(edges)):
        for j in range(0, len(edges[i])):
            sources.append(i)
            targets.append(j)
    return sources, targets

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
            features={'label': np.array(arr[2])}, sizes=[1]
        ))
    
    return g


###### Running gnn training ######

print("Creating train tensors")
# Train tensors
train_tensors = []
for i in range(0, len(edges_train)):
    tensor = create_graph_tensor(nodes_train[i], edges_train[i], label_to_numeric(labels_train[i], LABELS_TO_PREDICT))
    train_tensors.append(tensor)

print("Creating validation tensors")
# Validation tensors
val_tensors = []
for i in range(0, len(edges_val)):
    tensor = create_graph_tensor(nodes_val[i], edges_val[i], label_to_numeric(labels_val[i], LABELS_TO_PREDICT))
    val_tensors.append(tensor)


# Save the graph tensors into one file
def write_tensors(tensors, filename):
    with tf.io.TFRecordWriter(filename) as writer:
        for i in range(0, len(tensors)):
            graph = tensors[i]
            example = tfgnn.write_example(graph)
            writer.write(example.SerializeToString())

# Filenames for train and validation sets
filename_train = os.path.join(OUT_DIR, 'train.tfrecords')
filename_validate = os.path.join(OUT_DIR, 'val.tfrecords')

print("Writing tensors")
# Save the graph tensors
write_tensors(train_tensors, filename_train)
write_tensors(val_tensors, filename_validate)
print("Tensors written")
# Obtain file paths
# train_path = os.path.join(os.getcwd(), filename_train)
# val_path = os.path.join(os.getcwd(), filename_validate)
train_path = filename_train
val_path = filename_validate

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

print("Loading tensors")
# Load datasets from file
train_ds = tf.data.TFRecordDataset([train_path]).map(decode_fn)
val_ds = tf.data.TFRecordDataset([val_path]).map(decode_fn)

print("Batching tensors")
# Set up batches for training
batch_size = 32
train_ds_batched = train_ds.batch(batch_size=batch_size).repeat()
val_ds_batched = val_ds.batch(batch_size=batch_size)

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
  for i in range(len(activations)):
    graph = tfgnn.keras.layers.GraphUpdate(
        node_sets={
            "players": tfgnn.keras.layers.NodeSetUpdate(
                {"distances": tfgnn.keras.layers.SimpleConv(
                     sender_edge_feature=tfgnn.HIDDEN_STATE,
                     message_fn=dense(message_dim),
                     reduce_type="sum",
                     receiver_tag=tfgnn.TARGET)},
                tfgnn.keras.layers.NextStateFromConcat(dense(dimensions[i], activations[i])))}
    )(graph)

  # After the GNN has computed a context-aware representation of the "atoms",
  # the model reads out a representation for the graph as a whole by averaging
  # (pooling) nde states into the graph context. The context is global to each
  # input graph of the batch, so the first dimension of the result corresponds
  # to the batch dimension of the inputs (same as the labels).
  readout_features = tfgnn.keras.layers.Pool(
      tfgnn.CONTEXT, "mean", node_set_name="players")(graph)

  # Put a linear classifier on top (not followed by dropout).
  logits = tf.keras.layers.Dense(1)(readout_features)

  # Build a Keras Model for the transformation from input_graph to logits.
  return tf.keras.Model(inputs=[input_graph], outputs=[logits])

print("Creating model")
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
print("Training model")
start = time.time()
history = model.fit(train_ds_batched,
                    steps_per_epoch=10,
                    epochs=epochs,
                    validation_data=val_ds_batched)
end = time.time()

modelName = str(opt.number)
model.save(os.path.join(OUT_DIR, modelName))
print("\n*** SAVED MODEL {} ***".format(modelName))

print("\n\n==================== MODEL STATS ====================")
print("     > Model Name:              {}".format(modelName))
print("     > Training Time:           {} seconds".format(round(end - start, 2)))
for k, hist in history.history.items():
    print("     > ", end = "")
    print(k, end =": ")
    print(" " * (23 - len(k)), end = "")
    print(round(hist[-1], 8))
    plt.clf()
    plt.plot(hist)
    plt.title(k)
    plt.savefig(os.path.join(OUT_DIR, modelName + "_" + k + '.pdf'))
print("=====================================================")

script_end = time.time()
print("\n     Total Time Elapsed: {} seconds".format(round(script_end - script_start, 2)))

# For matplotlib graph outputs
# for k, hist in history.history.items():
#   plt.plot(hist)
#   plt.title(k)
#   plt.show()

import tensorflow_gnn as tfgnn
from tensorflow import TensorSpec, float32, int32
from tensorflow.io import TFRecordWriter
import numpy as np
import pandas as pd

from ..labels import LABELS


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
                  'label': TensorSpec(shape=(len(LABELS),), dtype=int32)
    }),
    node_sets_spec={
        'players':
            tfgnn.NodeSetSpec.from_field_specs(
                features_spec={
                    tfgnn.HIDDEN_STATE:
                        TensorSpec((None, 2), float32)
                },
                sizes_spec=TensorSpec((1,), int32))
    },
    edge_sets_spec={
        'distances':
            tfgnn.EdgeSetSpec.from_field_specs(
                features_spec={
                    tfgnn.HIDDEN_STATE:
                        TensorSpec((None, 1), float32)
                },
                sizes_spec=TensorSpec((1,), int32),
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
    label_matrix = np.zeros(len(LABELS))
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
    with TFRecordWriter(filename) as writer:
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
    # print(context_features)
    label = context_features.pop('label')
    # print(context_features)
    new_graph = graph.replace_features(context=context_features)

    return new_graph, label
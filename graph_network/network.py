import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from graph import Graph
import pandas as pd

nodes = [0,1,2,3]
edges = [
    (0, 1, 1.5),
    (0, 2, 1),
    (0, 3, 1),
    (1, 2, 3.5)]

test = Graph(nodes, edges)

node_df = test.get_node_df()
print(node_df)
edge_df = test.get_edge_df()
print(edge_df)

#takes in two data frames for edges and nodes and creates a graph tensor
#unsure of how test train split will be done and how the labels will be handled for the action in the network for training
def create_graph_tensor(node_df. edge_df):
    graph_tensor = tfgnn.GraphTensor.from_pieces(
        node_sets = {
            #our nodes currently have no features attached and are just points to be source and destinations for edges
            "Players": tfgnn.NodeSet.from_fields(
                sizes = [len(node_df)],
                features ={})},
        edge_sets ={
            "Edges": tfgnn.EdgeSet.from_fields(
                sizes = [len(edge_df)],
                #edges have weights attached representing the distance between the two ends
                features = {
                    'weight': np.array(edge_df['weight'],
                                        dtype='float32').reshape(len(edge_df),1)},
                #using the nodes to set source/dest points
                adjacency = tfgnn.Adjacency.from_indices(
                    source = ("Players", np.array(edge_df['source'], dtype='int32')),
                    target = ("Players", np.array(edge_df['destination'], dtype='int32'))))
    })

    return graph_tensor

train_tensor = create(node_df, edge_df)

#not sure how to handle the dataset types yet as I'm still unclear on the actual input, may require a pair session with Joe.

def create_dataset(graph, func):
    #takes the tensor and coverts to dataset tensor
    dataset = tf.data.Dataset.from_tensors(graph)
    #here is where batch splitting and mapping over functions to split target labels from training data

    return dataset.map(func)


train_dataset = create_dataset(train_tensor, label_split)



#The idea of all of this is to create a tf.data.Dataset which is a pairing of the graph_tensor and the asociated training label.
#Given that I'm not sure how the labels will be presented in preprocessing I have left it blank with placeholder function names


'''train_ds = ...  # A tf.data.Dataset of (graph, label) pairs.
model_input_graph_spec, label_spec = train_ds.element_spec
input_graph = tf.keras.layers.Input(type_spec=model_input_graph_spec)''' #from documentation

model_input_graph_spec, label_spec = train_dataset.element_spec
input_graph = tf.keras.layers.Input(type_spec=model_input_graph_spec)

#setting initial values for the hidden state
#currently our nodes have no features attached so an empty tensor is needed and we have one node set
def set_initial_node_state(node_set, node_set_name):
  
    return tfgnn.keras.layers.MakeEmptyFeature()(node_set)
#our edges have their weights as the only feature, just make a reul dense layer for now
def set_initial_edge_state(edge_set, edge_set_name):
    features = [
        tf.keras.layers.Dense(32,activation="relu")(edge_set['weight']),
        
    ]
    return tf.keras.layers.Concatenate()(features)

#takes input_graph which was specified from our training dataset above
#there is an argument to be made for having some context values for the graph as well due to whole graph predictions, number of nodes, mean of distances etc
graph = tfgnn.keras.layers.MapFeatures(
    node_sets_fn=set_initial_node_state,
    edge_sets_fn=set_initial_edge_state)(input_graph)

#if we define our own graph update instead of using a pre given model

graph_updates = 5 #hyper parameter
for x in range(graph_updates):
    graph = tfgnn.keras.layers.GraphUpdate(
        edge_sets = {'weight': tfgnn.keras.layers.EdgeSetUpdate(
            next_state = tfgnn.keras.layers.NextStateFromConcat(
                dense_layer(64,activation='relu')))},
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
    logits = tf.keras.layers.Dense(NUMBER OF DIFFERENT LABELS ,activation='softmax')(LIST OF LABELS[tfgnn.HIDDEN_STATE])

our_model = tf.keras.Model(input_graph, logits)

our_model.compile(
    tf.keras.optimizers.Adam(learning_rate=0.01),
    loss = 'categorical_crossentropy',
    metrics = ['categorical_accuracy']
)
#can play around with different losses.

our_model.summary()




















import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Concatenate, Dense, Flatten, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from .bb_cnn_model import BoundingBoxCNN
from .dir_paths import MODULE_DIR
from .hyperparameters import EPOCHS, NUM_CLASSES
from .labels import LABELS, LabelMapper
from .threed_cnn_model import ThreeDCNN

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow.keras.models import load_model
import pathlib

GNN_MODEL_PATH = os.path.join(pathlib.Path(__file__).parent, "22")

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

def build_model():
    ## The BB_CNN model part
    bb_cnn = BoundingBoxCNN()

    bb_in_size = bb_cnn._cnn_input_size
    bb_num_cnn = bb_cnn._num_cnn
    bb_cnn_ins = [Input(shape=bb_in_size, name=f"input_{i}") for i in range(0, bb_num_cnn)]
    bb_cnn_out = bb_cnn.get_tensors(bb_cnn_ins)

    ## The 3D CNN model part
    threed_cnn = ThreeDCNN()
    threed_cnn_in_shape = threed_cnn.input_shape
    threed_input = Input(shape=threed_cnn_in_shape, name="threedcnn")
    threed_cnn_out = threed_cnn.get_tensors(threed_input)

    ## The GNN model
    gnn_model = load_model(GNN_MODEL_PATH, compile=True)
    gnn_model_inputs = tf.keras.layers.Input(type_spec=graph_tensor_spec)
    for layer in gnn_model.layers:
        gnn_out_shape = layer.output_shape


    ## Concattenate all the outputs and combine them into a final dense layer
    concatted = Concatenate()([bb_cnn_out, threed_cnn_out, gnn_out_shape])
    curr_layer = Flatten()(concatted)

    ## TODO add some more dense layers here?
    for size in [1024, 512, 256, 128, 64, 32]:
        curr_layer = Dense(size, activation="relu")(curr_layer)
        curr_layer = Dropout(0.3)(curr_layer)

    ## The final output layer
    output_layer = Dense(NUM_CLASSES, activation="softmax")(curr_layer)

    ## Collate the inputs together
    model_inputs = [threed_input]
    for input_layer in bb_cnn_ins:
        model_inputs.append(input_layer)
    model_inputs.append([gnn_model_inputs])

    ## Build and compile the model
    model = Model(model_inputs, output_layer)
    model.compile(optimizer=Adam(), loss=CategoricalCrossentropy())

    return model

def train_model(model, train_sequence, val_sequence, checkpoint_dir, save_path, verbose=2):
    file_name = os.path.join(checkpoint_dir, "model_checkpoint.h5")
    checkpoint = ModelCheckpoint(file_name)

    history = model.fit(x=train_sequence, 
              steps_per_epoch = len(train_sequence),
              validation_data=val_sequence,
              validation_steps=len(val_sequence),
              epochs=EPOCHS,
              callbacks=[checkpoint], 
            #   use_multiprocessing=True,
            #   workers=6,
              verbose=verbose)
    
    model.save(save_path)

    for k, hist in history.history.items():
        print("     > ", end = "")
        print(k, end =": ")
        print(" " * (23 - len(k)), end = "")
        print(round(hist[-1], 8))
        plt.clf()
        plt.plot(hist)
        plt.title(k)
        plt.savefig("Model_" + k + '.pdf')

    return model

def test_model(model, test_sequence):
    labels = test_sequence.labels["label"].apply(lambda x : LABELS[x.value-1])
    # preds = [np.argmax(p) for p in model.predict(test_sequence)]
    preds = [LABELS[np.argmax(p)] for p in model.predict(test_sequence)]
    labels = labels[:len(preds)]

    conf_matrix = confusion_matrix(labels, preds, labels=np.unique(labels))
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=np.unique(labels))
    disp.plot()
    plt.savefig(os.path.join(MODULE_DIR, "big_confustion_matrix.png"))

def naiive_train_model(model, generator, checkpoint_dir, save_path, verbose=0):
    file_name = os.path.join(checkpoint_dir, "model_checkpoint")
    checkpoint = ModelCheckpoint(file_name)
    
    model.fit(x=generator, callbacks=[checkpoint], use_multiprocessing=False, verbose=verbose)
    model.save(save_path)

    return model

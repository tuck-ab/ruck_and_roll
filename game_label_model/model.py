import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from .bb_cnn_model import BoundingBoxCNN
from .dir_paths import MODULE_DIR
from .graph_model import graph_tensor_spec
from .hyperparameters import (EPOCHS, JOINT_LAYER_ACTIVATIONS,
                              JOINT_LAYER_DROPOUT_RATE, JOINT_LAYER_SIZES,
                              NUM_CLASSES)
from .labels import LABELS, LabelMapper
from .threed_cnn_model import ThreeDCNN

# GNN_MODEL_PATH = os.path.join(pathlib.Path(__file__).parent, "22")
GNN_MODEL_PATH = "./22"

## CHANGE TO ENSURE NO OVERWRITING
MODEL_NUM = "3"

def build_model(gnn_model_path):
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
    gnn_model = load_model(gnn_model_path, compile=True)
    for layer in gnn_model.layers:
        gnn_out_shape = layer.output_shape
    gnn_in = Input(shape=(gnn_out_shape[1],), name="gnn")


    ## Concattenate all the outputs and combine them into a final dense layer
    concatted = Concatenate()([bb_cnn_out, threed_cnn_out, gnn_in])
    curr_layer = Flatten()(concatted)

    for size, activiation in zip(JOINT_LAYER_SIZES, JOINT_LAYER_ACTIVATIONS):
        curr_layer = Dense(size, activation=activiation)(curr_layer)
        curr_layer = Dropout(JOINT_LAYER_DROPOUT_RATE)(curr_layer)

    ## The final output layer
    output_layer = Dense(NUM_CLASSES, activation="softmax")(curr_layer)

    ## Collate the inputs together
    model_inputs = [threed_input]
    for input_layer in bb_cnn_ins:
        model_inputs.append(input_layer)
    model_inputs.append([gnn_in])

    ## Build and compile the model
    model = Model(model_inputs, output_layer)
    model.compile(optimizer=Adam(), loss=CategoricalCrossentropy())

    return model

def train_model(model, train_sequence, val_sequence, checkpoint_dir, save_path, verbose=2):
    file_name = os.path.join(checkpoint_dir, "model_checkpoint.h5")
    checkpoint = ModelCheckpoint(file_name)

    print("\nData Generated, Now Fitting")
    history = model.fit(x=train_sequence, 
              steps_per_epoch = 10,
              validation_data=val_sequence,
              validation_steps=len(val_sequence),
              epochs=EPOCHS,
              callbacks=[checkpoint], 
            #   use_multiprocessing=True,
            #   workers=6,
              verbose=verbose)
    
    print("Saving model")
    model.save(save_path)

    print("Saving stats")
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
    plt.savefig(os.path.join(MODULE_DIR, "big_confusion_matrix" + MODEL_NUM + ".pdf"))

def naiive_train_model(model, generator, checkpoint_dir, save_path, verbose=0):
    file_name = os.path.join(checkpoint_dir, "model" + MODEL_NUM + "_checkpoint")
    checkpoint = ModelCheckpoint(file_name)
    
    model.fit(x=generator, callbacks=[checkpoint], use_multiprocessing=False, verbose=verbose)
    model.save(save_path)

    return model

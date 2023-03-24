import os

from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Concatenate, Dense, Flatten
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from .bb_cnn_model import BoundingBoxCNN
from .hyperparameters import NUM_CLASSES


def build_model():
    ## The BB_CNN model part
    bb_cnn = BoundingBoxCNN()

    bb_in_size = bb_cnn._cnn_input_size
    bb_num_cnn = bb_cnn._num_cnn
    bb_cnn_ins = [Input(shape=bb_in_size, name=f"input_{i}") for i in range(0, bb_num_cnn)]
    bb_cnn_out = bb_cnn.get_tensors(bb_cnn_ins)



    ## Concattenate all the outputs and combine them into a final dense layer
    concatted = Concatenate()([bb_cnn_out])
    concatted = Flatten()(concatted)
    output_layer = Dense(NUM_CLASSES, activation="softmax")(concatted)

    ## Collate the inputs together
    model_inputs = bb_cnn_ins

    ## Build and compile the model
    model = Model(model_inputs, output_layer)
    model.compile(optimizer=Adam(), loss=CategoricalCrossentropy())

    return model

def train_model(model, train_sequence, val_sequence, checkpoint_dir, save_path, verbose=0):
    file_name = os.path.join(checkpoint_dir, "model_checkpoint")
    checkpoint = ModelCheckpoint(file_name)

    model.fit(x=train_sequence, validation_data=val_sequence,
              callbacks=[checkpoint], use_multiprocessing=False,
              verbose=verbose)
    
    model.save(save_path)
    return model

def naiive_train_model(model, generator, checkpoint_dir, save_path, verbose=0):
    file_name = os.path.join(checkpoint_dir, "model_checkpoint")
    checkpoint = ModelCheckpoint(file_name)
    
    model.fit(x=generator, callbacks=[checkpoint], use_multiprocessing=False, verbose=verbose)
    model.save(save_path)

    return model

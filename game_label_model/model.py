import os

from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Concatenate, Dense, Flatten
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from .bb_cnn_model import BoundingBoxCNN
from .hyperparameters import NUM_CLASSES
from .threed_cnn_model import ThreeDCNN


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


    ## Concattenate all the outputs and combine them into a final dense layer
    concatted = Concatenate()([bb_cnn_out, threed_cnn_out])
    concatted = Flatten()(concatted)

    ## TODO add some more dense layers here?


    ## The final output layer
    output_layer = Dense(NUM_CLASSES, activation="softmax")(concatted)

    ## Collate the inputs together
    model_inputs = [threed_input]
    for input_layer in bb_cnn_ins:
        model_inputs.append(input_layer)

    ## Build and compile the model
    model = Model(model_inputs, output_layer)
    model.compile(optimizer=Adam(), loss=CategoricalCrossentropy())

    return model

def train_model(model, generator, checkpoint_dir, save_path, verbose=0):
    file_name = os.path.join(checkpoint_dir, "model_checkpoint")
    checkpoint = ModelCheckpoint(file_name)
    
    model.fit(x=generator, callbacks=[checkpoint], use_multiprocessing=False, verbose=verbose)
    model.save(save_path)

    return model

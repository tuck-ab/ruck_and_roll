import os
from typing import Tuple

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Concatenate, Conv2D, Dense, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint

from ..dir_paths import MODULE_DIR
from ..hyperparameters import NUM_CNNS, BB_SIZE


class BoundingBoxCNN:
    def __init__(self, num_classes: int,cnn_input_size: Tuple=BB_SIZE,
                 num_cnn: int=NUM_CNNS, pool_window_size: int=4, plot_im: bool=False):
        """Class for the Bounding Box CNN model. It defines the architecture of the
        model as well as providing implimentation for outputting it as an image. It
        can train the model using a `keras.utils.Sequence` object as a generator.

        Args:
            num_classes (int): Number of classes for the classifier to predict
            cnn_input_size (Tuple, optional): The input size of the input CNNs. 
            Defaults to `hyperparameters.BB_SIZE`.
            num_cnn (int, optional): The number of parallel CNNs. 
            Defaults to `hyperparameters.NUM_CNNS`.
            pool_window_size (int, optional): Size of the window for the MaxPooling layer. 
            Defaults to 4.
            plot_im (bool, optional): Whether to save the model architecture as
            an image in `/MODULE_DIR/../images`. Defaults to False.
        """

        self._cnn_input_size = (*cnn_input_size, 3)
        self._num_cnn = num_cnn
        self._pool_window_size = pool_window_size
        self._num_classes = num_classes
        
    def get_tensors(self, ins):
        conv_layers = []
        
        for input_layer in ins:

            conv_layer = Conv2D(2, 3, activation="relu", padding="same", 
                                input_shape=self._cnn_input_size)(input_layer)
            pool_layer = MaxPooling2D(pool_size=self._pool_window_size, 
                                      padding="same")(conv_layer)
            conv_layers.append(pool_layer)

        concatted = Concatenate()(conv_layers)
        concatted = Flatten()(concatted)

        dense = Dense(32)(concatted)

        out = Dense(self._num_classes, activation="softmax")(dense)

        return dense

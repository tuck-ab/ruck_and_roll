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
    def __init__(self, cnn_input_size: Tuple=BB_SIZE,
                 num_cnn: int=NUM_CNNS, pool_window_size: int=4):
        """The Bounding Box CNN. This defines the architecture from the BB_CNN component
        of the model. It returns the connected layers using given inputs.

        Args:
            cnn_input_size (Tuple, optional): The size of the input for each CNN. Defaults to BB_SIZE.
            num_cnn (int, optional): The number of parallel CNNs. Defaults to NUM_CNNS.
            pool_window_size (int, optional): The size of the window in the pooling layer. Defaults to 4.
        """

        self._cnn_input_size = (*cnn_input_size, 3)
        self._num_cnn = num_cnn
        self._pool_window_size = pool_window_size
        
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

        return dense

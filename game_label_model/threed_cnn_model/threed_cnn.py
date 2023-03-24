from typing import Tuple

from tensroflow.keras.layers import Conv3D, Dense, Flatten, MaxPooling3D

from ..hyperparameters import (THREED_CNN_FILTERS, THREED_CNN_INPUT_SHAPE,
                               THREED_CNN_KERNEL_SIZE, THREED_CNN_POOL_SIZE,
                               THREED_CNN_POOL_STRIDES)


class ThreeDCNN:
    def __init__(self, num_filters: int=THREED_CNN_FILTERS,
                 kernel_size: Tuple[int, int, int]=THREED_CNN_KERNEL_SIZE, 
                 input_shape: Tuple[int, int, int, int]=THREED_CNN_INPUT_SHAPE,
                 pool_size: Tuple[int, int, int]=THREED_CNN_POOL_SIZE,
                 pool_strides: Tuple[int, int, int]=THREED_CNN_POOL_STRIDES):

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.input_shape = input_shape

        self.pool_size = pool_size
        self.pool_strides = pool_strides

    def get_tensors(self, input_layer):
        conv = Conv3D(
            self.num_filters,
            self.kernel_size,
            activation="relu",
            input_shape=self.input_shape
            )(input_layer)
        
        pooled = MaxPooling3D(
            pool_size=self.pool_size,
            strides=self.pool_strides
        )(conv)

        flattened = Flatten()(pooled)

        ## Deffo needs to be more layers here
        

        return flattened

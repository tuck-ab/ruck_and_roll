from typing import Tuple

from tensorflow.keras.layers import (BatchNormalization, Conv3D, Dense,
                                     Dropout, Flatten, MaxPooling3D)

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

        batch_normalised = BatchNormalization(center=True, scale=True)(pooled)
        ## dropout = Dropout(0.5)(batch_normalised)
        conv3d = Conv3D(self.num_filters, kernel_size=self.kernel_size, 
                        activation='relu', kernel_initializer='he_uniform')(batch_normalised)
        maxpooled = MaxPooling3D(pool_size=self.pool_size)(conv3d)
        batch_normalised2 = BatchNormalization(center=True, scale=True)(maxpooled)
        ## dropout2 = Dropout(0.5)(batch_normalised2)
        flattened = Flatten()(batch_normalised2)
        dense1 = Dense(256, activation='relu', kernel_initializer='he_uniform')(flattened)
        dropout = Dropout(0.5)(dense1)
        dense2 = Dense(256, activation='relu', kernel_initializer='he_uniform')(dropout)

        return dense2


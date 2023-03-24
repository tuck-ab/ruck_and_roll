from typing import Tuple

from tensorflow.keras.layers import Conv3D, Dense, Flatten, MaxPooling3D, BatchNormalization

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

        ##TODO: Deffo needs to be more layers here
        # model = Sequential()
        # model.add(Conv3D(
        #     32, (3,3,3), activation='relu', input_shape=(25,224,224,3)
        # ))
        # model.add(MaxPooling3D(pool_size=(2,2,2), strides=(1,2,2)))
        # model.add(BatchNormalization(center=True, scale=True))
        # model.add(Dropout(0.5))
        # model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')) ##...
        # model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        # model.add(BatchNormalization(center=True, scale=True))
        # model.add(Dropout(0.5))
        # model.add(Flatten())
        # model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
        # model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
        # model.add(Dense(13, activation='softmax'))
        

        return flattened

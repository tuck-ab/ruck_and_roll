import os
from typing import Tuple

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Concatenate, Conv2D, Dense, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.utils import plot_model

from ..dir_paths import MODULE_DIR
from ..video_handler import VideoHandler
from ..yolo_handler import YOLORunner
from .hyperparameters import BB_CNN_FILTERS


class BoundingBoxCNN:
    def __init__(self, vid_path: str, model_path: str, num_classes: int,
                 cnn_input_size: Tuple=(500, 500, 3), num_cnn: int=10, 
                 pool_window_size: int=4, plot_im: bool=False):
        self._vid_handler = VideoHandler().load_video(vid_path)
        self._yolo_handler = YOLORunner(model_path)

        self._cnn_input_size = cnn_input_size
        self._num_cnn = num_cnn
        self._pool_window_size = pool_window_size
        self._num_classes = num_classes

        self._model = self.get_model(plot_im)

    def run_frame(self):
        frame, valid = self._vid_handler.get_next_frame()

        if not valid:
            print("NOT VALID FRAME IN BoundingBoxCNN.run_frame")
            return
            
        result = self._yolo_handler.run(frame)

        for bb in result.get_store():
            x_start, x_end = bb.x, bb.x+bb.w
            y_start, y_end = bb.y, bb.y+bb.h

            bb_im = frame[y_start: y_end, x_start: x_end, :]

    def train(self):
        pass

    def get_model(self, plot_im):
        conv_layers = []
        inputs = []

        for i in range(0, self._num_cnn):
            input_layer = Input(shape=self._cnn_input_size)
            inputs.append(input_layer)

            conv_layer = Conv2D(2, 3, activation="relu", padding="same", 
                                input_shape=self._cnn_input_size)(input_layer)
            pool_layer = MaxPooling2D(pool_size=self._pool_window_size, 
                                      padding="same")(conv_layer)
            conv_layers.append(pool_layer)

        concatted = Concatenate()(conv_layers)
        concatted = Flatten()(concatted)

        dense = Dense(32)(concatted)

        out = Dense(self._num_classes, activation="softmax")(dense)

        model = Model(inputs, out)

        if plot_im:
            plot_model(model, to_file=os.path.join(MODULE_DIR, os.pardir, 
                                                   "images", "BB_CNN_model.png"))

        return model
        
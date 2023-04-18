import os

import pandas as pd
from tensorflow.keras.models import load_model

from .cli import CommandLineInterface
from .tests import run_tests
from .u1903266 import run as u1903266_run
from .data_gen import IntermediateDataGenerator
from .sequence import get_train_test_val
from .model import build_model, train_model, test_model

## TODO change these to command line argurments
YOLO_MODEL = "yolov7_480x640.onnx"
VIDEO = "220611galleivnor_2_movie-001.mov"
LABEL_FILE = "gallconcat.lbl"
NEED_TO_GEN = False
## Intermediate file path thing
GRAPH_PATH = "~/ruck_and_roll/game_label_model/graph_model/graph.tfrecords"

def main(video_dir, yolo_model_dir, temp_dir, label_dir):
    vid_path = os.path.join(video_dir, VIDEO)
    yolo_path = os.path.join(yolo_model_dir, YOLO_MODEL)
    labels_path = os.path.join(label_dir, LABEL_FILE)

    int_data_dir = os.path.join(temp_dir, "first_model_int")
    if NEED_TO_GEN:
        data_gen = IntermediateDataGenerator(vid_path, yolo_path)

        if not os.path.isdir(int_data_dir):
            os.mkdir(int_data_dir)

        data_gen.generate(int_data_dir)

    train_seq, val_seq, test_seq = get_train_test_val(vid_path, yolo_path, GRAPH_PATH, int_data_dir, labels_path)

    model = build_model()

    # raise KeyboardInterrupt

    checkpoint_dir = os.path.join(temp_dir, "first_big_model_checkpoint")
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    save_path = os.path.join(temp_dir, "first_big_model.h5")

    model = train_model(model, train_seq, val_seq, checkpoint_dir, save_path, verbose=1)

    model = load_model(os.path.join(temp_dir, "first_big_model.h5"))

    test_model(model, test_seq)

if __name__ == "__main__":
    cli = CommandLineInterface()
    cli.parse()
    
    if cli.get_test_flag():
        run_tests()

    video_dir = cli.get_vid_dir()

    yolo_model_dir = cli.get_yolo_model_dir()

    temp_dir = cli.get_temp_dir()

    label_dir = cli.get_label_dir()
    
    what_to_run = cli.get_what_to_run()

    if what_to_run == "u1903266":
        u1903266_run(video_dir, yolo_model_dir, temp_dir, label_dir)
    elif what_to_run == "main":
        main(video_dir, yolo_model_dir, temp_dir, label_dir)
    else:
        main(video_dir, yolo_model_dir, temp_dir, label_dir)

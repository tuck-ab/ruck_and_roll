import os
import pathlib

MODULE_DIR = pathlib.Path(__file__).parent

VIDEO_DIR = os.path.join(MODULE_DIR, os.pardir, "videos")
YOLO_MODEL_DIR = os.path.join(MODULE_DIR, os.pardir, "yolo_models")
LABEL_DIR = os.path.join(MODULE_DIR, os.pardir, "labels")
TEMP_DIR = os.path.join(MODULE_DIR, os.pardir, "temp")
GRAPH_MODEL_DIR = os.path.join(MODULE_DIR, os.pardir, "graph_models")

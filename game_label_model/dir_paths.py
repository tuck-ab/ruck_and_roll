import os
import pathlib

MODULE_DIR = pathlib.Path(__file__).parent

VIDEO_DIR = os.path.join(MODULE_DIR, os.pardir, "videos")
YOLO_MODEL_DIR = os.path.join(MODULE_DIR, os.pardir, "yolo_models")

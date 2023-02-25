import numpy as np
import math
import argparse
import pathlib
import os
import cv2
import time

from ...video_handler.videohandler import VideoHandler
from ...yolo_handler.yolo_frame import YOLORunner
from .graph_gen import GraphGenerator

# Run with: python -m game_label_model.graph_model.generation.graph_save_load_script --file "RugbyVideoVShort.mp4" --model "accurate"

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='', help='File name of video')
parser.add_argument('--model', type=str, default='', help='Type of yolo model to use')
opt = parser.parse_args()

FILE_DIR = pathlib.Path(__file__).parent
OUT_DIR = os.path.join(FILE_DIR, "outputs")
IN_DIR = os.path.join(FILE_DIR, "inputs")
INFILE = os.path.join(IN_DIR, opt.file)

yolo_name_fast = "yolov7-tiny_Nx3x480x640.onnx"
yolo_name_accurate = "yolov7_Nx3x480x640.onnx"
yolo_name_experimental = "yolov7_Nx3x736x1280.onnx"

MODEL_DIR = pathlib.Path(__file__).parent.parent.parent.parent
MODEL_DIR = os.path.join(MODEL_DIR, "yolov7")
MODEL_DIR = os.path.join(MODEL_DIR, "models")

if opt.model == "fast":
    MODEL_FILE = os.path.join(MODEL_DIR, yolo_name_fast)
elif opt.model == "accurate":
    MODEL_FILE = os.path.join(MODEL_DIR, yolo_name_accurate)
elif opt.model == "experimental":
    MODEL_FILE = os.path.join(MODEL_DIR, yolo_name_experimental)
else:
    err = "Model options are '{}', '{}' or '{}. '{}' is an invalid option".format("fast", "accurate", "experimental", opt.model)
    raise FileExistsError(err)

SAVE_DIR = os.path.join(OUT_DIR, opt.file[:-4])
NODES_DIR = os.path.join(SAVE_DIR, "nodes")
EDGES_DIR = os.path.join(SAVE_DIR, "edges")
try:
    os.mkdir(SAVE_DIR)
    os.mkdir(NODES_DIR)
    os.mkdir(EDGES_DIR)
except:
    err = "Unable to create folder, folder name {} may alrady exist (File path: {})".format(opt.file[:-4], SAVE_DIR)
    raise FileExistsError(err)

vh = VideoHandler()
vh.load_video(INFILE)

yh = YOLORunner(MODEL_FILE)

frame = vh.get_next_frame()
count = 0
fps = vh.fps

start = time.time()
while frame[1]:
    if count % 100 == 0:
        print("Now on frame {}. At {} fps this is {} seconds into the video".format(count, fps, round(count / fps, 2)))

    bbstore = yh.run(frame[0])
    graph_gen = GraphGenerator(bbstore)
    m, nodes = graph_gen.get_graph().get_edge_matrix()

    save_file = str(count) + ".txt"
    EDGES_SAVE_FILE = os.path.join(EDGES_DIR, save_file)
    NODES_SAVE_FILE = os.path.join(NODES_DIR, save_file)

    np.savetxt(EDGES_SAVE_FILE, m)
    np.savetxt(NODES_SAVE_FILE, nodes, fmt='%s')

    frame = vh.get_next_frame()
    count += 1
end = time.time()

print("\n\n\n\n==================== FINISHED ====================\n")
print("     > Total Time Elapsed: {} seconds".format(round(end - start, 2)))
print("     > Video Length:       {} seconds".format(round(count / fps, 2)))
print("     > Ratio:              {}x realtime".format(round((end - start) / (count / fps), 2)))
print("\n==================== FINISHED ====================\n\n\n\n")
print("Files found at: {}".format(SAVE_DIR))




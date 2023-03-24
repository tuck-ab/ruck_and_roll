import os
import json
import time

from .bb_cnn_model import BoundingBoxDataGatherer, BoundingBoxCNN, YOLOSequence
from .dir_paths import MODULE_DIR
from .video_handler import VideoHandler, YOLOVideoWriter
from .yolo_handler import YOLORunner
from .model import build_model, train_model

VIDS = [
    "220611galleivnor_2_movie-001.mov",
    "cambridge-v-cinderford-bip.mp4",
    "221015ramsvcambridge-bip.mp4",
    "20220917cambridgevtauntonnational-1-bip.mp4",
    "2022101cambridgevsale-fc-bip.mp4",
    "20220910bishops-stortfordvcambridgenational-1-bi.mp4",
    "20220924dmpvcambridge.mp4",
    "20220903cambridgevplymouthnational-1-bip.mp4"
]

def run(video_dir, yolo_model_dir, temp_dir, label_dir):
    words = ["chair"]
    for word in words:
        print(f"Running for {word}")
        show_yolo_from_word(
            os.path.join(video_dir, "220611galleivnor_2_movie-001.mov"),
            os.path.join(yolo_model_dir, "yolov7_480x640.onnx"),
            word
        )

def generate_data_somewhat(video_dir, yolo_model_dir, temp_dir, labels_dir):
    start = time.time()

    mod = "yolov7_480x640.onnx"
    lab = "gallconcat.lbl"
    vid = "220611galleivnor_2_movie-001.mov"

    gen = YOLOSequence(labels_dir, lab, video_dir, vid, yolo_model_dir, mod, 
                        temp_dir, generate_data=False)

    model = build_model()
    checkpoint_dir = os.path.join(temp_dir, "checkpoints")
    save_path = os.path.join("/dcs/large/u1903266/models/secondmodel")
    model = train_model(model, gen, checkpoint_dir, save_path)

    end = time.time()

    print(f"Runtime: {end-start}")

def get_bb_data(video_dir, yolo_model_dir):
    for vid in ["nootnoot.mp4"]:
        print(f"Runing on video: {vid}")
        bb_cnn = BoundingBoxDataGatherer(
            os.path.join(video_dir, vid),
            os.path.join(yolo_model_dir, "yolov7_480x640.onnx")
        )

        to_save = os.path.join(MODULE_DIR, "bb_cnn_model", "bb_data_new", f"{vid.split('.')[0]}.json")
        bb_cnn.get_bbs_data_per_frame(save_to_file=to_save)

def get_bb_counts(video_dir, yolo_model_dir):
    for vid in VIDS:
        bb_cnn = BoundingBoxDataGatherer(
            os.path.join(video_dir, vid),
            os.path.join(yolo_model_dir, "yolov7-tiny_256x320.onnx")
        )

        to_save = os.path.join(MODULE_DIR, "bb_cnn_model", "bb_nums", f"{vid.split('.')[0]}.txt")
        bb_cnn.get_num_bbs_per_frame(save_to_file=to_save)

def load_json_data(f_name):
    json_dir = os.path.join(MODULE_DIR, "bb_cnn_model", "bb_data_new")
    with open(os.path.join(json_dir, f_name), "r") as f:
        data = json.load(f)
    return data

def show_yolo_from_word(vid_path, yolo_path, word):
    data = load_json_data("220611galleivnor_2_movie-001.json")
    frame_nums = []

    MAX_LEN = 100

    for i, frame in enumerate(data):
        for bb in frame["bbs"]:
            if word == bb["class"]:
                frame_nums.append(i)

    vh = VideoHandler().load_video(vid_path)
    vw = YOLOVideoWriter(
        os.path.join(MODULE_DIR, "..", "videos", f"lei_v_nor_{word}.mp4"),
        vh.fps,
        vh.width,
        vh.height
    )
    ym = YOLORunner(yolo_path)
    for num in frame_nums[:MAX_LEN]:
        frame = vh.get_frame(num, show=False)
        combined = ym.get_detections(frame)
        vw.add_frame(combined)


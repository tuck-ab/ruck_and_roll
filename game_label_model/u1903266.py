import os

from .bb_cnn_model import BoundingBoxCNN
from .dir_paths import MODULE_DIR

def run(video_dir, yolo_model_dir):
    print("Running u1903266 `run`")


def get_bb_counts(video_dir, yolo_model_dir):
    vids = [
        "cambridge-v-cinderford-bip.mp4",
        "221015ramsvcambridge-bip.mp4",
        "20220917cambridgevtauntonnational-1-bip.mp4",
        "2022101cambridgevsale-fc-bip.mp4",
        "20220910bishops-stortfordvcambridgenational-1-bi.mp4",
        "20220924dmpvcambridge.mp4",
        "20220903cambridgevplymouthnational-1-bip.mp4",
        "220611galleivnor_2_movie-001.mov"
    ]
    
    for vid in vids:
        bb_cnn = BoundingBoxCNN(
            os.path.join(video_dir, vid),
            os.path.join(yolo_model_dir, "yolov7-tiny_256x320.onnx")
        )

        to_save = os.path.join(MODULE_DIR, "bb_cnn_model", "bb_nums", f"{vid.split('.')[0]}.txt")
        bb_cnn.get_num_bbs_per_frame(save_to_file=to_save)

import os

from .bb_cnn_model import BoundingBoxCNN
from .dir_paths import MODULE_DIR

def run(video_dir, yolo_model_dir):
    vids = [
        "cambridge-v-cinderford-bip.mp4",
        "221015ramsvcambridge-bip.mp4",
        "20220917cambridgevtauntonnational-1-bip.mp4",
        "2022101cambridgevsale-fc-bip.mp4",
        "20220910bishops-stortfordvcambridgenational-1-bi.mp4",
        "20220924dmpvcambridge.mp4",
        "20220903cambridgevplymouthnational-1-bip.mp4"
    ]
    
    # for vid in vids:
    #     bb_cnn = BoundingBoxCNN(
    #         os.path.join(video_dir, vid),
    #         ## os.path.join(video_dir, "nootnoot.mp4"),
    #         os.path.join(yolo_model_dir, "yolov7-tiny_256x320.onnx")
    #     )
        
        

        # bb_cnn.get_num_bbs_per_frame(save_to_file=os.path.join(MODULE_DIR, "bb_cnn_model", "bb_nums", f"{vid.split('.')[0]}.txt"))
    # vid = "220611galleivnor_2_movie-001.mov"

    # bb_cnn = BoundingBoxCNN(
    #     os.path.join(video_dir, vid),
    #     os.path.join(yolo_model_dir, "yolov7-tiny_256x320.onnx")
    # )

    # to_save = os.path.join(MODULE_DIR, "bb_cnn_model", "bb_nums", f"{vid.split('.')[0]}.txt")

    # bb_cnn.get_num_bbs_per_frame(save_to_file=to_save)

    print("IT WORKSS LOL")

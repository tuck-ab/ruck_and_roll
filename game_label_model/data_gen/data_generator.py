import os

import numpy as np

from ..video_handler import VideoHandler, DataGenWriter
from ..yolo_handler import YOLORunner
from ..labels import load_from_file as load_labels_from_file
from ..hyperparameters import VALID_YOLO_LABELS, NUM_CNNS, BB_SIZE

class IntermediateDataGenerator:
    def __init__(self, vid_path, yolo_path):
        self._video_handler = VideoHandler().load_video(vid_path)
        self._yolo_handler = YOLORunner(yolo_path)

    def generate(self, out_dir, start=0, size=None):

        frame, valid = self._video_handler.set_frame(start)

        if size is None:
            size = np.inf

        while valid and self._video_handler.current_frame_num + start < size:
            result = self._yolo_handler.run(frame)

            bb_ims = []

            for bb in result.get_store():
                if bb.class_name in VALID_YOLO_LABELS:
                    x_start, x_end = bb.x, bb.x+bb.w
                    y_start, y_end = bb.y, bb.y+bb.h

                    bb_im = frame[y_start: y_end, x_start: x_end, :]
                    bb_ims.append(bb_im)

            ## Choose which bounding boxes to use
            ## TODO Make this a better choice
            bb_ims = bb_ims[:NUM_CNNS]

            temp = [pad_image(im, BB_SIZE) for im in bb_ims]
            bb_ims = temp

            while len(bb_ims) < 10:
                bb_ims.append(np.zeros((*BB_SIZE, 3)))

            bb_ims = np.array(bb_ims)        

            ## Save the bb_ims_to_size
            f_name = f"yolo-{self._video_handler.current_frame_num}.npy"
            np.save(os.path.join(out_dir, f_name), bb_ims)

            frame, valid = self._video_handler.get_next_frame()

def pad_image(im, target_size):
    new_im = np.zeros((*target_size, 3)).astype(np.uint8)
    new_im[:im.shape[0], :im.shape[1]] = im[:new_im.shape[0], :new_im.shape[1]]

    return new_im



# def old_generation(self, clip_size, out_dir, verbose=False):
#     global_frame_counter = 0

#     id_label_pairs = []

#     for label, count in self._labels:
#         if verbose:
#             print(f"Processing a run of {count} for label {label}")
        
#         local_frame_count = 0

#         while local_frame_count + clip_size < count:

#             frames = self._video_handler.get_clip(global_frame_counter + local_frame_count,
#                                                     clip_size)

#             clip_id = global_frame_counter + local_frame_count
#             writer = DataGenWriter(
#                 os.path.join(out_dir, f"{clip_id}.mp4"),
#                 self._video_handler.fps,
#                 self._video_handler.width,
#                 self._video_handler.height
#             )

#             for frame in frames:
#                 writer.add_frame(frame)

#             writer.done()

#             id_label_pairs.append((clip_id, label))

#             local_frame_count += clip_size


#         global_frame_counter += count

#     ## TODO Save the id_label_pairs
#     df = pd.DataFrame(id_label_pairs)
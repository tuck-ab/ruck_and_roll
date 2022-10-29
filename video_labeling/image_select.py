import os
import pathlib
import enum

import cv2

## -- Defining useful directory paths
FILE_DIR = pathlib.Path(__file__).parent
VIDEO_DIR = os.path.join(FILE_DIR, "..", "videos")
FRAME_DIR = os.path.join(FILE_DIR, "..", "images")

## -- Labels for the game
LABELS = ["NOTHING", "FORCEFUL_TACKLE", "ABSORBING_TACKLE",
          "OTHER_TACKLE", "RUCK", "MAUL", "LINEOUT", "SCRUM"]

Label = enum.Enum("Label", LABELS)

class Video_Label:
    def __init__(self):
        self.runs = []
        
        self._previous_label = Label.NOTHING
        self._current_run_len = 0
        
    def add_label(self, label):
        ## If the label is in the same run
        if self._previous_label == label:
            ## Increase run length
            self._current_run_len += 1
        else:
            ## If its different then add the completed run to the list
            self.runs.append((self._previous_label, self._current_run_len))
            
            ## Start a new run
            self._previous_label = label
            self._current_run_len = 1
        
    def end_label(self):
        self.runs.append((self._previous_label, self._current_run_len))
        
    def save_labels(self, file_name, path="."):
        with open(os.path.join(path, file_name), "wt") as f:
            f.write("".join([f"{x[0]}:{x[1]}\n" for x in self.runs]))

def save_video_frames(file_name, vid_path=VIDEO_DIR, frame_path=FRAME_DIR):
    """Saves every frame in a video as a .jpg in a given directory

    Args:
        file_name (str): The name of the video to save frames from
        vid_path (str / os.path, optional): Directory where the video is saved. 
        Defaults to VIDEO_DIR.
        frame_path (str / os.path, optional): Directory where the frame images
        are saved. Defaults to FRAME_DIR.
    """
    
    cam = cv2.VideoCapture(os.path.join(vid_path, file_name))
    
    ## Check if save directory exists and make one if not
    try:
        if not os.path.exists(frame_path):
            os.makedirs(frame_path)
    except OSError:
        print(f"Could not find directory to store frames\nCreating directory: {frame_path}")

    current_frame = 0

    try:
        while True:
            ## Read the image frame
            ret, frame = cam.read()
            
            if ret:
                frame_image_path = os.path.join(frame_path, f"frame{current_frame}.jpg")
                print(f"Creating: {frame_image_path}")
    
                ## Save the frame
                cv2.imwrite(frame_image_path, frame)

                current_frame += 1
            else:
                break
    except KeyboardInterrupt:
        pass 
    finally:
        ## Close and shut down objects properly
        cam.release()
        cv2.destroyAllWindows()
        
def play_video(file_name, vid_path=VIDEO_DIR):
    """Plays a video frame by frame. Allows the labeling of each frame.

    Args:
        file_name (_type_): The name of the video to play
        vid_path (_type_, optional): Directory where the video is saved.
        Defaults to VIDEO_DIR.
    """
    
    video_label = Video_Label()
    
    cam = cv2.VideoCapture(os.path.join(vid_path, file_name))
    
    frame = cam.get(cv2.CAP_PROP_FRAME_COUNT)
    
    cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)
    
    frame_number = 0
    
    while cam.isOpened():
        ret, frame = cam.read()
        
        ## Write labels to help user
        ## Frame number
        cv2.putText(frame, 
                    f"Frame: {frame_number}", 
                    (20, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 0), 2, 1)
        
        ## What key to press for label number
        for i, label in enumerate(LABELS):
            cv2.putText(frame, 
                        f"{i} - {label}", 
                        (20, 20 + (i+1) * 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 0), 2, 1)
        
        cv2.imshow("video", frame)
        key = cv2.waitKey(1000)
        
        ## Valid inputs when viewing a frame
        VALID_KEYS = {ord('0'), ord('1'), ord('2'), ord('3'), ord('4'),
                      ord('5'), ord('6'), ord('7'), ord('q'), ord('n')}
        
        while not key in VALID_KEYS:
            key = cv2.waitKey(1000)
            if not cv2.getWindowProperty('video', cv2.WND_PROP_VISIBLE) >= 1:
                key = ord('q')
            
        ## Decode the input from user inputs
        if key == ord('q'):
            break
        elif key == ord('0'):
            video_label.add_label(Label.NOTHING)
        elif key == ord('1'):
            video_label.add_label(Label.FORCEFUL_TACKLE)
        elif key == ord('2'):
            video_label.add_label(Label.ABSORBING_TACKLE)
        elif key == ord('3'):
            video_label.add_label(Label.OTHER_TACKLE)
        elif key == ord('4'):
            video_label.add_label(Label.RUCK)
        elif key == ord('5'):
            video_label.add_label(Label.MAUL)
        elif key == ord('6'):
            video_label.add_label(Label.LINEOUT)
        elif key == ord('7'):
            video_label.add_label(Label.SCRUM)
        
        frame_number += 1
        
    video_label.end_label()
    video_label.save_labels(f"{file_name}.lbl")
        
    cam.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    play_video("20220924dmpvcambridge.mp4")
    
        
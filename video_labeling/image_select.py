import os
import pathlib
import enum
import argparse

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
    """Object that stores runs of labels for videos. It can
    process adding labels frame by frame automatically calculating runs.
    It can also write the encoded labels as text to a file.
    """
    def __init__(self):
        self.runs = []
        
        self._previous_label = Label.NOTHING
        self._current_run_len = 0
        
    def add_label(self, label):
        """Add the label for the next frame

        Args:
            label (Label): The label of the frame
        """
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
            
    def undo_label(self):
        """Undo the label assigned to the most recent frame
        """
        ## If its the first label in a run
        if self._current_run_len <= 1:
            if len(self.runs) > 0:
                ## Reload the previous run
                self._previous_label, self._current_run_len = self.runs.pop()
            else: ## If its the start of the video then restart all the runs
                self._previous_label = Label.NOTHING
                self._current_run_len = 0
        else: ## Otherwise decrease size of the run
            self._current_run_len -= 1
        
    def end_label(self):
        """Ends the final run. Done before writing runs to file at the end of
        a video
        """
        self.runs.append((self._previous_label, self._current_run_len))
        
    def save_labels(self, file_name, path="."):
        """Save the runs to a text file

        Args:
            file_name (str): Name of the file to save the runs to
            path (str / os.path, optional): Directory to save the file to. Defaults to ".".
        """
        with open(os.path.join(path, file_name), "wt") as f:
            f.write("".join([f"{x[0]}:{x[1]}\n" for x in self.runs]))

class Frame_Replayer:
    """Stores the frames in a video allowing the replay of them
    """
    def __init__(self, first_frame):
        self._data = [first_frame]
        self._index = 0
        
        self.current_frame = first_frame
        
    def push(self, frame):
        """Add new frame from video to the data structure

        Args:
            frame (frame): Frame to add

        Returns:
            frame: Current frame being played
        """
        self._data.append(frame)
        
        return self.play_next_frame()
        
    def replay_previous_frame(self):
        """Rewind video to show previous frame

        Returns:
            frame: Current frame being played
        """
        self._index = max(0, self._index - 1)
        
        self.current_frame = self._data[self._index]
        
        return self.current_frame
    
    def play_next_frame(self):
        """If video has been rewound then plays the next frame stored
        in the data structure. If at most recent frame in data structure
        then returns `None`.

        Returns:
            frame: `None` if frame is most recent, else current frame being
            played
        """
        self._index += 1
        if self._index >= len(self._data):
            self._index -= 1
            return None
        
        self.current_frame = self._data[self._index]
        
        return self.current_frame  
    
    def get_frame_number(self):
        """Returns the frame number of the current frame being played

        Returns:
            int: Frame number of current frame being played
        """
        return self._index      
        

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
        
def annotate_frame(frame, frame_no):
    """Annotates frames with keyboard inputs to help user.

    Args:
        frame (frame): The frame to annotate
        frame_no (int): The number of the frame in the video

    Returns:
        frame: The annotated frame
    """

    cv2.putText(frame, 
                f"Frame: {frame_no}, Quit: q, Previous frame: b", 
                (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 0, 0), 2, 1)
    
    ## What key to press for label number
    for i, label in enumerate(LABELS):
        cv2.putText(frame, 
                    f"{i} - {label}", 
                    (20, (i+2) * 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 0), 2, 1)
        
    return frame
        
def play_video(file_name, vid_path=VIDEO_DIR, out_dir=None):
    """Plays a video frame by frame. Allows the labeling of each frame.

    Args:
        file_name (str): The name of the video to play
        vid_path (str / os.path, optional): Directory where the video is saved.
        Defaults to VIDEO_DIR.
    """
    
    video_label = Video_Label()
    
    cam = cv2.VideoCapture(os.path.join(vid_path, file_name))
    
    # frame = cam.get(cv2.CAP_PROP_FRAME_COUNT)
    
    cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)
    
    ret, frame = cam.read()
    
    if not ret:
        print(f"Video is empty...")
        return
    
    frame = annotate_frame(frame, 0)
    frame_replayer = Frame_Replayer(frame)
    
    while cam.isOpened():        
       
        cv2.imshow("video", frame_replayer.current_frame)
        key = cv2.waitKey(1000)
        
        ## Valid inputs when viewing a frame
        ## -- 81 is left arrow (<-)
        VALID_KEYS = {ord('0'), ord('1'), ord('2'), ord('3'), ord('4'),
                      ord('5'), ord('6'), ord('7'), ord('q'), ord('b')}
        
        while not key in VALID_KEYS:
            key = cv2.waitKey(1000)
            if not cv2.getWindowProperty('video', cv2.WND_PROP_VISIBLE) >= 1:
                key = ord('q')
            
        get_next_frame = True
            
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
        elif key == ord('b'): ## Left arrow
            get_next_frame = False
            video_label.undo_label()
            
        if get_next_frame:
            frame = frame_replayer.play_next_frame()
            if frame is None:
                ret, frame = cam.read()
                frame = annotate_frame(frame, frame_replayer.get_frame_number() + 1)
                frame_replayer.push(frame)
        else:
            frame_replayer.replay_previous_frame()

        
    video_label.end_label()
    
    if out_dir is None:
        out_dir = f"{file_name}.lbl"
        
    video_label.save_labels(out_dir)
        
    cam.release()
    cv2.destroyAllWindows()
    
parser = argparse.ArgumentParser()

parser.add_argument("-f", "--File", help="Input video file")
parser.add_argument("-o", "--Output", help="Output file name")

args = parser.parse_args()
    
if __name__ == "__main__":
    if args.File is None:
        print("No file given. Use -f or --File to specify video")
        exit(0)
        
    if not os.path.exists(os.path.join(VIDEO_DIR, args.File)):
        print("File doesn't exist")
        exit(1)
        
    play_video(args.File, out_dir=args.Output)
    
        
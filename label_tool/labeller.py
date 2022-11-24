import cv2

from .labels import LABELS, Label
from .frame_buffer import FrameBuffer
from .rle import LabelTracker

EXIT_KEY = ord('q')  

## Set of all valid keys that would move a frame forward
VALID_KEYS = {
    ord('0'),
    ord('1'),
    ord('2'),
    ord('3'),
    ord('4'),
    ord('5'),
    ord('6'),
    ord('7'),
    ord('8'),
    ord('9'),
    ord('u'),
    ord('m'),
    ord('s'),
    ord('l'),
    ord('r'),
    ord('b'),
    EXIT_KEY
}

## Map from input key to label
ACTION_MAP = {
    ord('0'): Label.NOTHING,
    ord('1'): Label.CARRY,
    ord('2'): Label.PASS_L,
    ord('3'): Label.PASS_R,
    ord('4'): Label.KICK_L,
    ord('5'): Label.KICK_R,
    ord('6'): Label.TACKLE_S_D,
    ord('7'): Label.TACKLE_S,
    ord('8'): Label.TACKLE_D_D,
    ord('9'): Label.TACKLE_D,
    ord('u'): Label.TACKLE_M,
    ord('m'): Label.MAUL,
    ord('s'): Label.SCRUM,
    ord('l'): Label.LINEOUT,
    ord('r'): Label.RUCK,
}

class ErrorPlayingVideo(Exception):
    pass

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
    for i, key in enumerate(ACTION_MAP):
        cv2.putText(frame, 
                    f"{chr(key)} - {ACTION_MAP[key]}", 
                    (20, 100 + (i) * 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 0, 0), 1, 1)
        
    return frame

class Labeller:
    """Object that deals with the interaction between the tool and
    the user. Loads and plays the video using a `LabelTracker` to
    store the labels and a `FrameBuffer` to allow partial rewinding.
    """
    def __init__(self, video_path):
        ## Object to access frames in video
        self._cap = cv2.VideoCapture(video_path)
        
        ## Name of the window
        self._window_name = "Video"
        
    def run(self, start_frame=None):
        """Plays the video and tracks the user labels.

        Args:
            start_frame (int, optional): What frame the video should start 
            if the video should start part of the way through. Defaults to None.

        Raises:
            ErrorPlayingVideo: If there was an error playing the video

        Returns:
            LabelTracker: The label tracker containing the labels given by
            the user.
        """
        cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)
        
        if start_frame:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
        else:
            start_frame = 0
        
        ret, frame = self._cap.read()
        
        if not ret:
            raise ErrorPlayingVideo("Did not return valid frame")
        
        frame_buffer = FrameBuffer(annotate_frame(frame, start_frame))
        label_tracker = LabelTracker()
        
        while ret:
            cv2.imshow(self._window_name, frame_buffer.current_frame)
            
            ## Key input detection in active loop
            key = cv2.waitKey(1000)
            while not key in VALID_KEYS:
                key = cv2.waitKey(1000)
                
                ## Check if window was closed
                if not cv2.getWindowProperty(self._window_name, cv2.WND_PROP_VISIBLE) >= 1:
                    key = EXIT_KEY
                    
            frame_forward = True        
            
            ## Parse the key
            if key == EXIT_KEY:
                break
            elif key == ord('b'):
                frame_forward = False
            elif key in ACTION_MAP:
                label_tracker.add_label(ACTION_MAP[key])            
            
            if frame_forward:
                if frame_buffer.get_next_frame() is None:
                    ret, frame = self._cap.read()
                    
                    if not ret:
                        break
                    
                    frame = annotate_frame(frame, frame_buffer.get_frame_number())
                    frame_buffer.push_frame(frame)
            else:
                _, has_replayed = frame_buffer.get_previous_frame()
                
                if has_replayed:
                    label_tracker.undo_label()
    
        cv2.destroyAllWindows()
        
        return label_tracker
        
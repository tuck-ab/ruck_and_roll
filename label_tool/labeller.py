import cv2

from .labels import LABELS, Label
from .frame_buffer import FrameBuffer
from .rle import LabelTracker

EXIT_KEY = ord('q')  

VALID_KEYS = {
    ord('0'),
    ord('1'),
    ord('2'),
    ord('3'),
    ord('4'),
    ord('5'),
    ord('6'),
    ord('7'),
    ord('b'),
    EXIT_KEY
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
    for i, label in enumerate(LABELS):
        cv2.putText(frame, 
                    f"{i} - {label}", 
                    (20, (i+2) * 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 0), 2, 1)
        
    return frame

class Labeller:
    def __init__(self, video_path):
        ## Object to access frames in video
        self._cap = cv2.VideoCapture(video_path)
        
        ## Name of the window
        self._window_name = "Video"
        
    def run(self, start_frame=None):
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
            elif key == ord('0'):
                label_tracker.add_label(Label.NOTHING)
            elif key == ord('1'):
                label_tracker.add_label(Label.FORCEFUL_TACKLE)
            elif key == ord('2'):
                label_tracker.add_label(Label.ABSORBING_TACKLE)
            elif key == ord('3'):
                label_tracker.add_label(Label.OTHER_TACKLE)
            elif key == ord('4'):
                label_tracker.add_label(Label.RUCK)
            elif key == ord('5'):
                label_tracker.add_label(Label.MAUL)
            elif key == ord('6'):
                label_tracker.add_label(Label.LINEOUT)
            elif key == ord('7'):
                label_tracker.add_label(Label.SCRUM)
            elif key == ord('b'):
                frame_forward = False
            
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
        
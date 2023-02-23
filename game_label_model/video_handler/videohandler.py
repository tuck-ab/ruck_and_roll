from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .custom_exceptions import ErrorPlayingVideoException, BufferTooSmallException

class VideoHandler:
    """Class for wrapping accessing videos for the rest
    of the module to interact with.
    """
    
    def __init__(self, buffer_size: int = 1):
        self._buffer = []
        self._buffer_index = -1
        if buffer_size < 1:
            raise BufferTooSmallException(
                "Buffer must be of minimum size 1")
        else:
            self._buffer_max_size = buffer_size
    
    def load_video(self, vid_path: str, start_frame: Optional[int] = None) -> VideoHandler:
        """Loads the video from a given path to interact with.

        Args:
            vid_path (str): Path of the video to load
            start_frame (Optional[int], optional): What frame to start the video at. 
            If None then starts at the begining. Defaults to None.

        Returns:
            VideoHandler: The instance of the object so can be used with the constructor eg. `VideoHandler().load_video()`
        """
        self._cap = cv2.VideoCapture(vid_path)
        
        if not self._cap.isOpened():
            raise ErrorPlayingVideoException(f"Video file not found: {vid_path}")
        
        if start_frame:
            self._start_frame_num = start_frame
            self.current_frame_num = start_frame - 1
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        else:
            self._start_frame_num = 0
            self.current_frame_num = -1
            
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) #Get framerate to match for output rate
        self.width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width` converted to int
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height` converted to int
            
        return self
    
    def get_next_frame(self) -> Tuple[np.ndarray, bool]:
        """Returns the next frame in the video. If the bool is true then it is a new frame, if not
        it is the same frame as the previous call and means the video has ended. This will still return
        True if the frame has been played before but the video has been rewound. Will only return False
        at the end of the video.

        Raises:
            ErrorPlayingVideoException: Is raised if the first frame in the video fails to load

        Returns:
            Tuple[np.ndarray, bool]: A tuple containing the next frame, and a bool saying whether this
            is a new frame. If False then the frame was the same as last call and the video is over.
        """
        
        ## If at the end of the buffer
        if self._buffer_index == len(self._buffer) - 1:
            ret, frame = self._cap.read()
            self.current_frame_num += 1

            if not ret:
                if self.current_frame_num == self._start_frame_num:
                    raise ErrorPlayingVideoException(
                        "Error in VideoHandler.get_next_frame, did not return valid frame from video")
                else: ## At the end of the video
                    return self.current_frame, False
                
            self.current_frame = frame
            self._buffer.append(self.current_frame)
            self._buffer_index += 1
            
            if len(self._buffer) > self._buffer_max_size:
                self._buffer.pop(0)
                self._buffer_index -= 1
            
            return self.current_frame, True
        else: ## There is something in the buffer to return
            self.current_frame = self._buffer[self._buffer_index]
            self._buffer_index += 1
            
            return self.current_frame, True
    
    def get_previous_frame(self) -> Tuple[np.ndarray, bool]:
        """Returns the previous frame stored in the buffer. If the buffer has reached the
        end then it returns the same frame as the previous call and the bool would be
        False.

        Returns:
            Tuple[np.ndarray, bool]: A tuple containing the previous frame, and a bool saying
            whether this is a new frame or whether its the same frame as the previous call.
        """

        if self._buffer_index == 0:
            return self.current_frame, False
        else:
            self._buffer_index -= 1
            self.current_frame = self._buffer[self._buffer_index]
            
            return self.current_frame, True

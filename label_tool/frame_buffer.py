import cv2

BUFFER_LIMIT = 100

class FrameBuffer:
    """Stores the frames in a video allowing the replay of them
    """
    def __init__(self, first_frame):
        self._data = [first_frame]
        self._index = 0
        self._frame = 0
        
        self.current_frame = first_frame
        
    def push_frame(self, frame):
        """Add new frame from video to the data structure

        Args:
            frame (frame): Frame to add

        Returns:
            frame: Current frame being played
        """
        self._data.append(frame)
        
        if len(self._data) > BUFFER_LIMIT:
            self._data.pop(0)
            self._index -= 1
        
        return self.get_next_frame()
        
    def get_previous_frame(self):
        """Rewind video to show previous frame

        Returns:
            frame: Current frame being played
        """
        self._frame = max(0, self._frame - 1)
        self._index = self._index - 1
        
        has_frame = True
        
        if self._index < 0:
            self._index = 0
            has_frame = False
        
        self.current_frame = self._data[self._index]
        
        return self.current_frame, has_frame
    
    def has_previous_frame(self):
        return self._index <= 0
    
    def get_next_frame(self):
        """If video has been rewound then plays the next frame stored
        in the data structure. If at most recent frame in data structure
        then returns `None`.

        Returns:
            frame: `None` if frame is most recent, else current frame being
            played
        """
        self._index += 1
        self._frame += 1
        
        if self._index >= len(self._data):
            self._index -= 1
            self._frame -= 1
            return None
        
        self.current_frame = self._data[self._index]
        
        return self.current_frame  
    
    def get_frame_number(self):
        """Returns the frame number of the current frame being played

        Returns:
            int: Frame number of current frame being played
        """
        return self._frame
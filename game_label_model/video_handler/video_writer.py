import cv2

class VideoWriter:
    def __init__(self, out_path, fps, width, height):
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self._out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        self.fps = fps

    def done(self):
        self._out.release()

class YOLOVideoWriter(VideoWriter):
    def __init__(self, out_path, fps, width, height):
        super().__init__(out_path, fps, width, height)

    def add_frame(self, frame):
        num_frames = int(round(self.fps / 2))

        for _ in range(0, num_frames):
            self._out.write(frame)
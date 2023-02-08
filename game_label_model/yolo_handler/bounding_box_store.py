from .bounding_box import BoundingBox

class BoundingBoxStore:
    """
    Wrapper class for storing a group of bounding boxes at a specific frame.
    This allows for better data organisation through knowledge of which bounding boxes occurred in which frames
    """

    def __init__(self, yolo_boxes, yolo_scores, yolo_classes, frameNum = -1):
        self.store = []
        self.frame = frameNum
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.append_boxes(yolo_boxes, yolo_classes, yolo_scores)

    def append_boxes(self, yolo_boxes, yolo_classes, yolo_scores):
        for box, score, classID in zip(yolo_boxes, yolo_scores, yolo_classes):
            x1, y1, x2, y2 = box.astype(int)
            bb = BoundingBox(x1, y1, x2 - x1, y2 - y1, score, self.class_names[classID], self.frame)
            self.store.append(bb)

    def get_store(self):
        """
        Method for safe access to the store array, allows access even if the direct variable name changes in the future
        """
        return self.store

    def print_store(self):
        """
        Method for printing the contents of the boundary box store.
        Useful for debugging
        """
        for bb in self.store:
            bb.printBB()

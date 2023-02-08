# Imports and dependencies
from decimal import InvalidContext
from operator import index
import os
import pathlib

import cv2

from .bounding_box_store import BoundingBoxStore
from .YOLOv7 import YOLOv7





class YOLOResult:
    """Wrapper class containing the output of the YOLO algorithm.
    
    Contains the boundry box, score, type, frame, position
    
    Don't want having random tuples with the results as it makes
    it more complex to understand
    """

    # Class Variables
    # Bounding boxes are stored as a class containing x, y, w, and h values of the bounding box
    # Also stores timestamp value of the detection, the confidence scores, and the predicted class
    bbs = []

    def __init__(self):
        self.bbs = []
    
    def get_BBs(self):
        """
        Returns an array of Bounding_Box_Stores, 1 store per frame
        Using this allows for safe access even if the variable name changes
        """
        return self.bbs

    def execute_YOLO(self, file_name, file_output = "output.mp4", visual_output = True):
        """
        Executes the YOLO algorithm, on the specified file.
        Writes out to another file and also stores bounding box results within a class variable

        Args:
            file_name       - The name of the file upon which yolo is desired to be performed.
            file_output     - The name of the file where the yolo result will be output
            visual_output   - Boolean variable. If true then the yolo output is displayed while processing
        """
        FILE_DIR = os.path.join(pathlib.Path(__file__).parent.parent.parent, "yolov7")
        MODEL_DIR = os.path.join(FILE_DIR, "models")
        OUT_DIR = os.path.join(FILE_DIR, "outputs")
        IN_DIR = os.path.join(FILE_DIR, "inputs")
        INFILE = os.path.join(IN_DIR, file_name)

        # Initialize video
        cap = cv2.VideoCapture(INFILE)

        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) #Get framerate to match for output rate
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width` converted to int
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height` converted to int
        else:
            raise Exception("File not found at: " + str(INFILE))

        outputFile = os.path.join(OUT_DIR, file_output)
        out = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height)) #Specify output file

        # Initialize YOLOv7 model
        model_name = "yolov7-tiny_384x640.onnx"                 # The chosen YOLO model, follow instructions in the README for download
        model_path = os.path.join(MODEL_DIR, model_name)
        #model_path = "models/yolov7-tiny_Nx3x384x640.onnx" 
        yolov7_detector = YOLOv7(model_path, conf_thres=0.5, iou_thres=0.5) #Initialise yolo model

        # Create and display window while processing
        cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

        frameNum = 0
        while cap.isOpened():

            # Press key q to stop
            if cv2.waitKey(1) == ord('q'):
                break

            try:
                # Read frame from the video
                ret, frame = cap.read()
                if not ret:
                    break
            except Exception as e:
                print(e)
                continue

            # Update object localizer
            boxes, scores, class_ids = yolov7_detector(frame)

            bbStore = BoundingBoxStore(boxes, class_ids, scores, frameNum)
            self.bbs.append(bbStore)
            frameNum += 1
            
            combined_img = yolov7_detector.draw_detections(frame)       # Draw the boxes on the image
            if visual_output:
                cv2.imshow("Detected Objects", combined_img)            # Display new image
            out.write(combined_img)                                     # Save the image to the output file

        # Clear up and save the output video
        out.release()

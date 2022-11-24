"""
#############################################
#
#   YOLOv7 Object Detection For Videos
#   23/11/22
#
############################################

Follow the README for explicit instructions

Prior to running:
    Download the yolov7 model prior to running (link in README) and place in a ./models folder
    Create a ./inputs and ./outputs directory, these are gitignored

Run the program using:
    python video_object_detection.py --source <video.mp4> --output <outfile.mp4>

All inputs and outputs are in specific directories so do not need to specify it in --source and --output

"""

# Imports and dependencies
import argparse
from decimal import InvalidContext
from operator import index
import os
import pathlib

import cv2
import numpy as np

from YOLOv7 import YOLOv7

if __name__ == "__main__":

    #Possible arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='videos/', help='output file')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()

    FILE_DIR = pathlib.Path(__file__).parent
    MODEL_DIR = os.path.join(FILE_DIR, "models")
    OUT_DIR = os.path.join(FILE_DIR, "outputs")
    IN_DIR = os.path.join(FILE_DIR, "inputs")
    INFILE = os.path.join(IN_DIR, opt.source)

    # Initialize video
    cap = cv2.VideoCapture(INFILE)

    fps = cap.get(cv2.CAP_PROP_FPS) #Get framerate to match for output rate

    outputFile = os.path.join(OUT_DIR, opt.output)
    out = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (1280, 720)) #Specify output file

    # Initialize YOLOv7 model
    model_name = "yolov7-tiny_Nx3x384x640.onnx"         # The chosen YOLO model, follow instructions in the README for download
    model_path = os.path.join(MODEL_DIR, model_name)
    #model_path = "models/yolov7-tiny_Nx3x384x640.onnx" 
    yolov7_detector = YOLOv7(model_path, conf_thres=0.5, iou_thres=0.5) #Initialise yolo model

    # Create and display window while processing
    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
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

        combined_img = yolov7_detector.draw_detections(frame)   # Draw the boxes on the image
        cv2.imshow("Detected Objects", combined_img)            # Display new image
        out.write(combined_img)                                 # Save the image to the output file

    # Clear up and save the output video
    out.release()

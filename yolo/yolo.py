"""
#############################################
#
#   YOLO Object Detection For Videos
#   03/11/22
#
############################################

Before running, make sure you obtain the weights file yolov3.weights (~200MB so in gitignore) and place it in the same directory

To run use the command:
    python3 yolo.py --image <Video.mp4> --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt

To not needlessly fill up the git, inputs and outputs are to be placed in a gitignored directory:
    ./inputs
    ./outputs
The program is set up to take account of these. 
As all inputs are in the inputs directory, you do not need to specify it in the arguments.
"""


# Imports + Dependencies
import argparse
import os
import pathlib

import cv2
import numpy as np


# Handle arguments used when called
# Final tool may or may not use this depending on our implementation, this just makes it easier for now
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = 'path to input image')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()


def get_output_layers(net):
    """
    Provide the layers for the output classes in the network

    Args:
        net - the read-in neural network
    """
    
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


# 
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    Draw the bounding boxes on the screen and also provide class label
    Uncomment the part in the label variable to add in confidence levels in prediction for testing

    Args:
        img         - The image to draw predictions on
        class_id    - The id of the class of the prediction
        confidence  - The numeric confidence level for the class prediction
        x           - The top left x coordinate of the box
        y           - The top left y coordinate of the box
        x_plus_w    - The bottom right x coordinate of the box
        y_plus_h    - The bottom right y coordinate of the box
    """

    label = str(classes[class_id])# + " - " +  str(round(confidence, 2))

    color = COLORS[class_id]    #Gives nice random colour :)

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



def yolo(image, net):
    """
    Method containing the YOLO algorithm for a single image
    Video analysis calls this function each frame

    Args:
        image   - The frame to make predicitons on
        net     - The pre-loaded neural network in use
    """

    # Get image dimensions
    width = image.shape[1]
    height = image.shape[0]
    scale = 0.00392

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    # Provide the image as the network input
    net.setInput(blob)

    # Forward propagate from the image to get an array of classes of detections
    outs = net.forward(get_output_layers(net))


    class_ids = []          # Classes detected
    confidences = []        # Confidence for each class
    boxes = []              # Box dimensions
    conf_threshold = 0.5    # If the network is less than 0.5 confidence, it will not make this prediction
    nms_threshold = 0.4     # Non-Maximum Supression Threshold


    # One output can have multiple difference detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)                # We want the class with the greatest confidence
            confidence = scores[class_id]               # Find what that greatest confidence is
            if confidence > 0.5:                        # If it's greater than our threshold then we accept it
                center_x = int(detection[0] * width)    # Find box centre
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)           # Calculate box width and height
                h = int(detection[3] * height)
                x = center_x - w / 2                    # Find the top left corner of the box
                y = center_y - h / 2
                class_ids.append(class_id)              # Add the info to the relevant arrays
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    # Perform non-maximum supression on the boxes found
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    #Find those now supressed boxes and draw them on the image
    for i in indices:
        try:
            box = boxes[i]
        except:
            i = i[0]
            box = boxes[i]
        
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
    
    # Return the image with the boxes now drawn on it
    return image


def saveVideo(images, name, fps, OUTPUT_DIR):
    """
    Function used to save the final output video. Takes the set of images and a filename as arguments
    Currently only saves to mp4 format but can be modified if necessary

    Args:
        images      - The array of images to be converted and saved in video format
        name        - The filename of the video to save
        fps         - The output fps of the video
        OUTPUT_DIR  - The directory where outputs videos are saved
    """

    width = images[0].shape[1]
    height = images[0].shape[0]
    name = os.path.join(OUTPUT_DIR, name)   # Saves output video to gitignored filepath
    video = cv2.VideoWriter(name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width,height))

    for image in images:
        video.write(image)

    # Clear up memory and save
    cv2.destroyAllWindows()
    video.release()


###############################################################################


if __name__ == "__main__":
    ## -- Defining useful directory paths
    FILE_DIR = pathlib.Path(__file__).parent
    INPUT_DIR = os.path.join(FILE_DIR, "inputs")
    OUTPUT_DIR = os.path.join(FILE_DIR, "outputs")

    # Initial set up of classes
    classes = None

    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    #Get a nice distribution of colours for boxes
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    # Read in the neural net and load weights
    net = cv2.dnn.readNet(args.weights, args.config)

    #Set up the video processor
    capture = cv2.VideoCapture(os.path.join(INPUT_DIR, args.image))

    count = 0       # Count can be used here if we wish to perform frame skipping
    images = []     # Array of all images for output video
    while True:

        _, image = capture.read()

        # If we want to greyscale stuff for some reason, this is the way
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Current way of detecting the end of video processing is when it throws an error as there is no image
        # This is caught and used as a loop exit condition
        try:
            image = yolo(image, net)
            images.append(image)
        except Exception as ex:
            print(ex)
            print("Finished processing")
            break

        # Provide an output whilst it is processing
        cv2.imshow('Object Detection In Videos', image)

        # 1ms wait to exit or pause the output video, funky behaviour occurs at 0 (frames don't display in preview) 
        # but this is known behaviour of the waitkey function
        k = cv2.waitKey(1) & 0xff

        if k == 27:
            break

    fps = capture.get(cv2.CAP_PROP_FPS)

    # Close the window
    capture.release()

    # De-allocate any associated memory usage
    cv2.destroyAllWindows()

    #Save the video output
    saveVideo(images, args.image, fps, OUTPUT_DIR)

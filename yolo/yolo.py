#############################################
#
#   YOLO Object Detection For Videos
#   03/11/22
#
############################################
#
# Before running, make sure you obtain the weights file yolov3.weights (~200MB so in gitignore) and place it in the same directory
# To run use the command:
#       python3 yolo.py --image <Video.mp4> --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt
#
# To not needlessly fill up the git, inputs and outputs are to be placed in a gitignored directory:
#       ./inputs
#       ./outputs
# The program is set up to take account of these. 
# As all inputs are in the inputs directory, you do not need to specify it in the arguments.


# Imports + Dependencies
import cv2
import argparse
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


# Provide the layers for the output classes in the network
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


# Draw the bounding boxes on the screen and also provide class label
# Uncomment the part in the label variable to add in confidence levels in prediction for testing
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])# + " - " +  str(round(confidence, 2))

    color = COLORS[class_id]    #Gives nice random colour :)

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



# Method containing the YOLO algorithm for a single image
# Video analysis calls this function each frame
def yolo(image, classes, COLORS):
    #image = cv2.imread(img)

    # Get image dimensions
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    # Read in the neural net and load weights
    # Going through this now this could be alot of the algorithmic complexity
    net = cv2.dnn.readNet(args.weights, args.config)

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
                center_x = int(detection[0] * Width)    # Find box centre
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)           # Calculate box width and height
                h = int(detection[3] * Height)
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

    #cv2.imshow("object detection", image)
    #cv2.waitKey()
        
    #cv2.imwrite("object-detection.jpg", image)
    #cv2.destroyAllWindows()


# Function used to save the final output video. Takes the set of images and a filename as arguments
# Currently only saves to mp4 format but can be modified if necessary
# Current bug exists with output framerate not matching input
def saveVideo(images, name):
    width = images[0].shape[1]
    height = images[0].shape[0]
    name = "outputs/" + name    # Saves output video to gitignored filepath
    video = cv2.VideoWriter(name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), cv2.CAP_PROP_FPS, (width,height))

    for image in images:
        video.write(image)

    # Clear up memory and save
    cv2.destroyAllWindows()
    video.release()


###############################################################################


# Initial set up of classes
classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

#Get a nice distribution of colours for boxes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

#Set up the video processor
capture = cv2.VideoCapture("inputs/" + args.image)

count = 0       # Count can be used here if we wish to perform frame skipping
images = []     # Array of all images for output video
while 1:

    _, image = capture.read()

    # If we want to greyscale stuff for some reason, this is the way
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Current way of detecting the end of video processing is when it throws an error as there is no image
    # This is caught and used as a loop exit condition
    try:
        image = yolo(image, classes, COLORS)
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

# Close the window
capture.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()

#Save the video output
saveVideo(images, "Test Output Game.mp4")

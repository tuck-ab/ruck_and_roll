# You Only Look Once: Object Detection

The You Only Look Once algorithm is intended to draw bounding boxes arounf players and the ball for this project.


## Setting up the repository
- Download the file: 'yolov3.weights' from this link: https://pjreddie.com/media/files/yolov3.weights

- Create directories called 'input' and 'output'. In the input directory place any *.mp4 videos to be input. The output videos will be saved into the output directory. 

## Command usage

```
python3 yolo.py --image <Video.mp4> --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt
```
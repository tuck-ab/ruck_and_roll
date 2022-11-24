# YOLOv7 setup and running

## Folders
Within the yolov7 directory, 3 new folders must be created:

* inputs
* outputs
* models

These folders are all covered in the gitignore due to file sizes

---

## YOLOv7 model download

The YOLOv7 models can be downloaded [here](https://drive.google.com/uc?export=download&id=1PrV-9oY1n5ptCF-YRymwp4mHOadK5DZK) and placed in the ./models folder. **This is under the GPLv3 license**. 

This downloads a folder full of them, I have found the best to be `yolov7-tiny_Nx3x384x640.onnx` which is what the current implemenatation uses, feel free to experiment by modifying the model name in video_object_detection.py.

---

## Requirements

New requirements are necessary for this, namely onnxruntime and pytorch, rerun the updated requirements.txt in your venv
```
pip install -r requirements.txt
```

---

## Running

Completing the prior means the model can now be run. To do so the command is:
```
python video_object_detection.py --source <video.mp4> --output <output.mp4>
```

---
Credits:
* YOLOv7 models: https://github.com/PINTO0309/PINTO_model_zoo/tree/main/307_YOLOv7
* YOLOv7 processing: https://github.com/ibaiGorordo/ONNX-YOLOv7-Object-Detection
* Original YOLOv7 paper: https://arxiv.org/abs/2207.02696
* YOLOv7 paper implementation: https://github.com/WongKinYiu/yolov7



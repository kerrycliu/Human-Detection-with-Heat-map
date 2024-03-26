# Human-Detection-with-Heat-map
An application that detects humans from a camera and presents the population desity as a heap map.
Utilizes YOLO, OpenCV, and other Machine Learning Libraries for detecting human objects accurately. 

## Installation
Libraries required:
from ultralytics import YOLO
from ultralytics.solutions import heatmap
import cv2
import json

Include pip install instructions
pip install ultralytics #this installs all dependencies



## Usage
How do users use your program/code
```
#for mac users
$ python3 main.py
#windows users
python .\main.py

```
### Testing

The program will ask for permission to laptop's camera.

The camera window will pop up named "frame" and green box will identify the objects in frame.

### Utilizing a webcam / camera

The webcam/camera will just be used for one purpose which is to harvest data for the heatmap, nothing else will be done using the camera. 

## Features

What are some of the features of the projects. What can people do and how to use it. 
1. Camera
2. 
>  > Detector people in static or motion.
>  > Set more than one camera. Different view points. Will the camera recongize the same human and only count 1.
3. Heat Map

## Sample Image
Put some screenshots if applicable. 

## Roadmap
1. Using CVAT, an open-source image and video annotation tool, to accurately detect humans only.
2. Gather human count from each camera frame and store in database.
3. Generate heatmap based on the data. 

## Contacts
Michelle Fang:

Kerry Liu:
PHONE: (510)3201972
EMAIL: chinan.liu@sjsu.edu

Anthony Nguyen: 
PHONE:(408)7057746

EMAIL:hoanganthony.nguyen@sjsu.edu

Klein Sicam:

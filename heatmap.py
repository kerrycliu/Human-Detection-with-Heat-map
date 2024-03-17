# new code to resize the video
from ultralytics import YOLO
from ultralytics.solutions import heatmap
import cv2

#capture console outputs
import sys

with open("text.txt", "a") as f:
    #redirect stdout to the text file
    sys.stdout = f

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training) (replace 'yolov8n.pt' with 'best.pt')

#Delete this after
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error reading video file"
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define target width and height
target_width = 1280
target_height = 720

# Video writer
video_writer = cv2.VideoWriter("heatmap_output.avi",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (target_width, target_height))

# Init heatmap
heatmap_obj1 = heatmap.Heatmap()
heatmap_obj1.set_args(colormap=cv2.COLORMAP_WINTER,
                    imw=target_width,
                    imh=target_height,
                     view_img=True,
                     shape="circle")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Resize frame to target dimensions
    frame = cv2.resize(frame, (target_width, target_height))

    # Perform object tracking
    tracks = model.track(frame, persist=True, show=False, classes=0)

    # Generate heatmap
    frame = heatmap_obj1.generate_heatmap(frame, tracks)

    # Write frame to output video
    video_writer.write(frame)
    
    #flush buffer to immediately write
    sys.stdout.flush()

cap.release()
video_writer.release()
cv2.destroyAllWindows()
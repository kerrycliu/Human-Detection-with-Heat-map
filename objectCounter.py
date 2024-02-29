from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)  # 0 represents the default webcam, you can change it if you have multiple webcams
assert cap.isOpened(), "Error accessing webcam"
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30  # Assuming default webcam records at 30 fps

# Define region points
region_points = [(0, 350), (640, 350), (640, 200), (0, 200)]

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=region_points,
                 classes_names=model.names,
                 draw_tracks=True)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Failed to capture frame from webcam.")
        break
    tracks = model.track(im0, persist=True, show=False)

    im0 = counter.start_counting(im0, tracks)

cap.release()
cv2.destroyAllWindows()

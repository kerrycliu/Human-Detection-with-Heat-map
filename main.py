import numpy as np
import cv2 as cv

vidCap = cv.VideoCapture(0)
if not vidCap.isOpened():
    print("Camera cannot be opened")
    exit(2)
while True:
    ret, frame = vidCap.read()

    if not ret:
        print("did not receive frame. closing")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

vidCap.release()
cv.destroyAllWindows()
exit(0)
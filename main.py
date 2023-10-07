import cv2
import cv2 as cv
import numpy as np
import time

vid = cv.VideoCapture(0)  #0 is for the main camera
if not vid.isOpened():  #check if there is a camera to open
    print("Can't open camera")
    exit()

hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

start_time = time.time()#inital start time
count = 0 #number of images

frame_count = 0

while True:
    #capture frame by frame
    ret, frame = vid.read()
    frame_count += 1

    #if frame is read corretly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Existing...")
        break

    if frame_count == 1: #skip 10 frames before detect a person

        #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Detect humans in the frame
        humans, _ = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

        # Draw rectangles around detected humans
        for (x, y, w, h) in humans:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        end_time = time.time()
        elapsed_time = end_time - start_time

        if elapsed_time >= 5:  # if time passed is 5 seconds take image
            print("taking image")  # comment this out later
            print(elapsed_time)  # comment this out later
            cv.imwrite("frame%d.jpg" % count, frame)  # save image
            start_time = time.time()  # new start time since its time after image is taken
            # count += 1 #used to save multiple images

        frame_count = 0

    cv.imshow('Human Detection', frame)  # display webcam

    if cv.waitKey(1) == ord('q'):#press q to turn of webcam and quit the program
        break

#when everything is done release the caputre
vid.release()
cv.destroyAllWindows()
exit()

import cv2 as cv
import numpy as np
import time

vid = cv.VideoCapture(0)  #0 is for the main camera
if not vid.isOpened():  #check if there is a camera to open
    print("Can't open camera")
    exit()

start_time = time.time()#inital start time
count = 0 #number of images
while True:
    #capture frame by frame
    ret, frame = vid.read()

    #if frame is read corretly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Existing...")
        break
    cv.imshow('frame', frame)  #display webcam

    end_time = time.time()
    elapsed_time = end_time - start_time

    if elapsed_time >= 5:  #if time passed is 5 seconds take image
        print("taking image") #comment this out later
        print(elapsed_time) #comment this out later
        cv.imwrite("frame%d.jpg" %count, frame)#save image
        start_time = time.time()  # new start time since its time after image is taken
        #count += 1 #used to save multiple images

    if cv.waitKey(1) == ord('q'):#press q to turn of webcam and quit the program
        break

#when everything is done release the caputre
vid.release()
cv.destroyAllWindows()
exit()

import cv2 #Package of AI 
import numpy as np 

#Lets capture the camera. 0 for webcam other webcam then we change to index to 1,2
cap = cv2.VideoCapture(0)

#Lets load the frame
while True:
    _, frame = cap.read() #Read the frame\

    #we convert this format to hsv , bgr library this sis color format red , green , blue
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#Lets frame on the window
    cv2.imshow("Frame", frame)
    

# weight key event ehich is 1 and which is 27 then break the loop that means we are closing the window
    key = cv2.waitKey(1)
    if key == 27:         # esc button from the keyboard
        break

    
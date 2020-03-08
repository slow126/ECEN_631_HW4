import cv2
import numpy as np





vc_left = cv2.VideoCapture("BaseBall_Pitch_R.avi")
if vc_left.isOpened():
    rval, im = vc_left.read()
else:
    rval = False

counter = 0;

while rval:
    rval, im_L = vc_left.read()
    cv2.imshow("left", im_L)
    cv2.waitKey()


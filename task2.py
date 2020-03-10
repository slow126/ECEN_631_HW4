import cv2
import numpy as np
import os

PATH = "pitch_img/"


prev_img = np.zeros((480, 640, 3))
base_imgL = cv2.imread(os.path.join(PATH, 'left/L_00001.jpeg'))
left_list = os.listdir(os.path.join(PATH, 'left'))
left_list.sort()
base_imgR = cv2.imread(os.path.join(PATH, 'right/R_00001.jpeg'))
right_list = os.listdir(os.path.join(PATH, 'right'))
right_list.sort()

params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 0
params.maxThreshold = 200

# Filter by Area.
params.filterByArea = True
params.minArea = 120

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.4

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.2

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.1
detector = cv2.SimpleBlobDetector_create(params)

prev_imgL = base_imgL
prev_imgR = base_imgR
prev_keyL = []
prev_keyR = []


for i in range(left_list.__sizeof__()):
    imgL = cv2.imread(os.path.join(PATH, "left", left_list[i]))
    imgR = cv2.imread(os.path.join(PATH, "right", right_list[i]))

    # if prev_keyL != []:
    #     for i in range(len(prev_keyL)):
    #         cv2.circle(prev_imgL, (int(prev_keyL[i].pt[0]), int(prev_keyL[i].pt[1])), int(prev_keyL[i].size / 2), (255,255,255), thickness=-1)
    #
    # cv2.imshow("prev", prev_imgL)
    # cv2.waitKey()

    diffL = cv2.absdiff(prev_imgL, imgL)
    diffR = cv2.absdiff(prev_imgR, imgR)

    diffL[diffL < 10] = 0
    diffL[diffL >= 10] = 150
    diffR[diffR < 10] = 0
    diffR[diffR >= 10] = 150

    # if prev_keyL != []:
    #     for i in range(len(prev_keyL)):
    #         cv2.circle(diffL, (int(prev_keyL[i].pt[0]), int(prev_keyL[i].pt[1])), int(prev_keyL[i].size / 2), (255,255,255), thickness=-1)



    diffL = ~diffL
    diffR = ~diffR
    # cv2.imshow("output", diff)
    # cv2.waitKey(0)
    diffL = cv2.erode(cv2.dilate(diffL, (6,6)),(6,6))
    diffR = cv2.erode(cv2.dilate(diffR, (6,6)),(6,6))

    keypoints = detector.detect(diffL)
    prev_keyL = keypoints
    if len(keypoints) >= 0:
        im_with_keypointsL = cv2.drawKeypoints(imgL, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    else:
        im_with_keypointsL = cv2.drawKeypoints(imgL, keypoints[1], np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    keypoints = detector.detect(diffR)
    prev_keyR = keypoints
    if len(keypoints) >= 0:
        im_with_keypointsR = cv2.drawKeypoints(imgR, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    else:
        im_with_keypointsR = cv2.drawKeypoints(imgR, keypoints[1], np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imwrite("Annotated_Images/" + file, im_with_keypoints)

    prev_imgL = imgL
    prev_imgR = imgR

    cv2.imshow("outputL", im_with_keypointsL)
    cv2.imshow("outputR", im_with_keypointsR)
    cv2.waitKey(50)



# count = 0
# vc_left = cv2.VideoCapture("BaseBall_Pitch_L.avi")
# if vc_left.isOpened():
#     rval, im = vc_left.read()
# else:
#     rval = False
#
# counter = 0;
#
# while rval:
#     count = count + 1
#     rval, im_L = vc_left.read()
#     cv2.imwrite(os.path.join("pitch_img/left", "L_" + str(count).zfill(5) + ".jpeg"), im_L)
#

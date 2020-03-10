import cv2
import numpy as np
import os
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

catcher_y_shift = 36
stereo_x_shift = 20

R1 = np.load("R_left.npy")
R2 = np.load("R_right.npy")
P1 = np.load("P_left.npy")
P2 = np.load("P_right.npy")
Q = np.load("Q.npy")
left_mtx = np.load("left-mtx.npy")
left_dist = np.load("left-dist.npy")
right_mtx = np.load("right-mtx.npy")
right_dist = np.load("right-dist.npy")


PATH = "one_pitch/"

prev_img = np.zeros((480, 640, 3))
base_imgL = cv2.imread(os.path.join(PATH, 'left/L_00450.jpeg'))
left_list = os.listdir(os.path.join(PATH, 'left'))
left_list.sort()
base_imgR = cv2.imread(os.path.join(PATH, 'right/R_00450.jpeg'))
right_list = os.listdir(os.path.join(PATH, 'right'))
right_list.sort()

params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 0
params.maxThreshold = 150

# Filter by Area.
params.filterByArea = True
params.minArea = 175

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.5

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.1

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.1
detector = cv2.SimpleBlobDetector_create(params)

prev_imgL = base_imgL
prev_imgR = base_imgR

my_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

y = 640
x = 480

xL = []
yL = []
zL = []
xR = []
yR = []
zR = []

def to3D(left_key, right_key):
    left_rect = cv2.undistortPoints(src=np.array([[(left_key[0], left_key[1])]]), cameraMatrix=left_mtx, distCoeffs=left_dist, R=R1, P=P1)
    right_rect = cv2.undistortPoints(src=np.array([[np.transpose(right_key)]]), cameraMatrix=right_mtx, distCoeffs=right_dist, R=R2, P=P2)

    t2 = [[0]]
    diff = np.squeeze(left_rect - right_rect)
    disparity = diff[0]


    t_left = np.dstack((left_rect, t2))
    t_right = np.dstack((right_rect, t2))
    # t = np.append(t, np.array([[disparity]]).transpose(), axis=2)
    for i in range(len(t2)):
        t_left[i,0,2] = disparity
        t_right[i,0,2] = disparity

    left_persp = cv2.perspectiveTransform(t_left, Q)
    right_persp = cv2.perspectiveTransform(t_right, Q)
    return left_persp, right_persp


for i in range(len(left_list)):
    imgL = cv2.imread(os.path.join(PATH, "left", left_list[i]))
    imgR = cv2.imread(os.path.join(PATH, "right", right_list[i]))


    diffL = cv2.absdiff(prev_imgL, imgL)
    diffR = cv2.absdiff(prev_imgR, imgR)

    diffL[diffL < 10] = 0
    diffL[diffL >= 10] = 150
    diffR[diffR < 10] = 0
    diffR[diffR >= 10] = 150

    diffL = ~diffL
    diffR = ~diffR
    # cv2.imshow("output", diff)
    # cv2.waitKey(0)
    diffL = cv2.medianBlur(diffL, 5)
    diffR = cv2.medianBlur(diffR, 5)
    diffL = cv2.erode(cv2.dilate(diffL, my_kernel), my_kernel)
    diffR = cv2.erode(cv2.dilate(diffR, my_kernel), my_kernel)



    keypoints = detector.detect(diffL)
    left_key = keypoints
    im_with_keypointsL = cv2.drawKeypoints(imgL, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    keypoints = detector.detect(diffR)
    right_key = keypoints
    im_with_keypointsR = cv2.drawKeypoints(imgR, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    if left_key != [] and right_key != []:
        left_pt, right_pt = to3D(left_key[0].pt, right_key[0].pt)
        xL.append(left_pt[0, 0, 0])
        yL.append(left_pt[0, 0, 1])
        zL.append(left_pt[0, 0, 2])
        xR.append(right_pt[0, 0, 0])
        yR.append(right_pt[0, 0, 1])
        zR.append(right_pt[0, 0, 2])


    # prev_imgL = imgL
    # prev_imgR = imgR

    cv2.imshow("outputL", im_with_keypointsL)
    cv2.imshow("outputR", im_with_keypointsR)
    cv2.waitKey(1)

ax = plt.axes(projection='3d')


ax.plot3D(xL, yL, zL, 'red')
plt.show()
plt.plot(xL, yL)
plt.show()
plt.plot(zL, yL)
plt.show()
plt.plot(zL, xL)
plt.show()

sio.savemat("xL.mat", {'xL':xL})
sio.savemat("yL.mat", {'yL':yL})
sio.savemat("zL.mat", {'zL':zL})

# ax.plot3D(xR, yR, zR, 'red')
# plt.show()
# plt.plot(xR, yR)
# plt.show()
# plt.plot(zR, yR)
# plt.show()
# plt.plot(zR, xR)
# plt.show()
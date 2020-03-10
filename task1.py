import cv2
import numpy as np
import os

PATH = "calibration_img"
files = os.listdir(PATH)

def find_corners(in_img, pattern=(10, 7)):
    color_img = in_img
    in_img = cv2.cvtColor(in_img, cv2.COLOR_RGB2GRAY)
    # color_img = cv2.cvtColor(in_img, cv2.COLOR_GRAY2RGB)
    # gray_gpu = cv2.UMat(gray)
    # pattern = (10, 7)
    pattern2 = (23, 11)

    objp = np.zeros((pattern[1] * pattern[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    ret, corners = cv2.findChessboardCorners(color_img, pattern)

    annotated_gray = color_img
    sub_corners = None

    if ret is True:
        objpoints.append(objp)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        sub_corners = cv2.cornerSubPix(in_img, np.float32(corners), (5, 5), (-1, -1), criteria)
        imgpoints.append(np.squeeze(sub_corners))
        annotated_gray = cv2.drawChessboardCorners(color_img, pattern, sub_corners, ret)

    # cv2.imshow("img", annotated_gray)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    objpoints = np.array(objpoints) * 3.88
    return annotated_gray, sub_corners, objpoints, imgpoints

for file in files:
    img = cv2.imread(os.path.join(PATH, file))
    if "L" in file:
        left = img
        annotated_left, left_sub, left_obj, left_img = find_corners(left)
    if "R" in file:
        right = img
        annotated_right, right_sub, right_obj, right_img = find_corners(right)

R1 = np.load("R_left.npy")
R2 = np.load("R_right.npy")
P1 = np.load("P_left.npy")
P2 = np.load("P_right.npy")
Q = np.load("Q.npy")
left_mtx = np.load("left-mtx.npy")
left_dist = np.load("left-dist.npy")
right_mtx = np.load("right-mtx.npy")
right_dist = np.load("right-dist.npy")

left_img_corners = [left_sub[0], left_sub[9], left_sub[-1], left_sub[-10]]
left_obj_corners = [left_obj[0][0], left_obj[0][9], left_obj[0][-1], left_obj[0][-10]]
right_img_corners = [right_sub[0], right_sub[9], right_sub[-1], right_sub[-10]]
right_obj_corners = [right_obj[0][0], right_obj[0][9], right_obj[0][-1], right_obj[0][-10]]

left_rect = cv2.undistortPoints(src=np.array(left_img_corners), cameraMatrix=left_mtx, distCoeffs=left_dist, R=R1, P=P1)
right_rect = cv2.undistortPoints(src=np.array(right_img_corners), cameraMatrix=right_mtx, distCoeffs=right_dist, R=R2, P=P2)

# left_rect = cv2.undistortPoints(np.array(left_img_corners), left_mtx, left_dist, np.array(left_obj_corners), R1, P1)
# right_rect = cv2.undistortPoints(np.array(right_img_corners), right_mtx, right_dist, np.array(right_obj_corners), R2, P2)

t_left = np.squeeze(left_rect)
t2 = [[0],[0],[0],[0]]
diff = np.squeeze(left_rect - right_rect)
disparity = diff[:,0]

# t = np.dstack((t, disparity))


t_left = np.dstack((left_rect, t2))
t_right = np.dstack((right_rect, t2))
# t = np.append(t, np.array([[disparity]]).transpose(), axis=2)
for i in range(4):
    t_left[i,0,2] = disparity[i]
    t_right[i,0,2] = disparity[i]

left_persp = cv2.perspectiveTransform(t_left, Q)
right_persp = cv2.perspectiveTransform(t_right, Q)
x = 1




import cv2
import numpy as np
import os

PATH = "calibration_img"
files = os.listdir(PATH)

def find_corners(in_img, pattern=(10, 7)):

    color_img = cv2.cvtColor(in_img, cv2.COLOR_GRAY2RGB)
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
    return annotated_gray, sub_corners, objpoints, imgpoints

for file in files:
    img = cv2.imread(os.path.join(PATH, file))
    if "L" in file:
        left = img
        annotated_left, left_sub, left_obj, left_img = find_corners(left)
    if "R" in file:
        right = img
        annotated_right, right_sub, right_obj, right_img = find_corners(right)





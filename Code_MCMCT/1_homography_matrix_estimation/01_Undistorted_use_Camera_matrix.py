"""
This code is used to remove the distortion of the image
"""


import cv2
import numpy as np
# 读取图像
imageName = '02_01.jpg'
image = cv2.imread('H_get_1K/'+imageName)


# 相机标定参数
fx = 2.478905937954560e+03  # x轴焦距
fy = 2.481041361558221e+03  # y轴焦距
cx = 9.699174431836150e+02   # 主点在x轴上的坐标
cy = 5.273809309141056e+02   # 主点在y轴上的坐标
k1 = -1.235571086123972   # 径向畸变系数 k1
k2 = 2.098583076902621  # 径向畸变系数 k2
p1 = 0.000000001 # 切向畸变系数 p1
p2 = 0.000000001 # 切向畸变系数 p2
k3 = 0     # 径向畸变系数 k3

# 相机内参矩阵
camera_matrix = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]], dtype=np.float32)


# 畸变参数
distortion_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

# 畸变校正
undistorted_image = cv2.undistort(image, camera_matrix, distortion_coeffs)
undistorted_imageName = "undistorted_"+imageName
cv2.imwrite('H_get_1K_undistorted/'+undistorted_imageName, undistorted_image)


# 显示结果
cv2.imshow("Original Image", cv2.resize(image, (990, 540)))
cv2.imshow("Undistorted Image", cv2.resize(undistorted_image, (990, 540)))
cv2.waitKey(0)
cv2.destroyAllWindows()

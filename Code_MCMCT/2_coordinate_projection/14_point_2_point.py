"""
此代码用于将原始图像中的点(x,y)转换为透射投影后的点(x2,y2)
"""
import cv2
import numpy as np


def undistort_image_point(x, y, camera_matrix, distortion_coeffs):
    # 重构成一个点的数组格式
    points = np.array([[[x, y]]], dtype=np.float32)

    # 使用cv2.undistortPoints进行校正
    undistorted_points = cv2.undistortPoints(points, camera_matrix, distortion_coeffs)

    # 获取校正后的坐标
    x1, y1 = undistorted_points[0][0]

    return x1, y1


# 相机标定参数
fx = 2.478905937954560e+03  # x轴焦距
fy = 2.481041361558221e+03  # y轴焦距
cx = 9.699174431836150e+02  # 主点在x轴上的坐标
cy = 5.273809309141056e+02  # 主点在y轴上的坐标
k1 = -1.235571086123972  # 径向畸变系数 k1
k2 = 2.098583076902621  # 径向畸变系数 k2
p1 = 0.000000001  # 切向畸变系数 p1
p2 = 0.000000001  # 切向畸变系数 p2
k3 = 0  # 径向畸变系数 k3

# 相机内参矩阵
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float32)

# 畸变参数
distortion_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

# 输入原始图像坐标
# x = 271
# y = 482

x = 858
y = 319

# 校正图像坐标
x1, y1 = undistort_image_point(x, y, camera_matrix, distortion_coeffs)

# 图像尺寸
image_width = 1920
image_height = 1080

# 以像素为单位的校正后图像坐标
x1_pixel = int(x1 * fx + cx)
y1_pixel = int(y1 * fy + cy)

# 输出以像素为单位的校正后图像坐标
print("原始图像坐标 ({}, {}) 对应的校正后图像坐标为 ({}, {})".format(x, y, x1_pixel, y1_pixel))

# 透射变换矩阵
H = np.array([[8.96006877e-01, 2.31916020e+00, -8.51326377e+02],
              [-5.12414468e-01, 4.65595720e+00, -6.64092910e+01],
              [-8.59537155e-05, 2.21599522e-03, 1.00000000e+00]])

# 转换为齐次坐标
point_homogeneous = np.array([[x1_pixel], [y1_pixel], [1]])

# 应用透射变换矩阵
transformed_point = np.dot(H, point_homogeneous)

# 获取真实坐标 (x2, y2)
x2 = transformed_point[0, 0] / transformed_point[2, 0]
y2 = transformed_point[1, 0] / transformed_point[2, 0]

# 输出真实坐标
print("校正图像中的点 ({}, {}) 对应的透射投影后的真实坐标为 ({}, {})".format(x1_pixel, y1_pixel, x2, y2))

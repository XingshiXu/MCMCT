"""

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


# 读取图像
imageName = '02_01.jpg'
image = cv2.imread('H_get_1K/' + imageName)

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
# x = 825
# y = 170
x=1053
y=194

# 校正图像坐标
x1, y1 = undistort_image_point(x, y, camera_matrix, distortion_coeffs)

# 以像素为单位的校正后图像坐标
x1_pixel = int(x1 * fx + cx)
y1_pixel = int(y1 * fy + cy)

# 创建一个空白的图像，大小与原始图像相同
image_size = (image.shape[1], image.shape[0])
undistorted_image = np.zeros(image_size, dtype=np.uint8)

# 在校正后的图像上绘制一个圆来标记原始坐标点
radius = 5
color = 255  # 白色
thickness = -1  # 实心圆
cv2.circle(undistorted_image, (x1_pixel, y1_pixel), radius, color, thickness)

# 保存校正后的图像
undistorted_imageName = "undistorted_" + imageName
cv2.imwrite(undistorted_imageName, undistorted_image)

# 输出校正后的图像坐标
print("原始图像坐标 ({}, {}) 对应的校正后图像坐标为 ({}, {})".format(x, y, x1_pixel, y1_pixel))

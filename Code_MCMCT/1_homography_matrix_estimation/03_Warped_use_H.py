"""
此代码用于利用 homography matrix获得 bird-eye 图像
"""

import cv2
import numpy as np

# 读取图像
image = cv2.imread(r'H_get_1K_undistorted/undistorted_02_01.jpg')
# image = cv2.imread(r'H_get_1K_undistorted/undistorted_03_01.jpg')

# 假设你已经有了透视变换矩阵H


H = np.array( [[ 8.96006877e-01,  2.31916020e+00, -8.51326377e+02],
 [-5.12414468e-01, 4.65595720e+00, -6.64092910e+01],
 [-8.59537155e-05,  2.21599522e-03,  1.00000000e+00]])

"""
H = np.array( [[-1.03787705e+00,  1.88670687e+00,  1.03010487e+03],
 [-4.18113744e-01, -1.58264043e-01,  2.51505809e+03],
 [-2.36097600e-04,  1.91716677e-03, 1.00000000e+00]])
"""

# 应用透视变换到图像上
warped_image = cv2.warpPerspective(image, H, (image.shape[1]*2, image.shape[0]*2))

# 调整图像大小以适应800x600的界面
# max_width = 990
# max_height = 540
max_width = 1980
max_height = 1080

# 原始图像调整
original_resized = cv2.resize(image, (max_width, max_height), interpolation=cv2.INTER_AREA)

# 变换后的图像调整
warped_resized = cv2.resize(warped_image, (max_width, max_height), interpolation=cv2.INTER_AREA)

# 显示原始图像和应用透视变换后的图像
cv2.imshow('Original Image', original_resized)
cv2.imshow('Warped Image', warped_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存调整后的原始图像
cv2.imwrite('original_resized.jpg', original_resized)

# 保存透视变换后的图像
cv2.imwrite('H_get_1K_undistorted_birdEye/warped_resized_02.jpg', warped_resized)



# 定义图像上的点的坐标
image_point = np.array([[202, 409]], dtype=np.float32)

# 将图像中的点转换为齐次坐标
image_point_homogeneous = np.array([[[image_point[0, 0], image_point[0, 1], 1]]], dtype=np.float32)

# 进行透视变换
inv_H = np.linalg.inv(H)
space_point_homogeneous = cv2.perspectiveTransform(image_point_homogeneous, inv_H)

# 将齐次坐标转换为三维空间坐标
space_point = space_point_homogeneous[:, :, :2] / space_point_homogeneous[:, :, 2:]

print("Image Point:", image_point)
print("Space Point:", space_point)


# Multi-Camera Multi-Cow Tracking under Non-Overhead Views

 
This is the official implementation of the core components of MCMCT, including homography matrix estimation, coordinate projection, dairy cow target detection, single-camera multi-cow tracking, and tracklet association.

 
# ⭐Homography_Matrix_Estimation
We provide the code for distortion correction as well as the computation of the homography matrix. These codes can be found in ./1_homography_matrix_estimation. After obtaining the Homography Matrix, you can try using 03_Warped_use_H.py to generate the bird’s-eye view image.

# 🚀 Coordinate Projection
This code is used to transform a point (x, y) in the original image to a point (x₂, y₂) in the perspective-transformed image. These codes can be found in ./2_coordinate_projection.

# 🐟 Dairy Cow Target Detection
This part is based on the YOLOv8 framework. You can refer to the official tutorial for implementation: https://docs.ultralytics.com/zh/models/yolov8/.
The implementation of Quaternion Convolution and Repulsion Loss is based on the following resources:
https://github.com/Orkis-Research/Pytorch-Quaternion-Neural-Networks and https://github.com/rainofmine/Repulsion_Loss.

# 🥂 Single-camera Multi-cow Tracking
This part is supported by the BoxMot repository. The original project code is available at: https://github.com/mikel-brostrom/boxmot.
You can run the implementation via ./boxmot/trackers/botsort/bot_sort.py and ./tracking/track2.py.

# 🚢 Tracklet Association
These codes can be found in ./5_tracklet_association.

# 🥳 Communication

Xingshi Xu: xingshixu@nwafu.edu.cn  
Huaibo Song: songhuaibo@nwsuaf.edu.cn  
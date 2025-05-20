import time
from ultralytics import YOLO

import torch
from thop import profile
from ultralytics import YOLO

# 加载模型
model = YOLO(model="ultralytics/cfg/models/v8/yolov8Quaternion.yaml",)

# 创建一个输入张量，假设输入尺寸为 640x640
input = torch.randn(1, 3, 640, 640)

# 计算 FLOPs 和参数量
flops, params = profile(model.model, inputs=(input,))

# 打印结果
print(f"FLOPs: {flops / 1e9} GFLOPs")
print(f"Parameters: {params / 1e6} M")


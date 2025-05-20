import time
from ultralytics import YOLO


# yolov8n模型训练
model = YOLO(model="ultralytics/cfg/models/v8/yolov8Quaternion.yaml",)  # load a pretrained model (recommended for training)
results = model.train(data=r'/media/v10016/实验室备份/XingshiXu/ultralytics/ultralytics/cfg/datasets/MutiCOW.yaml', model="ultralytics/cfg/models/v8/yolov8000.yaml",epochs=100, imgsz=640, device=[0,1,2,3], workers=2, batch=128, cache=True)  # 开始训练
time.sleep(5)

print()
from ultralytics import YOLO

# Load a model
model = YOLO("runs/pose/train/weights/best.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model(
    "/mnt/mydisk/xiaolei/code/plate/yolov11/yolov11-plate-landmarks/imgs/double_yellow.jpg",
    save=True,           # 保存识别结果图片
    project="runs/predict",  # 保存目录
    name="results",       # 子目录名
)

# Access the results
for result in results:
    xy = result.keypoints.xy  # x and y coordinates
    xyn = result.keypoints.xyn  # normalized
    kpts = result.keypoints.data  # x, y, visibility (if available)
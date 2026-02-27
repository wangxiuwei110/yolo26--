import os
# os.environ["OMP_NUM_THREADS"]='2'

from ultralytics import YOLO
# Load a model
model = YOLO('yolo26s-pose.yaml')  # build a new model from YAML
model = YOLO('yolo26s-pose.pt')  # load a pretrained model (recommended for training)  

# Train the model
model.train(data=r'cfg/plate.yaml', epochs=120, imgsz=640, batch=16, device=[0])
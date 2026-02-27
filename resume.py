from ultralytics import YOLO

# Load a model
model = YOLO("runs/pose/train/weights/last.pt")  # load a partially trained model

# Resume training
results = model.train(resume=True)
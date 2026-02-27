from ultralytics import YOLO

# Load a model
# model = YOLO(r"yolo26s-pose.yaml")  # build a new model from scratch
model = YOLO(r"weights/best_yolo26_pose.pt")  # load a pretrained model (recommended for training)

# Use the model
# model.train(data="coco128.yaml", epochs=3)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(
    format="onnx",
    simplify=True,          # 关闭简化，避免 onnxslim 依赖
    dynamic=False,          # 固定输入形状，避免形状推断问题
    opset=12,             # 指定 ONNX opset 版本
)
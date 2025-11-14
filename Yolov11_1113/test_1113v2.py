from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n model
model.train(data="coco128.yaml")  # train the model
model.val()  # evaluate model performance on the validation set
model.predict(source="https://ultralytics.com/images/bus.jpg",save=True)  # predict on an image
model.export(format="onnx")  # export the model to ONNX format
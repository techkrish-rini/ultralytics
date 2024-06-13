from ultralytics import YOLO


def train():
    # Load a model
    # model = YOLO('yolov8n.yaml')  # build a new model from YAML
    model = YOLO('yolov8m.pt')  # load a pretrained model (recommended for training)
    # model = YOLO('yolov8l.pt')  # load a pretrained model (recommended for training)
    results = model.train(data='bird_classifier.yaml', epochs=100, imgsz=1024)


if __name__ == "__main__":
    train()



from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo26n-cls.pt")
    results = model.train(data="/home/videk/Desktop/faks/rins/face_classification/augmented_personnel", epochs=100, imgsz=64)
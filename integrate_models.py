
import torch
import time
from yolo_model import YOLODetection
from facial_recognition import FaceRecognition
from utils import capture_screen

if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()

    yolo = YOLODetection(use_gpu=use_gpu)
    face_recog = FaceRecognition(model_path='models/facial_model.h5', use_gpu=use_gpu)
    
    while True:
        frame = capture_screen({"top": 100, "left": 100, "width": 800, "height": 600})
        yolo.detect_and_process(frame)
        face_recog.detect_and_process(frame)
        time.sleep(0.01)

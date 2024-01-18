from ultralytics import YOLO
import math
import cvzone
import cv2
cap = cv2.VideoCapture(0) #for webcam
#cap = cv2.VideoCapture("input_video.mp4")
cap.set(3,1280)
cap.set(4,720)
MODEL = "yolov8n.pt"
from ultralytics import YOLO
import cv2
model = YOLO(MODEL)


while True:
    success, img = cap.read()
    results = model(img ,show=True)

    cv2.waitKey(1)

import cv2
from ultralytics import YOLO
import math
import cvzone
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils.sort import *

#for power_rangers.mp4 use this link https://www.youtube.com/watch?v=iBbepfFzAiM
cap = cv2.VideoCapture("data/power_rangers.mp4")
model =  YOLO("yolos/yolov8l.pt")

classes = {0: 'person',
 1: 'bicycle',
 2: 'car',
 3: 'motorcycle',
 4: 'airplane',
 5: 'bus',
 6: 'train',
 7: 'truck',
 8: 'boat',
 9: 'traffic light',
 10: 'fire hydrant',
 11: 'stop sign',
 12: 'parking meter',
 13: 'bench',
 14: 'bird',
 15: 'cat',
 16: 'dog',
 17: 'horse',
 18: 'sheep',
 19: 'cow',
 20: 'elephant',
 21: 'bear',
 22: 'zebra',
 23: 'giraffe',
 24: 'backpack',
 25: 'umbrella',
 26: 'handbag',
 27: 'tie',
 28: 'suitcase',
 29: 'frisbee',
 30: 'skis',
 31: 'snowboard',
 32: 'sports ball',
 33: 'kite',
 34: 'baseball bat',
 35: 'baseball glove',
 36: 'skateboard',
 37: 'surfboard',
 38: 'tennis racket',
 39: 'bottle',
 40: 'wine glass',
 41: 'cup',
 42: 'fork',
 43: 'knife',
 44: 'spoon',
 45: 'bowl',
 46: 'banana',
 47: 'apple',
 48: 'sandwich',
 49: 'orange',
 50: 'broccoli',
 51: 'carrot',
 52: 'hot dog',
 53: 'pizza',
 54: 'donut',
 55: 'cake',
 56: 'chair',
 57: 'couch',
 58: 'potted plant',
 59: 'bed',
 60: 'dining table',
 61: 'toilet',
 62: 'tv',
 63: 'laptop',
 64: 'mouse',
 65: 'remote',
 66: 'keyboard',
 67: 'cell phone',
 68: 'microwave',
 69: 'oven',
 70: 'toaster',
 71: 'sink',
 72: 'refrigerator',
 73: 'book',
 74: 'clock',
 75: 'vase',
 76: 'scissors',
 77: 'teddy bear',
 78: 'hair drier',
 79: 'toothbrush'}

result_array = [classes[i] for i in range(len(classes))]

mask = cv2.imread("images/helper_images/r2.jpg")

tracker = Sort(max_age=20, min_hits=2,iou_threshold=0.3)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
m = cv2.VideoWriter('runner_tracking.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
size = (frame_width, frame_height)
while True:
    _, frame = cap.read()
    imgRegion = cv2.bitwise_and(frame,mask)
    results = model(imgRegion,stream=True)
    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1, x2, y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            w,h = x2-x1,y2-y1
            
            cls = int(box.cls[0])
            if result_array[cls] =='person':
                cvzone.cornerRect(frame,(x1,y1,w,h),l=5, rt = 2, colorC=(0,0,255), colorR=(71,99,255))
            conf = math.ceil((box.conf[0]*100))/100
            ins = np.array([x1,y1,x2,y2,conf])
            detections = np.vstack((detections,ins))

    
    tracks = tracker.update(detections)
    for result in tracks:
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        print(result)
        if result_array[cls] =='person':
            cvzone.putTextRect(frame,f'{result_array[cls]} {conf} id:{int(id)} ',(max(0,x1),max(35,y1-20)),scale=1, thickness=1, offset=3, colorR=(71,99,255))
    m.write(frame)
    cv2.imshow("Image",frame)
    cv2.waitKey(1)

cap.release()
m.release()
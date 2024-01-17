import cv2
from ultralytics import YOLO
import math
import cvzone
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils.sort import *

# cap = cv2.VideoCapture(0) #for webcam
# cap.set(3,1280)
# cap.set(4,720)

cap = cv2.VideoCapture("data/los_angeles.mp4")
model =  YOLO("yolos/yolov8n.pt")

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
mask = cv2.imread("r.jpg")


#tracker = DeepSort(max_age=50)
tracker = Sort(max_age=20, min_hits=2,iou_threshold=0.3)
l = [593,500,958,500]
totalCount = []

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
m = cv2.VideoWriter('traffic_tracking.avi', 
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
            cvzone.cornerRect(frame,(x1,y1,w,h),l=5, rt = 2, colorC=(255,215,0), colorR=(255,99,71))
            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            ins = np.array([x1,y1,x2,y2,conf])
            detections = np.vstack((detections,ins))

    
    tracks = tracker.update(detections)
    cv2.line(frame, (l[0],l[1]),(l[2],l[3]),color=(255,0,0),thickness=3)
    for result in tracks:
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        print(result)
        cvzone.putTextRect(frame,f'{result_array[cls]} {conf} id:{int(id)} ',(max(0,x1),max(35,y1-20)),scale=1, thickness=1, offset=3, colorR=(255,99,71))
        cx,cy = x1+w//2,y1+h//2
        if l[0]<cx<l[2] and l[1]-10<cy<l[3]+10:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(frame, (l[0],l[1]),(l[2],l[3]),color=(127,255,212),thickness=5)
    cvzone.putTextRect(frame,f' Total Count: {len(totalCount)} ',(70,70),scale=2, thickness=1, offset=3, colorR=(255,99,71))
    m.write(frame)
    cv2.imshow("Image",frame)
    cv2.waitKey(1)

cap.release()
m.release()
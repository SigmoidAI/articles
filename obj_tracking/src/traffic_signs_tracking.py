import cv2
from ultralytics import YOLO
import math
import cvzone
import numpy as np
cap = cv2.VideoCapture("data/redlight_greenlight.mp4")
model =  YOLO("yolos/road_signs_yolo.pt")
result = ['bus_stop', 'do_not_enter', 'do_not_stop', 'do_not_turn_l', 'do_not_turn_r', 'do_not_u_turn', 'enter_left_lane', 'green_light', 'left_right_lane', 'no_parking', 'parking', 'ped_crossing', 'ped_zebra_cross', 'railway_crossing', 'red_light', 'stop', 't_intersection_l', 'traffic_light', 'u_turn', 'warning', 'yellow_light']
result_array = [result[i] for i in range(len(result))]


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
m = cv2.VideoWriter('output.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
size = (frame_width, frame_height)
while True:
    success, img = cap.read()
    results = model(img,stream=True)
    detections = np.empty((0,5))
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1, x2, y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            w,h = x2-x1,y2-y1
            conf = math.ceil((box.conf[0]*100))/100 
            cls = int(box.cls[0])
            if result_array[cls] == 'red_light':
                cvzone.cornerRect(img,(x1,y1,w,h),l=5, rt = 3, colorR=(0, 0 ,255) ,colorC= (0, 0 ,255))
                cvzone.putTextRect(img,f'{conf} {result_array[cls]} ',(max(0,x1),max(35,y1-20)),scale=1, thickness=1, offset=3, colorR=(0, 0 ,255))
            elif result_array[cls] == 'green_light':
                cvzone.cornerRect(img,(x1,y1,w,h),l=5, rt = 3, colorR=(0, 255, 0))
                cvzone.putTextRect(img,f'{conf} {result_array[cls]} ',(max(0,x1),max(35,y1-20)),scale=1, thickness=1, offset=3, colorR=(0, 255, 0))

    m.write(img)
    cv2.imshow("Image",img)
    cv2.waitKey(1)

cap.release()
m.release()
    


---

Video Object Tracking with YOLOv8 and SORT Library
Original photo by Mak from UnsplashIntroduction
Object Detection and Object Tracking is a quite useful thing for the modern world, especially when talking about solving real-life problems in fields associate with businesses (all of them) like agriculture, robotics, transportation and so on and so forth. This article is meant to make you familiar with the "Detection" and "Tracking" terms, and of course teach you how to implement this terms in code and visualizing.

---

What is the difference between "Detection" and "Tracking"?
When speaking about detecting an object, we limit ourselves to one frame, which means that the Objection Detection algorithm has to work with one picture, and ONLY detect a certain object/objects. Object Tracking is about the whole video. The algorithm needs to track one object across the entire video, thus making sure that this object is unique. This is the task for the tracker algorithms (DeepSort, Sort, Native CV2 Tracking Algorithms).
Note: More deeply object tracking here: https://medium.com/red-buffer/want-object-tracking-try-deep-sort-47cb38e84c89

In general, we talk about certain ID's that are assigned to each object or the Kalman Filter (prediction of future position of the object). 

---

Methods and Algorithms Used
Okay, since we understood what is detection and tracking, we can move on to the methodology and some advanced techniques.
YOLOv8 model 
YOLOv8 is a model based on YOLO (You Only Look Once), by Ultralytics. Generally, this model specialized in:
Detecting Objects
Segmentation
Classifying Objects

The YOLOv8 family of models is widely considered one of the best in the field, offering superior accuracy and faster performance. Its ease of use is attributed to the fact that it consists of five separate models, each catering to different needs, time constraints, and scopes.
Comparison of different versions of V8 [Source: Roboflow]In summary, models vary in terms of mean average precision (mAP) and the number of parameters they possess. Additionally, some models can be resource-intensive and exhibit differences in speed. For instance, the X model is considered the most advanced, leading to higher accuracy. However, it may result in slower rendering of videos or images. On the other hand, the Nano model (N) is the fastest option but sacrifices some accuracy.
Note: More about YOLOv8 here: https://medium.com/red-buffer/want-object-tracking-try-deep-sort-47cb38e84c89

SORT Algorithm
The SORT Algorithm, by Alex Bewley is a tracking algorithm for 2D multiple object tracking in video sequences. It serves as the foundation for other tracking algorithms like DeepSort. Due to its minimalist nature, it is straightforward to use and implement. Here, you can find more about this algorithm and you can even look into the source code.
Both YOLOv8 and SORT Algorithm are based on CNN (telling you this to move on explaining what the heck is CNN).

---

Math Behind
CNN structure [Source: Here ]So, CNN's or Convolutional Neural Networks are neural networks that are based on convolution layers and pooling layer. As it is written in A Comprehensive Guide to Convolutional Neural Networks - the ELI5 way, "The objective of the Convolution Operation is to extract the high-level features such as edges, from the input image", so to put it simply, convolution layers are extracting the most important features from the initial input. The pooling layer, on the other part, is the one that simplifies things or "is responsible for reducing the spatial size of the Convolved Feature". This process enables the machine to understand the features of the initial input. Therefore, we receive a complex feature learning process , where Convolutional and Pooling layers are stacked upon each other.
Since this is a very short explanation for CNN's, you can find more information about them here:
A Comprehensive Guide to Convolutional Neural Networks - the ELI5 way
Convolutional Neural Networks Explained (CNN Visualized)

---

Making it Real
Let's first of all, understand how to deal with the YOLOv8 model.
```
pip install ultralytics 
# !pip install ultralytics for JUPYTER Notebook
```
Then:
```
from ultralytics import YOLO
import cv2 #assuming you have opencv installed
MODEL = "yolov8x.pt" 
model = YOLO(MODEL) #making an instance of your chosen model
results = model("people.jpg",show=True) 
cv2.waitKey(0) # "0" will display the window infinitely until any keypress (in case of videos)
#waitKey(1) will display a frame for 1 ms
```
Results obtained from YOLOv8x versionNow, since you understood the basics, let's go to the true object detection and tracking. (Note that a lot of info. is took from this video, this guy is awesome).
Object  Tracking
```
import cv2
from ultralytics import YOLO
import math
import cvzone #CV2 but prettier and easier to use
from sort import * #importing all functions from SORT
#cap = cv2.VideoCapture(0) #for webcam
#cap.set(3,1280)
#cap.set(4,720)

cap = cv2.VideoCapture("data/los_angeles.mp4")
model =  YOLO("yolos/yolov8n.pt")
```
The cap variable will be the instance of the video that we are using and model, the instance of the YOLOv8 model.
```
classes = {0: 'person',
1: 'bicycle',
2: 'car', 
...
78: 'hair drier',
79: 'toothbrush'}

result_array = [classes[i] for i in range(len(classes))]
```
Initially, the classes that you get from YOLOv8 API, are float numbers or class id's. Of course, each number has a name class attached to it. It is simpler to make a dict for this and then, if you need, to transform it into array (got to lazy to make it manually).
```
l = [593,500,958,500] #line coordinates (explain it later)
while True:
    _, frame = cap.read() #reading the content of the video by frames
    results = model(frame,stream=True) #every frame goes through the YOLO model
    for r in results: 
        boxes = r.boxes #creating bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0] #extracting coordinates
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            w,h = x2-x1,y2-y1 #creating width and height
            cvzone.cornerRect(frame,(x1,y1,w,h),l=5, rt = 2, colorC=(255,215,0), colorR=(255,99,71))
            conf = math.ceil((box.conf[0]*100))/100 #confidence or accuracy of every bounding box
            cls = int(box.cls[0]) #class id (number)
```
That was the part where we detect every object. Now it's time to track and count every car on the road:
```
while True:
    _, frame = cap.read()
    results = model(frame,stream=True)
    detections = np.empty((0,5)) #making an empty array 
    for r in results: 
        boxes = r.boxes 
        for box in boxes:
            ''' rest of the code '''
            ins = np.array([x1,y1,x2,y2,conf]) #every object should be recorded like this
            detections = np.vstack((detections,ins)) # then stacked together in a common array
    
    tracks = tracker.update(detections) #sending our detections to the tracker func
    cv2.line(frame, (l[0],l[1]),(l[2],l[3]),color=(255,0,0),thickness=3) #line as a threshold
```
Now, I will create an array to store all of our detections. Next, I will send this array to the tracker function, where I will extract the unique IDs and bounding box coordinates (which are the same as the previous ones). The important detail is  the cv2.line instance: I am  generating a line using specific coordinates. If cars, identified by  certain IDs, traverse this line, the OVERALL count will increment. In  essence, we are establishing a car counter that operates according to  the car ID.
```
for result in tracks:
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        cvzone.putTextRect(frame,f'{result_array[cls]} {conf} id:{int(id)} ',(max(0,x1),max(35,y1-20)),scale=1, thickness=1, offset=3, colorR=(255,99,71))
        #.putTextRect is for putting a rectangle above bounding box
        cx,cy = x1+w//2, y1+h//2 #coordinates for the center of bb
        if l[0]<cx<l[2] and l[1]-10<cy<l[3]+10:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(frame, (l[0],l[1]),(l[2],l[3]),color=(127,255,212),thickness=5)
    cvzone.putTextRect(frame,f' Total Count: {len(totalCount)} ',(70,70),scale=2, thickness=1, offset=3, colorR=(255,99,71))
    m.write(frame)
    cv2.imshow("Image",frame)
    cv2.waitKey(1)
```
Now, onto the most interesting aspect: cx and cy are the coordinates for the center of the bounding box. By utilizing these values, we can determine whether the car has crossed the designated line or not (check out the code). If the car has indeed crossed the line, our next step involves verifying whether the ID assigned to this car is unique, thus indicating that the car has not crossed the line previously.
Results

Tracking and Counting of the vehiclesAs shown, I am displaying the bounding boxes here, along with the corresponding object class, confidence level, and unique ID. The total object count is visible in the upper left corner.

---

Bonus.
As a bonus, let me show you what YOLOv8 and some tracking tools can do:
Traffic Lights Tracker

Traffic Lights TrackerThe wonderful feature of OpenCV is that you have complete control over everything within the screen frame. This means you can customize the appearance of bounding boxes based on different situations and conditions. For instance, when dealing with traffic lights, you can adjust the color of the bounding box depending on whether the traffic light is green, red, or yellow. This kind of customization can greatly assist drivers and make their tasks easier.
100m Dash Sprinting Tracker

Runner TrackerAdditionally, we could do a Dash Sprinting Tracker for tracking athletes. With some masking magic and SORT lib, I could make a quite good tracker. For instance, I used YOLOv8l, the large model. Of course, it has some flaws in accuracy, but taking into consideration the quality of the video, the framing and the speed of the objects, the model has shown itself pretty good. 
ALL the code can be found here: https://github.com/grumpycatyo-collab/obj_tracking/tree/main

In the same repository, you can see how I trained the YOLOv8 model, based on a dataset extracted from Roboflow and also how I used masking to avoid unnecessary objects.

---

Conclusions
In summary, object tracking stands as a versatile tool with applications spanning statistical analysis and everyday problem-solving. Its ability to provide data-driven insights aids strategic decision-making in business, while also simplifying routine tasks and enhancing personal convenience. As we embrace this technology, we unlock innovative solutions that bridge the gap between advanced analytics and practical solutions, enriching both our professional endeavors and daily lives.
Thanks for reading!

---

Hey, thanks a lot for reading my first article. Hope you liked it and truly enjoyed it.
You can connect with me on LinkedIn and also see my GitHub profile.
Thx!!!

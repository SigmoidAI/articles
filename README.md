---

Video Object Tracking with Optical Flow and Yolo
Original photo by Mak from UnsplashIntroduction
Object Detection and Object Tracking is a quite useful thing for the modern world, especially when talking about solving real-life problems in fields associate with businesses (all of them) like agriculture, robotics, transportation and so on and so forth. This article is meant to make you familiar with the "Detection" and "Tracking" terms, and of course teach you how to implement this terms in code and visualizing.

---

What is the difference between "Detection" and "Tracking"?
When speaking about detecting an object, we limit ourselves to one frame, which means that the Objection Detection algorithm has to work with one picture, and ONLY detect a certain object/objects. Object Tracking is about the whole video. The algorithm needs to track one object across the entire video, thus making sure that this object is unique. This is the task for the tracker algorithms (DeepSort, Sort, Native CV2 Tracking Algorithms)[1].
In general, we talk about certain ID's that are assigned to each object,the Kalman Filter (prediction of future position of the object) or even the Optical Flow (for tracking the moving objects). 

---

Methods and Algorithms Used
Okay, since we understood what is detection and tracking, we can move on to the methodology and some advanced techniques.
Optical Flow
Optical flow is the pattern of apparent motion of image objects between two consecutive frames caused by the movement of object or camera. It is 2D vector field where each vector is a displacement vector showing the movement of points from first frame to second .
(Image Courtesy: Wikipedia article on Optical Flow).It shows a ball moving in 5 consecutive frames. The arrow shows its displacement vector[2]. Optical flow has many applications in areas like :
Structure from Motion
Video Compression
Video Stabilization

More about Optical Flow in the Making it real chapter.
YOLOv8 model 
YOLOv8 is a model based on YOLO (You Only Look Once), by Ultralytics. Generally, this model specialized in:
Detecting Objects
Segmentation
Classifying Objects

The YOLOv8 family of models is widely considered one of the best in the field, offering superior accuracy and faster performance. Its ease of use is attributed to the fact that it consists of five separate models, each catering to different needs, time constraints, and scopes.
Comparison of different versions of V8 [Source: Roboflow]In summary, models vary in terms of mean average precision (mAP) and the number of parameters they possess. Additionally, some models can be resource-intensive and exhibit differences in speed. For instance, the X model is considered the most advanced, leading to higher accuracy. However, it may result in slower rendering of videos or images. On the other hand, the Nano model (N) is the fastest option but sacrifices some accuracy [3].
SORT Algorithm
The SORT Algorithm, by Alex Bewley is a tracking algorithm for 2D multiple object tracking in video sequences. It serves as the foundation for other tracking algorithms like DeepSort. Due to its minimalist nature, it is straightforward to use and implement. Here, you can find more about this algorithm and you can even look into the source code.
Both YOLOv8 and SORT Algorithm are based on CNN (telling you this to move on explaining what the heck is CNN).

---

Math Behind
CNN's
CNN structure [Source: Here ]So, CNN's or Convolutional Neural Networks are neural networks that are based on convolution layers and pooling layer. As it is written in A Comprehensive Guide to Convolutional Neural Networks - the ELI5 way, "The objective of the Convolution Operation is to extract the high-level features such as edges, from the input image", so to put it simply, convolution layers are extracting the most important features from the initial input. The pooling layer, on the other part, is the one that simplifies things or "is responsible for reducing the spatial size of the Convolved Feature". This process enables the machine to understand the features of the initial input. Therefore, we receive a complex feature learning process , where Convolutional and Pooling layers are stacked upon each other[4].
Optical Flow Math
Pixel movement between two consecutive frames is referred to as optical flow. Either the camera is moving or the scene is moving, depending on the motion.
The fundamental goal of optical flow is to calculate the displacement vector of an object as a result of camera motions or the object's motion. In order to calculate the motion vectors of all image pixels or a sparse feature collection, our main objective is to determine their displacement.
If we were to use a picture to illustrate the optical flow issue, it would look somewhat like this:
Optical Flow functions by defining a dense vector field and is a key component in several computer vision and machine learning applications, including object tracking, object recognition, movement detection, and robot navigation. Each pixel is given its own displacement vector in this field, which aids in determining the direction and speed of each moving object's pixel in each frame of the input video sequence [5].

---

Making it Real
Object Tracking with YOLOv8 and SORT
Let's first of all, understand how to deal with the YOLOv8 model.
```
pip install ultralytics 
# !pip install ultralytics for JUPYTER Notebook
```

Then:

```
from ultralytics import YOLO

# Assuming you have opencv installed
import cv2 

MODEL = "yolov8x.pt" 
# Creating an instance of your chosen model
model = YOLO(MODEL) 

results = model("people.jpg",show=True) 
# "0" will display the window infinitely until any keypress (in case of videos)
# waitKey(1) will display a frame for 1 ms
cv2.waitKey(0)
```

Results obtained from YOLOv8x versionNow, since you understood the basics, let's go to the true object detection and tracking.

```
import cv2
from ultralytics import YOLO
import math

# CV2 but prettier and easier to use
import cvzone

# Importing all functions from SORT 
from sort import * 

# cap = cv2.VideoCapture(0) #for webcam
# cap.set(3,1280)
# cap.set(4,720)

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
# Line coordinates (explain it later)
l = [593,500,958,500] 

while True:
    
    # Reading the content of the video by frames
    _, frame = cap.read() 
    
    # Every frame goes through the YOLO model
    results = model(frame,stream=True) 
    for r in results: 
        
        # Creating bounding boxes
        boxes = r.boxes 
        
        for box in boxes:
            
            # Extracting coordinates
            x1, y1, x2, y2 = box.xyxy[0] 
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            
            # Creating instances width and height
            w,h = x2-x1,y2-y1 
            
            cvzone.cornerRect(frame,(x1,y1,w,h),l=5, rt = 2, colorC=(255,215,0), colorR=(255,99,71))
            
            # Confidence or accuracy of every bounding box
            conf = math.ceil((box.conf[0]*100))/100 
            
            # Class id (number)
            cls = int(box.cls[0])
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
Now, I will create an array to store all of our detections. Next, I will send this array to the tracker function, where I will extract the unique IDs and bounding box coordinates (which are the same as the previous ones). The important detail is  the cv2.line instance: I am  generating a line using specific coordinates. If cars, identified by  certain IDs, traverse this line, the OVERALL count will increment. In  essence, we are establishing a car counter that operates according to  the car ID[6].
```
for result in tracks:
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        
        #.putTextRect is for putting a rectangle above bounding box
        cvzone.putTextRect(frame,f'{result_array[cls]} {conf} id:{int(id)} ',(max(0,x1),max(35,y1-20)),scale=1, thickness=1, offset=3, colorR=(255,99,71))
        
        #Coordinates for the center of bb
        cx,cy = x1+w//2, y1+h//2 
        if l[0]<cx<l[2] and l[1]-10<cy<l[3]+10:
            if totalCount.count(id) == 0:
                #Counting every new car that crosses the line
                totalCount.append(id)
                #Line changes its color when a object crosses it
                cv2.line(frame, (l[0],l[1]),(l[2],l[3]),color=(127,255,212),thickness=5)
    #Rectangle to display the nr. of counted cars
    cvzone.putTextRect(frame,f' Total Count: {len(totalCount)} ',(70,70),scale=2, thickness=1, offset=3, colorR=(255,99,71))
    m.write(frame)
    cv2.imshow("Image",frame)
    cv2.waitKey(1)
```
Now, onto the most interesting aspect: cx and cy are the coordinates for the center of the bounding box. By utilizing these values, we can determine whether the car has crossed the designated line or not (check out the code). If the car has indeed crossed the line, our next step involves verifying whether the ID assigned to this car is unique, thus indicating that the car has not crossed the line previously.
Results

Tracking and Counting of the vehiclesAs shown, I am displaying the bounding boxes here, along with the corresponding object class, confidence level, and unique ID. The total object count is visible in the upper left corner.

---

Object Tracking with Optical Flow
The main difference between the SORT algorithms that track via assigning ID's and making connections with the previous frame and the current one, the Optical Flow procedure, is more about the Motion Estimation. In less words, it estimates what objects are moving and are estimating the director or the vector of this object.
There are two types of Optical Flow:
Sparse Optical Flow
Dense Optical Flow

While Dense Optical Flow, computes the optical flow for every pixel of the frame (which creates a image of only the objects that are moving), the Sparse Optical Flow, computes the flow vector of the main features of the objects. 
Now let's take a look at the code for these:
```
flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
```
First of all, we need to calculate the Optical Flow. We can do that via the Open CV library, that already has this algorithm. Of course, don't forget to put this function in a while loop, so the algorithm will calculate the Optical Flow continously (The process of reading the video and manipulating with it is the same as in the  Object Tracking with Yolov8 and SORT subchapter)
```
 def draw_flow(img, flow, step=16):
    # Get the height and width of the image
    h, w = img.shape[:2]
    
    # Create points on the image in a grid pattern
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    
    # Get flow directions at the grid points
    fx, fy = flow[y,x].T
    
    # Create lines to show the flow direction
    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    
    # Convert the grayscale image to color
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Draw lines to represent flow direction
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))
    
    # Draw small circles at the starting points of the lines
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)
    
    return img_bgr
```
This is the def responsible for the Sparse Optical Flow. We just need to extract the points from image, transorm them in a flow vector and after that simply visualise them.
```
def draw_hsv(flow):
    # Get the height and width of the flow matrix
    h, w = flow.shape[:2]
    
    # Separate the flow matrix into its x and y components
    fx, fy = flow[:,:,0], flow[:,:,1]

    # Calculate the angle of the flow vectors and convert to degrees
    ang = np.arctan2(fy, fx) + np.pi

    # Calculate the magnitude of the flow vectors
    v = np.sqrt(fx*fx + fy*fy)

    # Create an empty HSV image
    hsv = np.zeros((h, w, 3), np.uint8)
    
    # Set the hue channel of the HSV image based on the flow angle
    hsv[...,0] = ang * (180 / np.pi / 2)
    
    # Set the saturation channel of the HSV image to maximum
    hsv[...,1] = 255
    
    # Set the value (brightness) channel of the HSV image based on flow magnitude
    hsv[...,2] = np.minimum(v * 4, 255)
    
    # Convert the HSV image to BGR color space
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Return the final BGR image with flow visualization
    return bgr
```
This is the def responsible for Dense Optical Flow. Here, this function converts the flow vectors into hues, so the final result will display only the moving pixels[7].
Results

Sparse Optical Flow in actionAs seen, the Sparse Optical Flow, clearly visualizes the movement of every object and even the movement of the scene.

Dense Optical FlowSame video, but with Dense Optical Flow. As seen, only the vectors that are moving are seen.

---

Shortly about other Video Object Tracking Methods
MDNet
Multi-Domain Net or MDNet is a type of CNN-based object tracking algorithm which uses large-scale data for training. It is trained to learn shared representations of targets using annotated videos, i.e., it takes multiple annotated videos belonging to different domains. Its goal is to learn a wide range of variations and spatial relationships.
SiamMask
SiamMask aims to enhance the fully-convolutional Siamese network's offline training method. Siamese networks are convolutional neural networks that combine a cropped image with a wider search image to produce a dense spatial feature representation.
The Siamese network has a single output. It assesses how similar two input images are and evaluates whether or not the target objects are present in both images. This method, which involves supplementing the loss with a binary segmentation task, is especially effective for object tracking systems. SiamMask provides class-independent object segmentation masks and rotating bounding boxes at 35 frames per second when it is used online and only needs to be trained once.
GOTURN
Despite having access to a massive library of films for offline training, the majority of generic object trackers are trained from scratch online. Generic Object Tracking Using Regression Networks, or GOTURN, uses neural networks based on regression to track generic objects in a way that allows for performance improvement through training using annotated movies.
This tracking method makes use of a straightforward feed-forward network that doesn't require any online training and can operate at 100 frames per second during testing. The algorithm avoids overfitting by learning from both labeled video and a vast collection of photos. As generic objects move through various areas, the tracking system gains the ability to follow them in real-time [8].
Bonus.
As a bonus, let me show you some more tracking experiences:
Traffic Lights Tracker

Traffic Lights TrackerThe wonderful feature of OpenCV is that you have complete control over everything within the screen frame. This means you can customize the appearance of bounding boxes based on different situations and conditions. For instance, when dealing with traffic lights, you can adjust the color of the bounding box depending on whether the traffic light is green, red, or yellow. This kind of customization can greatly assist drivers and make their tasks easier.
100m Dash Sprinting Tracker

Runner TrackerAdditionally, we could do a Dash Sprinting Tracker for tracking athletes (yeah, the same video as in Optical Flow subchapter). With some masking magic and SORT lib, I could make a quite good tracker. For instance, I used YOLOv8l, the large model. Of course, it has some flaws in accuracy, but taking into consideration the quality of the video, the framing and the speed of the objects, the model has shown itself pretty good. 
ALL the code can be found here: https://github.com/grumpycatyo-collab/obj_tracking/tree/main

In the same repository, you can see how I trained the YOLOv8 model, based on a dataset extracted from Roboflow and also how I used masking to avoid unnecessary objects.

---

Conclusions
In summary, object tracking stands as a versatile tool with applications spanning statistical analysis and everyday problem-solving. Its ability to provide data-driven insights aids strategic decision-making in business, while also simplifying routine tasks and enhancing personal convenience. As we embrace this technology, we unlock innovative solutions that bridge the gap between advanced analytics and practical solutions, enriching both our professional endeavors and daily lives.
Thanks for reading!
References
[1] https://medium.com/red-buffer/want-object-tracking-try-deep-sort-47cb38e84c89
[2] https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
[3] https://medium.com/red-buffer/want-object-tracking-try-deep-sort-47cb38e84c89
[4] Convolutional Neural Networks Explained (CNN Visualized)
[5] https://datahacker.rs/002-advanced-computer-vision-motion-estimation-with-optical-flow/
[6] https://www.youtube.com/watch?v=WgPbbWmnXJ8&t=10512s&ab_channel=Murtaza%27sWorkshop-RoboticsandAI
[7] https://www.youtube.com/watch?v=hfXMw2dQO4E
[8] https://encord.com/blog/object-tracking-guide/#:~:text=DeepSORT%20is%20a%20well%2Dknown,is%20quite%20effective%20against%20occlusion.

---

Hey, thanks a lot for reading my first article. Hope you liked it and truly enjoyed it.
You can connect with me on LinkedIn and also see my GitHub profile.
Thx!!!

# Advancements in Object Detection: A Comprehensive Analysis of Two-Stage Models


## Abstract


The concept of object detection, an important task in Computer Vision, might not be well-known to the daily users of technology, but it has various applications in real-life problems. In this article, we will describe the magic behind object detection algorithms, primarily focusing on two-stage models. The research methodology consists of an analysis of two-stage models, highlighting their architecture. The research question is: what are the architectures, advantages, and limitations of two-stage models in object detection?

**Keywords:**  Object Detection, Two-Stage Models, R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN, CNN

## **Introduction**

Object detection is a Computer Vision problem that detects visual items of specific kinds (animals, humans) in digital pictures like photographs or video frames. The purpose of object detection is to create computational models that answer the most basic question that computer vision applications have: “What things are there and where are they?”¹

For humans, object detection is an effortless task, because it was there throughout their whole evolution. However, for machines, object detection is a complex challenge to be tackled and significant efforts have been done to develop such algorithms. Among these, an important role was played by two-stage models, that offer enhanced accuracy in object detection tasks compared to one-stage models. In the following paragraphs, we will discuss the architecture, functionality, and effectiveness of two-stage models in Computer Vision.

## What are two-stage models?

Two-stage models are Machine Learning models that locate objects in an image and predict their class in two steps. The first step is a regression problem while the second one is a classification problem².

The first step is a  **Regional Proposal Network (RPN)**  that scans the regions in an image and proposes them as candidates to be recognized as objects.

The second step is a  **Classifier**  that receives as input the regions proposed by the RPN and tries to classify them into an object class.

![](https://miro.medium.com/v2/resize:fit:1050/1*Z4s4WRGxRQsKW9R01xNqRA.png)

**Fig.1**  Two-stage detector architecture (Source:  [https://www.mdpi.com/1424-8220/23/21/8981](https://www.mdpi.com/1424-8220/23/21/8981))

The two-stage model pioneer is R-CNN, whose limitations were improved by the Fast R-CNN and Faster R-CNN models. Later Mask R-CNN was created for enhanced accuracy and faster predictions.

## What is R-CNN?

Region-based Convolutional Neural Network (R-CNN) is a type of deep learning architecture that works by proposing a region-of-interest (RoI), extracting features with CNN, and classifying those regions based on the extracted features.

More concretely, the R-CNN consists of the following four steps:

1.  **Region Proposals**: The first step involves generating region proposals using a method like Selective Search. This step identifies potential regions in an image that may contain objects. Typically, around 2000 region proposals are generated for each image.
2.  **Feature Extraction**: Each region proposal is then resized to a fixed size (e.g., 224×224 pixels) and passed through a pre-trained CNN to extract features. This CNN acts as a feature extractor, producing a fixed-length feature vector for each region.
3.  **Classification**: The extracted feature vectors are then fed into a set of Support Vector Machines (SVMs), where each SVM is trained to classify a specific object category. These SVMs determine the presence of objects within the proposed regions.
4.  **Bounding Box Regression**: To improve localization accuracy, a bounding box regressor is applied to refine the coordinates of the proposed regions. This step adjusts the bounding boxes to better fit the objects³.

![](https://miro.medium.com/v2/resize:fit:1050/0*m_gGw6vtHrBG2Ud_.png)

Fig.2 R-CNN Architecture (Source:  [https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e](https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e))

R-CNN can handle many object classes, but it is computationally expensive, which leads to resource and time consumption. Depending on the training manner, it might have to classify, for example, 2000 region proposals per image which leads to increased training time. Each part of the model requires training separately and cannot be paralleled. As well, the testing time of inference is approximately 50 seconds per image. These drawbacks led to the introduction of the next model.

## What is Fast R-CNN?

Fast R-CNN resulted as an improvement over R-CNN that addresses the computational inefficiency of the original method. It achieves this by using a single CNN to process the entire image and generate the ROIs rather than running the CNN on each ROI individually.

The Fast R-CNN algorithm includes:

1.  **Region proposal:**  In this step, a set of ROIs is generated using a method such as selective search or edge boxes.
2.  **Feature extraction:**  A CNN extracts features from the entire image.
3.  **ROI pooling:**  The extracted features are then used to compute a fixed-length feature vector for each ROI. This is done by dividing the ROI into a grid of cells and max-pooling the features within each cell.
4.  **Classification and bounding box regression:** The fixed-length feature vectors for each ROI are fed into two fully connected (FC) layers: one for classification and one for bounding box regression. The classification FC layer predicts the object’s class within the ROI, while the bounding box regression FC layer predicts the refined bounding box coordinates for the object⁴.

![](https://miro.medium.com/v2/resize:fit:1050/0*3wT_MFDeQfdtPs5y.png)

**Fig.3**  _Fast R-CNN architecture (Source:_ [https://www.geeksforgeeks.org/fast-r-cnn-ml/](https://www.geeksforgeeks.org/fast-r-cnn-ml/))

Since we don’t have to pass 2000 region proposals for each image and the ConvNet operation is done only once per image, this algorithm is more time efficient. Yet, it still uses the selective search algorithm which is slow and it takes around 2 seconds per image to detect objects, which sometimes does not work properly with large real-life datasets⁵. To solve this issue, the implementation of next model was necessary.

## What is Faster R-CNN?

Faster R-CNN is an extension of Fast R-CNN. It reduces the computational cost of object detection by using a single CNN to generate both the ROIs and the features for each ROI, rather than using a separate CNN for each task, as in Fast R-CNN.

The Faster R-CNN pipeline can be divided into four main steps:

1.  **Feature extraction:** A CNN extracts features from the entire image.
2.  **Region proposal:**  A set of ROIs is generated using a fully convolutional network (FCN) that processes the extracted features.
3.  **ROI pooling:**  The extracted features are then used to compute a fixed-length feature vector for each ROI using the same ROI pooling process as in Fast R-CNN.
4.  **Classification and bounding box regression:** The fixed-length feature vectors for each ROI are fed into two separate FC layers: one for classification and one for bounding box regression. The classification FC layer predicts the object’s class within the ROI, while the bounding box regression FC layer predicts the refined bounding box coordinates for the object⁴.

![](https://miro.medium.com/v2/resize:fit:1050/0*wysTH2z4Y-T3bYTt.jpeg)

**Fig.4**  _Faster R-CNN architecture (Source:_ [https://towardsdatascience.com/faster-r-cnn-for-object-detection-a-technical-summary-474c5b857b46](https://towardsdatascience.com/faster-r-cnn-for-object-detection-a-technical-summary-474c5b857b46))

Faster-RCNN has maintained its definite advantages in terms of detection accuracy. However, Faster-RCNN has some shortcomings such as it has not reached real-time detection. The way of getting region boxes before classification needs a huge amount of computations⁶. This limitation has led to another advanced approach, discussed next.

## What is Mask R-CNN?

Mask R-CNN is a deep learning model that combines object detection and instance segmentation. It is an extension of the Faster R-CNN architecture.

The key innovation of Mask R-CNN lies in its ability to perform pixel-wise instance segmentation alongside object detection. This is achieved through the addition of an extra “mask head” branch, which generates precise segmentation masks for each detected object⁷.

The architecture of Mask R-CNN is built upon the Faster R-CNN architecture, with the addition of an extra “mask head” branch for pixel-wise segmentation.

The overall architecture can be divided into several key stages:

1.  **Backbone and RPN network:**  These networks run once per image to give a set of region proposals. Region proposals are regions in the feature map that contain the object.
2.  **Classification and bounding box prediction**: Each proposed region can be of different sizes whereas fully connected layers in the networks always require a fixed size vector to make predictions. The size of these proposed regions is fixed by using either RoI pool (which is very similar to MaxPooling) or RoIAlign method⁸.

![](https://miro.medium.com/v2/resize:fit:938/0*MuXglFj2ScGpjnte.png)

**Fig.5**  Mask R-CNN architecture (Source:  [https://developers.arcgis.com/python/guide/how-maskrcnn-works/](https://developers.arcgis.com/python/guide/how-maskrcnn-works/))

The implementation of Mask R-CNN enables fine-grained pixel-level boundaries for accurate and detailed instance segmentation. As well it can also handle overlapping objects, which can make semantic segmentation models confuse pixels from different instances of the same class. However, this model is computationally expensive and requires memory resources which can increase with the number of classes and the resolution of masks⁹.

**Conclusion**

Two-stage models have come a long way and have had positive growth based on the lack of each ancestor. From R-CNN to Mask R-CNN, these models started from the same goal, to detect objects, and have been improved to become more efficient, faster, and to require fewer resources. By pursuing these future directions, two-stage object detection models can continue to evolve, providing efficient and accurate solutions across various fields.

**Bibliography**
[1]:  [https://www.tasq.ai/glossary/two-stage-detector/](https://www.tasq.ai/glossary/two-stage-detector/)
[2]:  [https://www.machinelearningatscale.com/p/computer-vision](https://www.machinelearningatscale.com/p/computer-vision)
[3]:  [https://www.geeksforgeeks.org/how-does-r-cnn-work-for-object-detection/](https://www.geeksforgeeks.org/how-does-r-cnn-work-for-object-detection/)
[4]:  [https://www.shiksha.com/online-courses/articles/object-detection-using-rcnn/](https://www.shiksha.com/online-courses/articles/object-detection-using-rcnn/)
[5]:  [https://medium.com/analytics-vidhya/object-detection-algorithms-r-cnn-vs-fast-r-cnn-vs-faster-r-cnn-3a7bbaad2c4a](https://medium.com/analytics-vidhya/object-detection-algorithms-r-cnn-vs-fast-r-cnn-vs-faster-r-cnn-3a7bbaad2c4a)
[6]:  [https://www.researchgate.net/publication/346764538_A_brief_review_and_challenges_of_object_detection_in_optical_remote_sensing_imagery](https://www.researchgate.net/publication/346764538_A_brief_review_and_challenges_of_object_detection_in_optical_remote_sensing_imagery)
[7]:  [https://blog.roboflow.com/mask-rcnn/](https://blog.roboflow.com/mask-rcnn/)
[8]:  [https://developers.arcgis.com/python/guide/how-maskrcnn-works/](https://developers.arcgis.com/python/guide/how-maskrcnn-works/)
[9]:  [https://www.linkedin.com/advice/3/how-does-mask-r-cnn-improve-segmentation-accuracy](https://www.linkedin.com/advice/3/how-does-mask-r-cnn-improve-segmentation-accuracy)

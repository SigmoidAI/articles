import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

image_path = "img/testing/img5.jpg" # Drumul către fișierul cu imagine
image = cv2.imread(image_path)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

bilateral_filter = cv2.bilateralFilter(gray, 11, 17, 17)

# Convertăm iarăși din BGR în RGb rezultatul primit
bilateral_filter_rgb = cv2.cvtColor(
    bilateral_filter, 
    cv2.COLOR_BGR2RGB
)

plt.imshow(bilateral_filter_rgb)
plt.title('Bilateral Filter') # Setăm numele la grafic
plt.show() # Afișăm fereastra cu grafic


image_edged = cv2.Canny(bilateral_filter, 30, 200)

image_edged_rgb = cv2.cvtColor(
    image_edged, 
    cv2.COLOR_BGR2RGB
)

plt.imshow(image_edged_rgb)
plt.title('Edge Detection')
plt.show()


keypoints = cv2.findContours(
    image_edged, 
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE
)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None # Locația la dreptunghi, dacă va fi detectat
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)

    if len(approx) == 4:
        location = approx
        break

print(location) # Afișăm locația detectată


mask = np.zeros(gray.shape, np.uint8)
image_masked = cv2.drawContours(mask, [location], 0, 255, -1)
image_masked = cv2.bitwise_and(image, image, mask=mask)

image_masked_rgb = cv2.cvtColor(
    image_masked, 
    cv2.COLOR_BGR2RGB
)

plt.imshow(image_masked_rgb)
plt.title('Masked Image')
plt.show()


x, y = np.where(mask == 255) # Extragem coordonatele care sunt de nuanța albă din imaginea mascată și le sepăram în x și y
x1, y1 = (np.min(x), np.min(y)) # Selectăm coordonatele cu valorea minimă pentru a detecta un colț a dreptupunghiului
x2, y2 = (np.max(x), np.max(y)) # Selectăm coordonatele cu valorea maximă pentru a detecta un colț a dreptupunghiului

image_cropped = gray[x1:x2+1, y1:y2+1]

image_cropped_rgb = cv2.cvtColor(
    image_cropped, 
    cv2.COLOR_BGR2RGB
)

plt.imshow(image_cropped_rgb)
plt.title('Cropped Image')
plt.show()
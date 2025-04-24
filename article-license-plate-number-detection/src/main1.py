import cv2
import matplotlib.pyplot as plt
import os
import imutils
import numpy as np

def main(img_path):

    if not os.path.exists(img_path):
        print(f"Error: File not found at {img_path}")
        return
    img = cv2.imread(img_path)
    fig, axs = plt.subplots(3, 2, figsize=(8, 5))
    fig.canvas.manager.set_window_title('License Plate Number Detection') 

    axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    axs[0, 1].imshow(cv2.cvtColor(bfilter, cv2.COLOR_BGR2RGB))
    axs[0, 1].set_title('Processed Image')
    axs[0, 1].axis('off')

    edged = cv2.Canny(bfilter, 30, 200)
    axs[1, 0].imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title('Edge Detection')
    axs[1, 0].axis('off')

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    else: return

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    #new_image = cv2.bitwise_and(img, img, mask=mask)
    
    axs[1, 1].imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    axs[1, 1].set_title('Masked Image')
    axs[1, 1].axis('off')

    x, y = np.where(mask == 255)
    x1, y1 = (np.min(x), np.min(y))
    x2, y2 = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]

    axs[2, 0].imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    axs[2, 0].set_title('Cropped  Image')
    axs[2, 0].axis('off')
    axs[2, 1].remove()

    plt.show()

if __name__ == '__main__':
    main(img_path='img/testing/img5.jpg')
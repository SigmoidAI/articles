import numpy as np
import cv2
import time

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


def main():
  
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    success, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        
        success, current_frame = cap.read()

      
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        start_time = time.time()
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prev_gray = gray
        end_time = time.time()

        fps = 1 / (end_time - start_time)
        print(f"{fps:.2f} FPS")

        cv2.imshow('Optical Flow', draw_flow(gray, flow))
        cv2.imshow('Optical Flow HSV', draw_hsv(flow))

       
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
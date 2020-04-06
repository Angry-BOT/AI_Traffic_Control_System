import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('H:/Projects/AITrafficControlSystem/SampleVid/traffic_video.mp4')
while True:
    ret, lane_image = cap.read()
    gray_lane_image = cv2.cvtColor(lane_image,cv2.COLOR_BGR2GRAY)
    #remove gaussian noise
    gaussian_lane_image=cv2.GaussianBlur(gray_lane_image,(3,3),0)
    #edge and line detection
    canny_lane_image=cv2.Canny(gaussian_lane_image,50,150)

    #simiulation so far
    cv2.imshow('Mat',canny_lane_image)
    plt.show()

    #we need to consider the region of interest , coordinates of a lane
    points = np.array([[280,50],[150,700],[450,700],[325,50]])
    #Cropping the bounding rectangle
    lane = cv2.boundingRect(points)
    x,y,w,h = lane
    croped_lane = lane_image[y:y+h, x:x+w]

    #making a mask
    points = points - points.min(axis=0)
    mask_lane = np.zeros(croped_lane.shape[:2], np.uint8)
    x=cv2.drawContours(mask_lane, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)

    #bit-op
    result_lane1 = cv2.bitwise_and(croped_lane, croped_lane, mask=mask_lane)

    #add the white background
    background = np.ones_like(croped_lane, np.uint8)*255
    cv2.bitwise_not(background, background, mask=mask_lane)
    result_lane2 = background + result_lane1

    #simiulation so far
    cv2.imshow('result',result_lane1)
    plt.show()

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
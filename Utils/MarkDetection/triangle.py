import cv2
import numpy as np

cap = cv2.VideoCapture('numLineData/IMG_1760.MOV')
ret, img = cap.read()

#this is a pixel in the triangle
print(img[530, 250])

triangle_green = np.uint8([[[134,193,160]]])
hsv_tri_green = cv2.cvtColor(triangle_green, cv2.COLOR_BGR2HSV)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

radius = 10
hue = 47

low_green = np.array([hue-radius, 20,20])
up_green = np.array([hue + radius,255,255])
mask = cv2.inRange(hsv, low_green, up_green)

cv2.imshow("mask",mask)

res = cv2.bitwise_and(img,img, mask= mask)
cv2.circle(res, center = (250,530), radius = 2, color = (255,0,255), thickness = 10 )
cv2.imshow("composite ",res)

cv2.waitKey()
cv2.destroyAllWindows()
quit()



    

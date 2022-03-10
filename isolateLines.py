import numpy as np
import cv2

minv = 15
maxv = 105

img = cv2.imread('numLine.jpeg',0 )
edges = cv2.Canny(img,minv,maxv)

linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 500, 40)

print(linesP)


cdstP = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

for i in range(0, len(linesP)):
    l = linesP[i][0]
    starty = min(l[0],l[2])
    stopy = max(l[0],l[2])

    startx = min(l[1],l[3])
    stopx = max(l[1],l[3])

    crop = img[startx:stopx, starty:stopy]
    cv2.imshow("Crop numba " + str(i), crop)
    cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

cv2.imshow("original", cdstP)


cv2.waitKey(0)
for i in range (0,12):
    cv2.destroyAllWindows()


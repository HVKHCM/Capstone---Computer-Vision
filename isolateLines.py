import numpy as np
import cv2

def cropToLine(img, l, bufferSize=0):
    length = np.sqrt((l[0]-l[2])**2 + (l[1]-l[3])**2)
    theta = np.arctan((l[1]-l[3])/(l[0]-l[2]))
    print(length, theta)
    (h,w) = img.shape[:2]

    lineCenter = ((l[0]+l[2])//2, (l[1]+l[3])//2)
    M = cv2.getRotationMatrix2D(lineCenter, theta/np.pi * 180, 1.0)
    rotated = cv2.warpAffine(img, M, (w,h))

    x1 = int(max(0, lineCenter[0]- length//2-bufferSize))
    x2 = int(min(w, lineCenter[0]+ length//2+bufferSize))

    y1 = int(max(0,lineCenter[1]-10- bufferSize))
    y2 = int(min(h,lineCenter[1]+10 + bufferSize))


    return rotated[y1:y2,  x1:x2 ]


minv = 15
maxv = 105

img = cv2.imread('numLine.jpeg',0 )
edges = cv2.Canny(img,minv,maxv)

linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 500, 40)

print(linesP)


cdstP = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

for i in range(0, len(linesP)):
    l = linesP[i][0]
    cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
    cv2.imshow("reoriented" + str(i), cropToLine(img, l,10))

cv2.imshow("original", cdstP)


cv2.waitKey(0)
for i in range (0,12):
    cv2.destroyAllWindows()


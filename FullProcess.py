from imgFunctions import *
from lineGeom import *
from tester import *

cap = cv2.VideoCapture('numLineData/IMG_1773.MOV')

if (cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        quit()
    frame = 1
display = img.copy()


while True:
    #cv2.imshow("window",current)
    # key capture every 1ms
    ch = cv2.waitKey(10)
    if ch & 0xFF == ord('q'):
        break
    elif ch & 0xFF == ord('d'):
        if (cap.isOpened()):
            ret, img = cap.read()
            frame +=1
            if not ret:
                break
        else: 
            break
        display = cv2.putText(img, f"{frame}", (0,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    elif ch & 0xFF == ord('p'):
        display = process(img)

    cv2.imshow("window", display)




    # When everything done, release the video capture object
cap.release()


    # Closes all the frames
for i in range(0,12):
    cv2.destroyAllWindows()
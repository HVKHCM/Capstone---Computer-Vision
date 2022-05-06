import cv2
import mediapipe
import scipy.spatial
 
drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands
distanceModule = scipy.spatial.distance
 
capture = cv2.VideoCapture(0)
 
frameWidth = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
 
circleCenter = (round(frameWidth/2), round(frameHeight/2))
circleRadius = 10

def slope(x1,y1,x2,y2):
    ###finding slope
    if x2!=x1:
        return((y2-y1)/(x2-x1))
    else:
        return 'NA'

def drawLine(image,x1,y1,x2,y2):
    m=slope(x1,y1,x2,y2)
    h,w=image.shape[:2]
    if m!='NA':
        ### here we are essentially extending the line to x=0 and x=width
        ### and calculating the y associated with it
        ##starting point
        px=0
        py=-(x1-0)*m+y1
        ##ending point
        qx=w
        qy=-(x2-w)*m+y2
    else:
    ### if slope is zero, draw a line with x=x1 and y=0 and y=height
        px,py=x1,0
        qx,qy=x1,h
    cv2.line(image, (int(px), int(py)), (int(qx), int(qy)), (0, 255, 0), 2)

with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1) as hands:
    while (True):
        ret, frame = capture.read()
        if ret == False:
            continue
        frame = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        circleColor = (0, 0, 0)
        if results.multi_hand_landmarks != None:
            normalizedLandmark = results.multi_hand_landmarks[0].landmark[handsModule.HandLandmark.INDEX_FINGER_TIP]
            normalizedLandmark2 = results.multi_hand_landmarks[0].landmark[handsModule.HandLandmark.INDEX_FINGER_DIP]
            pixelCoordinatesLandmark = drawingModule._normalized_to_pixel_coordinates(normalizedLandmark.x,
                                                                                      normalizedLandmark.y,
                                                                                      frameWidth,
                                                                                      frameHeight)
            pixelCoordinatesLandmark2 = drawingModule._normalized_to_pixel_coordinates(normalizedLandmark2.x,
                                                                                      normalizedLandmark2.y,
                                                                                      frameWidth,
                                                                                      frameHeight)
            print(pixelCoordinatesLandmark[0])

            cv2.circle(frame, pixelCoordinatesLandmark, 2, (255,0,0), -1)
            cv2.circle(frame, pixelCoordinatesLandmark2, 2, (255,0,0), -1)
            if distanceModule.euclidean(pixelCoordinatesLandmark, circleCenter) < circleRadius:
                circleColor = (0,255,0)
            else:
                circleColor = (0,0,255)
        cv2.circle(frame, circleCenter, circleRadius, circleColor, -1)
        cv2.imshow('Test image', frame)
        if cv2.waitKey(1) == 27:
            break
cv2.destroyAllWindows()
capture.release()

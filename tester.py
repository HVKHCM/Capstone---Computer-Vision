from imgFunctions import *

def process(img):
    #extract expected ruler lines
    l1,l2 = rulerLines(img)
   
    #extend lines to edges of screen
    l1E = extend(l1,img)
    l2E = extend(l2,img)

    display = img.copy()

    #Use extended lines to look for the triangle
    spot = findTriangle(l1E,l2E,img)
    if spot != -1:
        cv2.circle(display, center = spot, radius = 30, color = (0,0,255), thickness = 8 )
        cv2.circle(display, center = spot, radius = 2, color = (0,0,0), thickness = 14 )
        cv2.circle(display, center = spot, radius = 2, color = (255,255,255), thickness = 8 )
        #Stretch original lines to triangle location, then run mark detection
        l1 = stretch(l1, spot)
        l2 = stretch(l2, spot)
    marks  = rulerMarks(l1,l2, img)
    if spot != -1:
        if blockDist(marks[0], spot) > blockDist(marks[1], spot):
            marks.reverse()




    for (x,y) in marks:
        cv2.circle(display, center = (x,y), radius = 2, color = (0,0,0), thickness = 14 )
        cv2.circle(display, center = (x,y), radius = 2, color = (255,255,255), thickness = 8 )
    
    fingertip = fingertip_coordinate(img)
    cv2.circle(display, center = fingertip, radius = 2, color = (0,0,0), thickness = 14 )
    dist = []
    for mark in marks:
        dist.append(lineLen((fingertip[0], fingertip[1], mark[0],mark[1])))
    closest = np.array(dist).argmin()
    print(closest + 2)
    cv2.circle(display, center = marks[closest], radius = 5, color = (255,0,255), thickness = 8 )
    display = cv2.putText(display, f"{closest +2}", marks[closest], cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    return display


if __name__ == "__main__":
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name

    # 1708 is the zoomed in one
    # 1709 has a hand and the full ruler
    # 1710 has random stuff in the way, includes distracting keyboard
    #60, 73, 74
    cap = cv2.VideoCapture('numLineData/IMG_1773.MOV')
    # Check if camera opened successfully

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
import cv2
import numpy as np

def lineLen(line):
    return np.sqrt((line[0]-line[2])**2 + (line[1]-line[3])**2)

def mb(l):
    return ((l[1]-l[3])/(l[0]-l[2]), -l[0]* (l[1]-l[3])/(l[0]-l[2]) + l[1]) 

def distApart(a,b):
    #only to be used on basically parallel lines
    b1 = mb(b)[1]
    m, b2 = mb(a)
    return abs(b2-b1)/np.sqrt(m**2 +1)

def parallelism(a,b):
    va = [a[0]-a[2], a[1]-a[3]]
    vb = [b[0]-b[2], b[1]-b[3]]
    prod = abs(np.dot(va,vb))
    return prod/lineLen(a)/lineLen(b) 

a = [0,0,10,1]
b = [20,18,30,18]

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('numLineData/IMG_1708.MOV')
# Check if camera opened successfully
if (cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        quit()

display = img.copy()

def process(img):
    #First, repeat canny alg until there are about 20 or so candidate lines. 
    width = img.shape[1]
    maxVal = 1000
    minVal = 700
    while True:
        edges = cv2.Canny(img, minVal, maxVal)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, (5, 5), iterations=1)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, width//3, width//30)
        if lines is not None and len(lines)>15 :
            break
        minVal *= 0.97
        maxVal *= 0.97
        maxVal = int(maxVal)
        minVal = int(minVal)
    
    #So we found a good few lines, concatenate to less than 10
    lines = lines[:15]
    lines = lines.squeeze()
    
    #init matrix to score line-bounding likelihood
    likely = np.zeros((len(lines)-1, len(lines)-1))
    for i, line1 in enumerate(lines):
        for j, line2 in enumerate(lines[i+1:]):
            init = parallelism(line1, line2)
            if init > 0.999:
                d = distApart(line1, line2)
                init += min(d/20, 1, 6 - d/50)
            #Likeliness value first checks if two lines are basically parallel, then if they are, 
            #additional points are added if they are a fair distance away, but not too far.
            likely[i][j] = init
            
    
    #extract the VERY best line
    i, j = np.unravel_index(likely.argmax(), likely.shape)
    
    #Consider making this a threshold thing. Like if likelihood over 1.5 mark them down.
    best = []
    for i in range(0, len(likely)):
        for j in range(0, len(likely)):
            if likely[i][j] >= 1.9:
                best.append(lines[i])
                best.append(lines[i+j+1])
    
    #display all lines, then best ones in green  
    display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    for l in lines:
        cv2.line(display, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
        
    for i, l in enumerate(best):
        cv2.line(display, (l[0], l[1]), (l[2], l[3]), ((i//2)* 510//len(best),255 - (i//2)* 510//len(best) ,0), 3, cv2.LINE_AA)
    
    return display

    #What else should be done:
    #run a few iterations of morph dilate
    #take the space between two matching lines.
    #a true ruler will have little noise between. Calc something like 100 random points inbetween the lines.
    #If the number of edge pieces is below some threshhold, we can deduce that it is legit. 

while True:


    #cv2.imshow("window",current)
    # key capture every 1ms
    ch = cv2.waitKey(10)
    if ch & 0xFF == ord('q'):
        break
    elif ch & 0xFF == ord('d'):
        if (cap.isOpened()):
            ret, img = cap.read()
            if not ret:
                break
        else: 
            break
        display = img
    elif ch & 0xFF == ord('p'):
        display = process(img)

    cv2.imshow("window", display)




# When everything done, release the video capture object
cap.release()


# Closes all the frames
for i in range(0,12):
    cv2.destroyAllWindows()
    
    
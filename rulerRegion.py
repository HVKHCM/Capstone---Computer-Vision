import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from skimage.draw import line

def discretize(l):
    x1, y1, x2, y2 = l
    return np.asarray(list(zip(*line(*(x1, y1 ), *(x2 , y2 )))))

def blockDist(p1,p2):
    return abs(p1[0]-p2[0]) + abs(p1[1] - p2[1])

def align(line1, line2):
    #The idea here is to be able to take any two nearly-parallel line segments and fit a rectangle to them. 
    #the geometry here is that we will take the endpoint of line1, then find the point on the extension of line2 
    #such that the vector between the points makes a right angle with line 1
    p1 = (line1[0], line1[1])
    p2 = (line1[2], line1[3])
    
    a,b = ab(line1)
    
    q1 = (line2[0], line2[1])
    q2 = (line2[2], line2[3])
    
    c,d = ab(line2)
    
    candidates1 = [p1,p2]
    candidates2 = [q1,q2]
    new1 = list()
    new2 = list()
    
    
    for p in candidates1:
        k = -(1 + d * p[1] + c * p[0]) / (c * a + d * b)
        new2.append( (int(p[0] + k * a), int(p[1]+ k * b)))
        
    for q in candidates2:
        k = -(1 + b * q[1] + a * q[0]) / (c * a + d * b)
        new1.append( (int(q[0] + k * c), int(q[1]+ k * d)))

    candidates1 = candidates1 + new1
    candidates2 = candidates2 + new2
    
    start1 = candidates1[0]
    end1 = candidates1[1]
    fard = blockDist(start1,end1)
    for i in range(0, 4):
        for j in range(i+1,4):
            d = blockDist(candidates1[i] , candidates1[j])
            if d > fard:
                start1 = candidates1[i]
                end1 = candidates1[j]
                fard = d
                
    start2 = candidates2[0]
    end2 = candidates2[1]
    fard = blockDist(start2 ,end2)
    for i in range(0, 4):
        for j in range(i+1,4):
            d = blockDist(candidates2[i] , candidates2[j])
            if d > fard:
                start2 = candidates2[i]
                end2 = candidates2[j]
                fard = d
                
    if blockDist(start1, start2) > blockDist(start1, end2):
        temp = start2
        start2 = end2
        end2 = temp            
    
    return [start1[0], start1[1], end1[0], end1[1]] , [start2[0], start2[1], end2[0], end2[1]]
    

def regionBetween(line1, line2, img):
    height = img.shape[0]
    width = img.shape[1]
    
    dx = int(0.5 * (line2[0] - line1[0]) + 0.5 * (line2[2] - line1[2]))
    dy = int(0.5 * (line2[1] - line1[1]) + 0.5 * (line2[3] - line1[3]))
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    strip_width = len(list(zip(*line(*(0, 0), *(dx, dy)))))
    line1_discrete = discretize(line1)
    img_roi = np.zeros((strip_width, line1_discrete.shape[0]), dtype=np.uint8)
    for idx, (x, y) in enumerate(line1_discrete):
        perpendicular_line_discrete = np.asarray(
            list(zip(*line(*(x, y), *(x + dx, y + dy))))
        )
        img_roi[:, idx] = img_gray[
            perpendicular_line_discrete[:, 1], perpendicular_line_discrete[:, 0]
        ]
    return img_roi

def normalizeGaps(peaks):
    #create array to store the gaps between marks
    gaps = np.subtract(peaks[1:], peaks[:-1])
    med = np.median(gaps)
    radius = med//10
    
    #create new array containing only the gaps with size close to the median
    normalGaps = gaps[abs(gaps - med) <= radius]

    newgaps = []
    for i, gapSize in enumerate(gaps):
        #if a gap is pretty close to a multiple of the average gap length, 
        #slice that fella up
        added = False
        for j in range(1,10):
            if abs(gapSize - j * med) < j*radius:
                for k in range(0, j):
                    newgaps.append(gapSize/j)
                added = True
                break
        if not added:
            newgaps.append(gapSize)
    newpeaks = [peaks[0]]
    for i, gapSize in enumerate(newgaps):
        newpeaks.append(newpeaks[i] + gapSize)
    newpeaks = np.array(newpeaks)  
    return newpeaks.astype(int)
    

def lineLen(line):
    return np.sqrt((line[0]-line[2])**2 + (line[1]-line[3])**2)

def midPoint(l):
    return [(l[0]+l[2])/2, (l[1]+l[3])/2]

def mb(l):
    #gives slope and y intercept
    return ((l[1]-l[3])/(l[0]-l[2]), -l[0]* (l[1]-l[3])/(l[0]-l[2]) + l[1]) 

def ab(l):
    #Gives standard form a,b for a given line 1. assume c = 1
    x1,y1,x2,y2 = l
    return (-(y1-y2)/(x2*y1-x1*y2), -(x2-x1)/(x2*y1-x1*y2))

def pointDist(p, l):
    #gives normal distance between a point p and line l.
    a, b = ab(l)
    x1, y1 = p
    
    return abs(a*x1 + b*y1 + 1)/np.sqrt(a**2+b**2)
    
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

def process(img):
    #KNOWN BUGS: sometimes tries to access pixels which are out of range. 
    # I've seen this happen during the display phase, involving line_discrete
    
    #First, repeat canny alg until there are about [candidates] or so candidate lines. 
    width = img.shape[1]
    maxVal = 1000
    minVal = 700
    
    candidates = 10
    while True:
        edges = cv2.Canny(img, minVal, maxVal)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, (5, 5), iterations=1)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, width//3, width//30)
        if lines is not None and len(lines)>candidates :
            break
        minVal *= 0.9
        maxVal *= 0.9
        maxVal = int(maxVal)
        minVal = int(minVal)
    
    #So we found a good few lines, concatenate to less than [candidates]
    lines = lines[:candidates]
    lines = lines.squeeze()
    
    #init matrix to score line-bounding likelihood
    likely = np.zeros((len(lines)-1, len(lines)-1))
    for i, line1 in enumerate(lines):
        for j, line2 in enumerate(lines[i+1:]):
            init = parallelism(line1, line2)
            if init > 0.999:
                #d = distApart(line1, line2)
                d = pointDist(midPoint(line1), line2)
                init += min(d/40, 1, 6 - d/50)
            #Likeliness value first checks if two lines are basically parallel, then if they are, 
            #additional points are added if they are a fair distance away, but not too far.
            likely[i][j] = init
            
    
    #extract the VERY best line
    i, j = np.unravel_index(likely.argmax(), likely.shape)
    veryBest = []
    veryBest.append(lines[i])
    veryBest.append(lines[i+j+1])
    
    
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
    
    for l in best:
        cv2.line(display, (l[0], l[1]), (l[2], l[3]), (255,0,0), 3, cv2.LINE_AA)
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    display = img.copy()
    
    
    veryBest = align(veryBest[0], veryBest[1])
    for l in veryBest:
        cv2.line(display, (l[0], l[1]), (l[2], l[3]), (0,255,0), 3, cv2.LINE_AA)
    #Below here I'm just gonna paste a bunch of khang code.  
    
    #We analyze territory between our best candidate lines.
    line1, line2 = veryBest 
    img_roi = regionBetween(line1, line2, img)
    
    #perhaps make this a dynamic thresholding image. 
    #_ , img_roi = cv2.threshold(img_roi,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    
    #The following chunk of code creates an array "peaks" which idicates which of the points of
    #line1_discrete are likely to be markings. 
    
    NOF_MARKERS = 100
    line1_discrete = discretize(line1)
    line2_discrete = discretize(line2)
    
    #this is a mean of the color of each column of the cropped image. a column containing a white mark will have a high value.
    roi_mean = np.mean(img_roi, axis=0)
    
    avg_ruler_color = np.mean(roi_mean)
    length = img_roi.shape[1]
    width = img_roi.shape[0]   
    
    
    #used when checking gap size between peaks. Perhaps we expect 20 pixels etwen marks, but we will allow peak checking
    #as soon as 15 pixels.
    ERROR_MARGIN = 0.75
    
    #a constant for the ratio of width to height on the standard numberline
    WIDTH_TO_HEIGHT = 1/75
    
    
    expected_portion_of_numline = min(1,  WIDTH_TO_HEIGHT * length /width)
    #add a multipliction by 2 when dealing with 2-wide numberlines
    expected_marks = NOF_MARKERS * expected_portion_of_numline
    d = length/expected_marks
    print(expected_marks)
    
    
    prom = 10
    peaks = []
    while len(peaks) < 0.7* expected_marks and prom >1:
        peaks, _ = find_peaks(roi_mean,prominence = prom, distance=d * ERROR_MARGIN)
        prom = 0.8 * prom
    
    
    betterPeaks = []
    
    #uncomment below for demo
    #frame 82 is a good example to test on
    #betterPeaks = normalizeGaps(peaks)
    
    for i, peak in enumerate(betterPeaks):
        cv2.line(display, (line1_discrete[peak,0], line1_discrete[peak,1]), (line2_discrete[peak,0], line2_discrete[peak,1]), (0,0,0), 3, cv2.LINE_AA)
        cv2.line(display, (line1_discrete[peak,0], line1_discrete[peak,1]), (line2_discrete[peak,0], line2_discrete[peak,1]), (255,0,0), 1, cv2.LINE_AA)

    #finally, display marks on the ruler. 
    for i, peak in enumerate(peaks):
        cv2.line(display, (line1_discrete[peak,0], line1_discrete[peak,1]), (line2_discrete[peak,0], line2_discrete[peak,1]), (0,0,0), 3, cv2.LINE_AA)
        cv2.line(display, (line1_discrete[peak,0], line1_discrete[peak,1]), (line2_discrete[peak,0], line2_discrete[peak,1]), (255,255,255), 1, cv2.LINE_AA)
        #below puts the marks on a cropped image of the ruler, if ya want
        #cv2.circle(img_roi, center = (t,10), radius = 2, color = (255,0,255), thickness = 3 )
    
    
    return display

    #What else should be done:
    #additional things to guarrantee that the region holds a ruler. maybe check if the markings are
    #have a common spacing. if no common spacing probably bad.
    #also we can do the probabilty thing but other stuff seems more fun atm.




pic = cv2.imread("numLine.jpeg")
pic = cv2.resize(pic,(0,0), fx =  1.2, fy = 1.2)


testPeaks = [86,  221,  287,  428,  498,  566,  634,  705,  774,  843,  910,  976, 1113, 1180,
 1249, 1317, 1389, 1459, 1678]

#print(normalizeGaps(testPeaks))
#quit()

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name

# 1708 is the zoomed in one
# 1709 has a hand and the full ruler
# 1710 has random stuff in the way, includes distracting keyboard

cap = cv2.VideoCapture('numLineData/IMG_1708.MOV')
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
    elif ch & 0xFF == ord('c'):
        display = cv2.Canny(img, 50,100)

    cv2.imshow("window", display)




# When everything done, release the video capture object
cap.release()


# Closes all the frames
for i in range(0,12):
    cv2.destroyAllWindows()
    
    

cv2.imshow("window", process(pic))
cv2.waitKey()
for i in range(0,12):
    cv2.destroyAllWindows()
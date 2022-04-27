from lineGeom import * 

def extend(l,img):
    #This function takes a segment and stretches it to the edge of the screen.
    
    if len(img.shape) ==3:
        h,w, _ = img.shape
    elif len(img.shape) ==2:
        h,w = img.shape
    else:
        return l
    a,b = ab(l)
    
    #we consider the 4 points of intersection with the provided lines and the following lines:
    #y = 0
    #x = 0
    #y = height
    #x = width
    #Of these 4 points, only two will be "legal, with 0<x<width and 0<y<height"
    
    candidates = []
    if a != 0:
        candidates.append( (int(-1/a), 0) )
        candidates.append( (int((-b*h-1)/a), h-1) )
    if b!=0:
        candidates.append((0, int(-1/b)))
        candidates.append((w-1, int((-a*w-1)/b)))
    final = []
    
    for (x,y) in candidates:
        if x < w and x>=0 and y<h and y >=0:
            final.append((x,y))
    if len(final) != 2:
        return l
    p1 = final[0]
    p2 = final[1]
    
    if blockDist(p1, (l1[0],l1[1])) > blockDist(p1, (l1[2],l1[3])):
        temp = p2
        p2 = p1
        p1 = temp     
    
    
    return p1[0], p1[1], p2[0],p2[1]

def regionBetween(line1, line2, img):
    
    dx = int(0.5 * (line2[0] - line1[0]) + 0.5 * (line2[2] - line1[2]))
    dy = int(0.5 * (line2[1] - line1[1]) + 0.5 * (line2[3] - line1[3]))
    
    dimensions = img.shape
    if len(dimensions) ==3:
        #color crop
        height,width,depth = dimensions
        strip_width = len(list(zip(*line(*(0, 0), *(dx, dy)))))
        line1_discrete = discretize(line1)
        img_roi = np.zeros((strip_width, line1_discrete.shape[0], depth), dtype=np.uint8)
        for idx, (x, y) in enumerate(line1_discrete):
            perpendicular_line_discrete = np.asarray(
                list(zip(*line(*(x, y), *(x + dx, y + dy))))
            )
            for i, (x,y) in enumerate(perpendicular_line_discrete):
                if x >=width: 
                    perpendicular_line_discrete[i, 0] = width-1
                if y>=height:
                    perpendicular_line_discrete[i, 1] = height-1
            img_roi[:, idx,:] = img[
                perpendicular_line_discrete[:, 1], perpendicular_line_discrete[:, 0], :
            ]
        return img_roi
    
    if len(dimensions) != 2:
        return img
    #Otherwise, we are cropping in black and white
    height,width = dimensions
    strip_width = len(list(zip(*line(*(0, 0), *(dx, dy)))))
    line1_discrete = discretize(line1)
    img_roi = np.zeros((strip_width, line1_discrete.shape[0]), dtype=np.uint8)
    for idx, (x, y) in enumerate(line1_discrete):
        perpendicular_line_discrete = np.asarray(
            list(zip(*line(*(x, y), *(x + dx, y + dy))))
        )
        for i, (x,y) in enumerate(perpendicular_line_discrete):
            if x >=width: 
                perpendicular_line_discrete[i, 0] = width-1
            if y>=height:
                perpendicular_line_discrete[i, 1] = height-1
        img_roi[:, idx] = img[
            perpendicular_line_discrete[:, 1], perpendicular_line_discrete[:, 0]
        ]
    return img_roi

def normalizeGaps(peaks):
    #create array to store the gaps between marks
    gaps = np.subtract(peaks[1:], peaks[:-1])
    med = np.median(gaps)
    radius = med//5
    
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

def rulerLines(img):
    #this will return 2 lines who hopefully bind a ruler. not necessarily the whole thing tho.
    #First, repeat canny alg until there are about [candidates] or so candidate lines. 
    width = img.shape[1]
    maxVal = 1000
    minVal = 700
    candidates = 10
    blur = cv2.GaussianBlur(img,(5,5),0)
    while True:
        edges = cv2.Canny(blur, minVal, maxVal)
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
    
    return align(veryBest[0], veryBest[1])

def rulerMarks(line1, line2, img):
    #We analyze territory between the given lines
    img_roi = regionBetween(line1, line2, img)
    if len(img_roi.shape) ==3:
        img_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
        
    NOF_MARKERS = 100
    line1_discrete = discretize(line1)
    line2_discrete = discretize(line2)
    #used when checking gap size between peaks. Perhaps we expect 20 pixels etwen marks, but we will allow peak checking
    #as soon as 15 pixels.
    ERROR_MARGIN = 0.75
    
    #this is a mean of the color of each column of the cropped image. a column containing a white mark will have a high value.
    roi_mean = np.mean(img_roi, axis=0)
    length = img_roi.shape[1]
    width = img_roi.shape[0]  
    
    #a constant for the ratio of width to height on the standard numberline
    WIDTH_TO_HEIGHT = 1/75
    
    
    expected_portion_of_numline = min(1,  WIDTH_TO_HEIGHT * length /width)
    #add a multipliction by 2 when dealing with 2-wide numberlines
    expected_marks = NOF_MARKERS * expected_portion_of_numline 
    d = length/expected_marks
    prom = 10
    peaks = []
    while len(peaks) < 0.7* expected_marks and prom >1:
        peaks, _ = find_peaks(roi_mean,prominence = prom, distance=d * ERROR_MARGIN)
        prom = 0.8 * prom
    
    betterPeaks = normalizeGaps(peaks)
    
    markCoordinates = []
    for peak in betterPeaks:
        markCoordinates.append(line1_discrete[peak,:])
    return markCoordinates


if __name__ == "__main__":
    cap = cv2.VideoCapture('numLineData/IMG_1760.MOV')
    ret, img = cap.read()
    
    l1,l2 = rulerLines(img)
    marks  = rulerMarks(l1,l2, img)
    
    for (x,y) in marks:
        cv2.circle(img, center = (x,y), radius = 2, color = (255,0,255), thickness = 10 )
    
    cv2.imshow("therer", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
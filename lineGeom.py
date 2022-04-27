import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from skimage.draw import line

def discretize(l):
    #gives a set of points that a given line segment contains
    x1, y1, x2, y2 = l
    return np.asarray(list(zip(*line(*(x1, y1 ), *(x2 , y2 )))))

def blockDist(p1,p2):
    #simple distance metric
    return abs(p1[0]-p2[0]) + abs(p1[1] - p2[1])

def lineLen(line):
    #simple euclidean distance
    return np.sqrt((line[0]-line[2])**2 + (line[1]-line[3])**2)

def midPoint(l):
    #returns the midpoint of a segment
    return [(l[0]+l[2])/2, (l[1]+l[3])/2]

def mb(l):
    #gives slope (m) and y intercept (b)
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


def align(line1, line2):
    #The idea is to be able to take any two nearly-parallel line segments and fit a rectangle to them. 
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
    
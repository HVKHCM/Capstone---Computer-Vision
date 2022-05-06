import numpy as np 
import cv2

#Program won't work unless you have an input image titled as the following:
img = cv2.imread('numLine.jpg',0 )
img = cv2.resize(img, (0,0), fx = 1, fy = 1)
current = img.copy()

cv2.imshow("OG", current)

point = (0,0)
pressed = False

def click(event, x, y, flags, param):
	global img, pressed
	update = False
	#Read in x and y mouse coords and take not eof which one is bigger. we scale down a bit for finer tuning
	smol = min(x,y)
	big = max(x,y)
	smol = smol/2
	big = big/2


	if event == cv2.EVENT_LBUTTONDOWN:
		current = cv2.Canny(img,smol,big)
		pressed = True
		update = True
		print(x, y)

	elif event == cv2.EVENT_MOUSEMOVE and pressed == True:
		current = cv2.Canny(img,smol,big)
		update = True

	elif event == cv2.EVENT_LBUTTONUP:
		pressed = False

	if update:
		cdstP = cv2.cvtColor(current, cv2.COLOR_GRAY2BGR)
		
		#Run a probabilistic hough transform on the image. last 2 params are of utmost importance
		#500 indicates that we are looking for lines 500 pixels long
		# 40 indicates that we will allow a line to be considered so long as no gaps of onger than 40 pixels exist within it.
		linesP = cv2.HoughLinesP(current, 1, np.pi / 180, 50, None, 500, 40)

		if linesP is not None:
			for i in range(0, len(linesP)):
				l = linesP[i][0]
				cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
		
		text = str(smol) + ", " + str(big)
		cdstP = cv2.putText(cdstP, text, (0,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
		cv2.imshow("OG", cdstP)

cv2.namedWindow("OG")
cv2.setMouseCallback("OG", click)

while True:


	#cv2.imshow("OG",current)
	# key capture every 1ms
	ch = cv2.waitKey(1)
	if ch & 0xFF == ord('q'):
		break
	
cv2.destroyAllWindows()

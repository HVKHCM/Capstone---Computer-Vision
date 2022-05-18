from tester import *

if __name__ == "__main__":


    cap = cv2.VideoCapture('numLineData/IMG_1774.MOV')
    if (cap.isOpened()):
        ret, img = cap.read()
        if not ret:
            quit()
        frame = 1
    else:
        quit()
    display = img.copy()
    h,w,l = img.shape
    size = (w,h)

    out = cv2.VideoWriter('results.mov', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    while cap.isOpened():
        ret, img = cap.read()
        frame +=1
        if not ret:
            break
        display = process(img)
        display = cv2.putText(display, f"{frame}", (0,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        out.write(display)
    out.release()
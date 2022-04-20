import cv2
import mediapipe as mp
import scipy.spatial
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from skimage.draw import line
import math

NOF_MARKERS = 60
 
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles
distanceModule = scipy.spatial.distance

img = cv2.imread("example6.jpg",1)
cv2.waitKey(0)
with mp_hands.Hands(static_image_mode=True, model_complexity=1, min_detection_confidence=0.1, min_tracking_confidence=0.3, max_num_hands=1) as hands:
    #image = cv2.flip(img, 1)
    image = img
    result = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_height, image_width, _ = image.shape
    copied_image = image.copy()
    print('Handedness:', result.multi_handedness)
    for hand_landmarks in result.multi_hand_landmarks:
        finger_tip_coordinate_orig = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        finger_tip_coordinate = mp_drawing._normalized_to_pixel_coordinates(finger_tip_coordinate_orig.x,
                                                                                      finger_tip_coordinate_orig.y,
                                                                                      image_width,
                                                                                      image_height)
# distance between two point:

def distance(pt1, pt2):
    distance = math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
    return distance

# Find smallest value
def find_min(dist_list):
    smallest = dist_list[0]
    smallest_idx = 0
    for i in range(len(dist_list)):
        if dist_list[i] <= smallest:
            smallest_idx = i
            smallest = dist_list[i]

    return smallest_idx

# Compare

def compare(anchor, theOne):
    if (anchor[0] < theOne[0]):
        return 1
    else:
        return 0

#segment

img_orig = img.copy()
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
fig, axs = plt.subplots(2, 2)


# Detect long lines in the image
img_edg = cv2.Canny(img, 50, 120)
img_edg = cv2.morphologyEx(img_edg, cv2.MORPH_CLOSE, (5, 5), iterations=1)
img_edg = cv2.cvtColor(img_edg, cv2.COLOR_GRAY2RGB)
axs[0, 0].set_title("Canny edges + morphology closed")
axs[0, 0].imshow(img_edg)
lines = cv2.HoughLinesP(
    img_edg[:, :, 0].copy(),
    rho=1,
    theta=np.pi / 360,
    threshold=70,
    minLineLength=1500,
    maxLineGap=40,
)

lines = lines.squeeze()
for x1, y1, x2, y2 in lines:
    cv2.line(img_edg, (x1, y1), (x2, y2), (255, 0, 0))
axs[0, 0].imshow(img_edg, aspect="auto")


def optimize_line_alignment(img_gray, line_end_points):
    # Shift endpoints to find optimal alignment with black line in the origial image
    opt_line_mean = 255
    x1, y1, x2, y2 = line_end_points
    for dx1 in range(-3, 4):
        for dy1 in range(-3, 4):
            for dx2 in range(-3, 4):
                for dy2 in range(-3, 4):
                    line_discrete = np.asarray(
                        list(zip(*line(*(x1 + dx1, y1 + dy1), *(x2 + dx2, y2 + dy2))))
                    )
                    line_pixel_values = img_gray[
                        line_discrete[:, 1], line_discrete[:, 0]
                    ]
                    line_mean = np.mean(line_pixel_values)
                    if line_mean < opt_line_mean:
                        opt_line_end_points = np.array(
                            [x1 + dx1, y1 + dy1, x2 + dx2, y2 + dy2]
                        )
                        opt_line_discrete = line_discrete
                        opt_line_mean = line_mean
    return opt_line_end_points, opt_line_discrete


# Optimize alignment for the 2 outermost lines
dx = np.mean(abs(lines[:, 2] - lines[:, 0]))
dy = np.mean(abs(lines[:, 3] - lines[:, 1]))
if dy > dx:
    lines = lines[np.argsort(lines[:, 0]), :]
else:
    lines = lines[np.argsort(lines[:, 1]), :]
line1, line1_discrete = optimize_line_alignment(img_gray, lines[5, :])
line2, line2_discrete = optimize_line_alignment(img_gray, lines[-1, :])
cv2.line(img, (line1[0], line1[1]), (line1[2], line1[3]), (255, 0, 0))
cv2.line(img, (line2[0], line2[1]), (line2[2], line2[3]), (255, 0, 0))
axs[0, 1].set_title("Edges of the strip")
axs[0, 1].imshow(img, aspect="auto")


# Take region of interest from image
dx = round(0.5 * (line2[0] - line1[0]) + 0.5 * (line2[2] - line1[2]))
dy = round(0.5 * (line2[1] - line1[1]) + 0.5 * (line2[3] - line1[3]))
strip_width = len(list(zip(*line(*(0, 0), *(dx, dy)))))
img_roi = np.zeros((strip_width, line1_discrete.shape[0]), dtype=np.uint8)
for idx, (x, y) in enumerate(line1_discrete):
    perpendicular_line_discrete = np.asarray(
        list(zip(*line(*(x, y), *(x + dx, y + dy))))
    )
    img_roi[:, idx] = img_gray[
        perpendicular_line_discrete[:, 1], perpendicular_line_discrete[:, 0]
    ]

axs[1, 0].set_title("Strip analysis")
axs[1, 0].imshow(img_roi, cmap="gray")
extra_ax = axs[1, 0].twinx()
roi_mean = np.mean(img_roi, axis=0)
extra_ax.plot(roi_mean, label="mean")
extra_ax.plot(np.min(roi_mean, axis=0), label="min")
plt.legend()

# Locate the markers within region of interest
black_bar = np.argmin(roi_mean)
length = np.max([img_roi.shape[1] - black_bar, black_bar])
if black_bar < img_roi.shape[1] / 2:
    roi_mean = np.append(roi_mean, 0)
    peaks, _ = find_peaks(roi_mean[black_bar:], prominence = 0, distance=length / NOF_MARKERS * 0.75)
    peaks = peaks + black_bar
else:
    roi_mean = np.insert(roi_mean, 0, 0)
    peaks, _ = find_peaks(roi_mean[:black_bar], prominence = 0, distance=length / NOF_MARKERS * 0.75)
    peaks = peaks - 1
extra_ax.vlines(
    peaks,
    extra_ax.get_ylim()[0],
    extra_ax.get_ylim()[1],
    colors="green",
    linestyles="dotted",
)
print(line1_discrete[peaks,0])
print(line1_discrete[peaks,1])


##print(finger_tip_coordinate)

mark_point = []
i = 0
j = 0
while i < len(line1_discrete[peaks,0]):
    mark_point.append([])
    mark_point[j].append(line1_discrete[peaks,0][i])
    mark_point[j].append(line1_discrete[peaks,1][i])
    i = i + 1
    j = j + 1



distance_lst = []
t = 0
for i in range(len(mark_point)):
    distance_lst.append(distance(finger_tip_coordinate, mark_point[i]))

theChosenOne = find_min(distance_lst)
print(theChosenOne)
comp_result = compare(finger_tip_coordinate, mark_point[theChosenOne])
if (comp_result == 0):
    output = theChosenOne
else:
    output = theChosenOne+1
print(distance_lst)
print("The mark that was pointed at is: ", output)

circleCenter = (int(finger_tip_coordinate[0]),int(finger_tip_coordinate[1]))
print(circleCenter)
cv2.circle(img_orig, circleCenter ,10, (0,0,255), thickness=-1)
##cv2.putText(img_orig, '{}'.format(output),(image_height,image_width))
for handLms in result.multi_hand_landmarks:
    mp_drawing.draw_landmarks(img_orig, handLms, mp_hands.HAND_CONNECTIONS)
axs[1, 1].set_title("Midpoints between markings")
axs[1, 1].imshow(img_orig, aspect="auto")
axs[1, 1].plot(line1_discrete[peaks, 0], line1_discrete[peaks, 1], "r+")
fig.show()

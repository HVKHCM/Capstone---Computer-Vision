import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from skimage.draw import line

NOF_MARKERS = 60

# Show input image
img = cv2.imread("sample.jpg")
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
    minLineLength=300,
    maxLineGap=15,
)
lines = lines.squeeze()
for x1, y1, x2, y2 in lines:
    cv2.line(img_edg, (x1, y1), (x2, y2), (255, 0, 0))
axs[0, 0].imshow(img_edg, aspect="auto")


def optimize_line_alignment(img_gray, line_end_points):
    # Shift endpoints to find optimal alignment with black line in the  origial image
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
line1, line1_discrete = optimize_line_alignment(img_gray, lines[0, :])
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
    peaks, _ = find_peaks(roi_mean[black_bar:], distance=length / NOF_MARKERS * 0.75)
    peaks = peaks + black_bar
else:
    roi_mean = np.insert(roi_mean, 0, 0)
    peaks, _ = find_peaks(roi_mean[:black_bar], distance=length / NOF_MARKERS * 0.75)
    peaks = peaks - 1
extra_ax.vlines(
    peaks,
    extra_ax.get_ylim()[0],
    extra_ax.get_ylim()[1],
    colors="green",
    linestyles="dotted",
)
axs[1, 1].set_title("Midpoints between markings")
axs[1, 1].imshow(img_orig, aspect="auto")
axs[1, 1].plot(line1_discrete[peaks, 0], line1_discrete[peaks, 1], "r+")
fig.show()

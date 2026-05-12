import cv2
import numpy as np

INPUT = "../mechanical_upload.png"

OUTPUT = "../outputs/layout_clean.png"

img = cv2.imread(INPUT)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------- THRESHOLD ----------

thresh = cv2.threshold(
    gray,
    210,
    255,
    cv2.THRESH_BINARY_INV
)[1]

# ---------- REMOVE SMALL NOISE ----------

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    thresh,
    8
)

clean = np.zeros_like(thresh)

MIN_AREA = 120

for i in range(1, num_labels):

    area = stats[i, cv2.CC_STAT_AREA]

    if area > MIN_AREA:

        clean[labels == i] = 255

# ---------- MORPH CLOSE ----------

kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (5,5)
)

clean = cv2.morphologyEx(
    clean,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=2
)

# ---------- DETECT LINES ----------

lines = cv2.HoughLinesP(
    clean,
    1,
    np.pi / 180,
    threshold=120,
    minLineLength=80,
    maxLineGap=15
)

canvas = np.full(
    (900,1800,3),
    (235,235,235),
    dtype=np.uint8
)

scale_x = 1800 / img.shape[1]
scale_y = 900 / img.shape[0]

kept = 0

if lines is not None:

    for l in lines:

        x1,y1,x2,y2 = l[0]

        dx = x2 - x1
        dy = y2 - y1

        angle = abs(np.degrees(np.arctan2(dy, dx)))

        horizontal = angle < 6 or angle > 174
        vertical = 84 < angle < 96

        if not horizontal and not vertical:
            continue

        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)

        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        cv2.line(
            canvas,
            (x1,y1),
            (x2,y2),
            (70,70,70),
            3
        )

        kept += 1

cv2.imwrite(OUTPUT, canvas)

print("DONE")
print("Lines kept:", kept)
print("Saved:", OUTPUT)
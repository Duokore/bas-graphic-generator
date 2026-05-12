import cv2
import numpy as np
import math

INPUT_IMAGE = "../mechanical_upload.png"

OUTPUT = "../outputs/filtered_walls.png"

MIN_LENGTH = 60

img = cv2.imread(INPUT_IMAGE)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 80, 180)

lines = cv2.HoughLinesP(
    edges,
    1,
    np.pi / 180,
    threshold=120,
    minLineLength=40,
    maxLineGap=8
)

canvas = np.full(
    (900, 1800, 3),
    (235,235,235),
    dtype=np.uint8
)

scale_x = 1800 / img.shape[1]
scale_y = 900 / img.shape[0]

kept = 0

if lines is not None:

    for l in lines:

        x1, y1, x2, y2 = l[0]

        dx = x2 - x1
        dy = y2 - y1

        length = math.hypot(dx, dy)

        if length < MIN_LENGTH:
            continue

        angle = abs(math.degrees(math.atan2(dy, dx)))

        # ONLY HORIZONTAL / VERTICAL

        horizontal = angle < 8 or angle > 172

        vertical = 82 < angle < 98

        if not horizontal and not vertical:
            continue

        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)

        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        cv2.line(
            canvas,
            (x1, y1),
            (x2, y2),
            (70,70,70),
            2
        )

        kept += 1

cv2.imwrite(OUTPUT, canvas)

print("DONE")
print("Walls kept:", kept)
print("Saved:", OUTPUT)
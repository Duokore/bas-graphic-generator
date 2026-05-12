import cv2
import numpy as np
import json

IMAGE_PATH = "mechanical_upload.png"

OUTPUT_JSON = "outputs/walls.json"
OUTPUT_DEBUG = "outputs/walls_debug.png"

MIN_WALL_LENGTH = 40

img = cv2.imread(IMAGE_PATH)

if img is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(
    gray,
    180,
    255,
    cv2.THRESH_BINARY_INV
)

kernel = np.ones((3, 3), np.uint8)

clean = cv2.morphologyEx(
    thresh,
    cv2.MORPH_OPEN,
    kernel
)

lines = cv2.HoughLinesP(
    clean,
    1,
    np.pi / 180,
    threshold=90,
    minLineLength=MIN_WALL_LENGTH,
    maxLineGap=10
)

walls = []
debug = img.copy()

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]

        dx = x2 - x1
        dy = y2 - y1

        length = ((dx * dx) + (dy * dy)) ** 0.5
        angle = abs(np.degrees(np.arctan2(dy, dx)))

        horizontal = angle < 12 or angle > 168
        vertical = 78 < angle < 102

        if not (horizontal or vertical):
            continue

        if length < MIN_WALL_LENGTH:
            continue

        walls.append({
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "length": round(float(length), 2),
            "type": "horizontal" if horizontal else "vertical"
        })

        cv2.line(
            debug,
            (x1, y1),
            (x2, y2),
            (180, 180, 180),
            2
        )

cv2.imwrite(OUTPUT_DEBUG, debug)

with open(OUTPUT_JSON, "w") as f:
    json.dump({"walls": walls}, f, indent=2)

print("DONE")
print("Walls:", len(walls))
print("Debug:", OUTPUT_DEBUG)
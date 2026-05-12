import cv2
import numpy as np
import json
import math

INPUT_IMAGE = "../mechanical_upload.png"

OUTPUT_IMAGE = "../outputs/topology_render.png"

CANVAS_W = 1800
CANVAS_H = 900

# ---------- COLORS ----------

BG = (235, 235, 235)

ROOM_FILL = (70, 70, 70)
ROOM_BORDER = (45, 45, 45)

DUCT_COLOR = (245, 245, 245)

VAV_COLOR = (120, 40, 20)

AHU_COLOR = (40, 140, 40)

# ---------- LOAD IMAGE ----------

img = cv2.imread(INPUT_IMAGE)

if img is None:
    print("ERROR loading image")
    exit()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# ---------- COLOR MASKS ----------

# RED = DUCTS

lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

duct_mask = mask_red1 + mask_red2

# BLUE = VAVS

lower_blue = np.array([90, 80, 50])
upper_blue = np.array([140, 255, 255])

vav_mask = cv2.inRange(hsv, lower_blue, upper_blue)

# GREEN = AHU

lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 255])

ahu_mask = cv2.inRange(hsv, lower_green, upper_green)

# ---------- CLEAN MASKS ----------

kernel = np.ones((5,5), np.uint8)

duct_mask = cv2.morphologyEx(
    duct_mask,
    cv2.MORPH_CLOSE,
    kernel
)

vav_mask = cv2.morphologyEx(
    vav_mask,
    cv2.MORPH_CLOSE,
    kernel
)

ahu_mask = cv2.morphologyEx(
    ahu_mask,
    cv2.MORPH_CLOSE,
    kernel
)

# ---------- CREATE CANVAS ----------

canvas = np.full(
    (CANVAS_H, CANVAS_W, 3),
    BG,
    dtype=np.uint8
)

# ---------- SCALE ----------

img_h, img_w = duct_mask.shape

scale = min(
    CANVAS_W / img_w,
    CANVAS_H / img_h
)

offset_x = 50
offset_y = 50

# ---------- DRAW ARCHITECTURE ----------

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

walls = cv2.Canny(gray, 80, 180)

lines = cv2.HoughLinesP(
    walls,
    1,
    np.pi / 180,
    threshold=120,
    minLineLength=40,
    maxLineGap=8
)

if lines is not None:

    for l in lines:

        x1, y1, x2, y2 = l[0]

        x1 = int(x1 * scale) + offset_x
        y1 = int(y1 * scale) + offset_y

        x2 = int(x2 * scale) + offset_x
        y2 = int(y2 * scale) + offset_y

        cv2.line(
            canvas,
            (x1, y1),
            (x2, y2),
            ROOM_BORDER,
            2
        )

# ---------- DRAW DUCTS ----------

contours, _ = cv2.findContours(
    duct_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

for cnt in contours:

    area = cv2.contourArea(cnt)

    if area < 120:
        continue

    x, y, w, h = cv2.boundingRect(cnt)

    x = int(x * scale) + offset_x
    y = int(y * scale) + offset_y

    w = int(w * scale)
    h = int(h * scale)

    cv2.rectangle(
        canvas,
        (x, y),
        (x + w, y + h),
        DUCT_COLOR,
        -1
    )

# ---------- DRAW VAVS ----------

contours, _ = cv2.findContours(
    vav_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

for cnt in contours:

    area = cv2.contourArea(cnt)

    if area < 100:
        continue

    x, y, w, h = cv2.boundingRect(cnt)

    x = int(x * scale) + offset_x
    y = int(y * scale) + offset_y

    w = int(w * scale)
    h = int(h *scale)

    cv2.rectangle(
        canvas,
        (x, y),
        (x + w, y + h),
        VAV_COLOR,
        -1
    )

# ---------- DRAW AHU ----------

contours, _ = cv2.findContours(
    ahu_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

for cnt in contours:

    area = cv2.contourArea(cnt)

    if area < 150:
        continue

    x, y, w, h = cv2.boundingRect(cnt)

    x = int(x * scale) + offset_x
    y = int(y * scale) + offset_y

    w = int(w * scale)
    h = int(h * scale)

    cv2.rectangle(
        canvas,
        (x, y),
        (x + w, y + h),
        AHU_COLOR,
        -1
    )

# ---------- SAVE ----------

cv2.imwrite(
    OUTPUT_IMAGE,
    canvas
)

print("DONE")
print("Saved:", OUTPUT_IMAGE)
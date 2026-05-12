import cv2
import numpy as np
import json
import math

IMAGE_PATH = "mechanical_upload.png"
OUTPUT_JSON = "outputs/color_hvac.json"
OUTPUT_DEBUG = "outputs/color_hvac_debug.png"

MIN_VAV_AREA = 40
MAX_VAV_AREA = 1200

MIN_AHU_AREA = 200

MAX_DISTANCE_TO_DUCT = 120

TARGET_VAVS = 9


def clean_mask(mask, iterations=2):
    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=iterations
    )

    mask = cv2.dilate(
        mask,
        kernel,
        iterations=1
    )

    return mask


def get_contours(mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contours


def contour_center(cnt):
    x, y, w, h = cv2.boundingRect(cnt)

    return (
        int(x + w / 2),
        int(y + h / 2)
    )


def distance_to_nearest_duct(center, ducts):
    cx, cy = center

    best = 999999

    for d in ducts:

        x = d["x"]
        y = d["y"]
        w = d["w"]
        h = d["h"]

        pts = [
            (x, y),
            (x + w, y),
            (x, y + h),
            (x + w, y + h),
            (x + w // 2, y + h // 2)
        ]

        for px, py in pts:

            dist = math.hypot(cx - px, cy - py)

            if dist < best:
                best = dist

    return best


img = cv2.imread(IMAGE_PATH)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# =====================================================
# RED DUCTS
# =====================================================

red1 = cv2.inRange(
    hsv,
    np.array([0, 60, 60]),
    np.array([10, 255, 255])
)

red2 = cv2.inRange(
    hsv,
    np.array([170, 60, 60]),
    np.array([180, 255, 255])
)

red_mask = cv2.bitwise_or(red1, red2)

red_mask = clean_mask(red_mask, 1)

ducts = []

for cnt in get_contours(red_mask):

    area = cv2.contourArea(cnt)

    if area < 40:
        continue

    x, y, w, h = cv2.boundingRect(cnt)

    ratio = max(w, h) / max(min(w, h), 1)

    if ratio > 2.2:

        ducts.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "area": round(float(area), 2)
        })

# =====================================================
# BLUE VAVS
# =====================================================

blue_mask = cv2.inRange(
    hsv,
    np.array([90, 60, 60]),
    np.array([140, 255, 255])
)

blue_mask = clean_mask(blue_mask, 1)

vavs = []

for cnt in get_contours(blue_mask):

    area = cv2.contourArea(cnt)

    if area < MIN_VAV_AREA:
        continue

    if area > MAX_VAV_AREA:
        continue

    x, y, w, h = cv2.boundingRect(cnt)

    ratio = w / max(h, 1)

    if ratio < 0.35 or ratio > 3.2:
        continue

    center = contour_center(cnt)

    dist = distance_to_nearest_duct(center, ducts)

    # VERY IMPORTANT FILTER
    if dist > MAX_DISTANCE_TO_DUCT:
        continue

    vavs.append({
        "x": int(x),
        "y": int(y),
        "w": int(w),
        "h": int(h),
        "area": round(float(area), 2),
        "distance_to_duct": round(float(dist), 2)
    })

# KEEP ONLY BEST 9
vavs = sorted(
    vavs,
    key=lambda item: (
        item["distance_to_duct"],
        -item["area"]
    )
)

vavs = vavs[:TARGET_VAVS]

# =====================================================
# GREEN AHU
# =====================================================

green_mask = cv2.inRange(
    hsv,
    np.array([40, 40, 40]),
    np.array([90, 255, 255])
)

green_mask = clean_mask(green_mask, 1)

ahus = []

for cnt in get_contours(green_mask):

    area = cv2.contourArea(cnt)

    if area < MIN_AHU_AREA:
        continue

    x, y, w, h = cv2.boundingRect(cnt)

    ahus.append({
        "x": int(x),
        "y": int(y),
        "w": int(w),
        "h": int(h),
        "area": round(float(area), 2)
    })

# KEEP ONLY BIGGEST AHU
ahus = sorted(
    ahus,
    key=lambda item: item["area"],
    reverse=True
)[:1]

# =====================================================
# DEBUG DRAW
# =====================================================

debug = img.copy()

# ducts
for d in ducts:

    cv2.rectangle(
        debug,
        (d["x"], d["y"]),
        (d["x"] + d["w"], d["y"] + d["h"]),
        (0, 0, 255),
        2
    )

# vavs
for i, v in enumerate(vavs, start=1):

    cv2.rectangle(
        debug,
        (v["x"], v["y"]),
        (v["x"] + v["w"], v["y"] + v["h"]),
        (255, 0, 0),
        3
    )

    cv2.putText(
        debug,
        f"VAV-{i}",
        (v["x"], v["y"] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 0),
        2
    )

# ahu
for i, a in enumerate(ahus, start=1):

    cv2.rectangle(
        debug,
        (a["x"], a["y"]),
        (a["x"] + a["w"], a["y"] + a["h"]),
        (0, 255, 0),
        3
    )

    cv2.putText(
        debug,
        f"AHU-{i}",
        (a["x"], a["y"] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

# =====================================================
# SAVE
# =====================================================

data = {
    "ducts": ducts,
    "vavs": vavs,
    "ahus": ahus
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(data, f, indent=2)

cv2.imwrite(OUTPUT_DEBUG, debug)

print("DONE")
print("Ducts:", len(ducts))
print("VAVs:", len(vavs))
print("AHUs:", len(ahus))
print("Debug:", OUTPUT_DEBUG)
import json
import cv2
import numpy as np
import math

INPUT_JSON = "outputs/walls.json"

OUTPUT_JSON = "outputs/walls_joined.json"
OUTPUT_DEBUG = "outputs/walls_joined.png"

CANVAS_W = 2200
CANVAS_H = 1200

JOIN_DISTANCE = 25


def load_walls():
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    return data["walls"]


def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


walls = load_walls()

new_walls = []

used = set()

for i, a in enumerate(walls):

    if i in used:
        continue

    ax1 = a["x1"]
    ay1 = a["y1"]
    ax2 = a["x2"]
    ay2 = a["y2"]

    merged = False

    for j, b in enumerate(walls):

        if i == j or j in used:
            continue

        bx1 = b["x1"]
        by1 = b["y1"]
        bx2 = b["x2"]
        by2 = b["y2"]

        da = distance(ax2, ay2, bx1, by1)

        if da < JOIN_DISTANCE:

            new_walls.append({
                "x1": ax1,
                "y1": ay1,
                "x2": bx2,
                "y2": by2
            })

            used.add(i)
            used.add(j)

            merged = True
            break

    if not merged:
        new_walls.append(a)

canvas = np.zeros(
    (CANVAS_H, CANVAS_W, 3),
    dtype=np.uint8
)

canvas[:] = (30, 30, 30)

for w in new_walls:

    cv2.line(
        canvas,
        (w["x1"], w["y1"]),
        (w["x2"], w["y2"]),
        (180, 180, 180),
        4
    )

cv2.imwrite(OUTPUT_DEBUG, canvas)

with open(OUTPUT_JSON, "w") as f:
    json.dump(
        {
            "walls": new_walls
        },
        f,
        indent=2
    )

print("DONE")
print("Original:", len(walls))
print("Joined:", len(new_walls))
print("Debug:", OUTPUT_DEBUG)
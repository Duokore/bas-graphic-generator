import json
import cv2
import numpy as np
import math

INPUT_JSON = "outputs/duct_centerlines.json"
OUTPUT_JSON = "outputs/duct_centerlines_healed.json"
OUTPUT_DEBUG = "outputs/duct_centerlines_healed_debug.png"

CANVAS_W = 2200
CANVAS_H = 1200

MAX_GAP = 35


def load_lines():
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    return data["centerlines"]


def distance(p1, p2):
    return math.hypot(
        p2[0] - p1[0],
        p2[1] - p1[1]
    )


def angle(line):
    dx = line["x2"] - line["x1"]
    dy = line["y2"] - line["y1"]

    return math.degrees(math.atan2(dy, dx))


def classify_direction(a):
    a = abs(a)

    if a < 20 or a > 160:
        return "horizontal"

    if 70 < a < 110:
        return "vertical"

    return "other"


def endpoints(line):
    return [
        (line["x1"], line["y1"]),
        (line["x2"], line["y2"])
    ]


def connect_lines(lines):
    healed = lines.copy()
    added = []

    for i, line1 in enumerate(lines):
        a1 = angle(line1)
        dir1 = classify_direction(a1)

        if dir1 == "other":
            continue

        for j, line2 in enumerate(lines):
            if i == j:
                continue

            a2 = angle(line2)
            dir2 = classify_direction(a2)

            if dir1 != dir2:
                continue

            if dir2 == "other":
                continue

            pts1 = endpoints(line1)
            pts2 = endpoints(line2)

            for p1 in pts1:
                for p2 in pts2:
                    d = distance(p1, p2)

                    if d < MAX_GAP:
                        added.append({
                            "x1": int(p1[0]),
                            "y1": int(p1[1]),
                            "x2": int(p2[0]),
                            "y2": int(p2[1]),
                            "length": round(float(d), 2),
                            "healed": True
                        })

    healed.extend(added)

    return healed


def draw_debug(lines):
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    for line in lines:
        color = (0, 255, 0)

        if line.get("healed"):
            color = (0, 255, 255)

        cv2.line(
            canvas,
            (line["x1"], line["y1"]),
            (line["x2"], line["y2"]),
            color,
            2
        )

    cv2.imwrite(OUTPUT_DEBUG, canvas)


def main():
    lines = load_lines()
    healed = connect_lines(lines)

    data = {
        "total_lines": len(healed),
        "original_lines": len(lines),
        "healed_connections": len(healed) - len(lines),
        "lines": healed
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=2)

    draw_debug(healed)

    print("DONE")
    print("Original:", len(lines))
    print("Healed total:", len(healed))
    print("New connections:", len(healed) - len(lines))
    print("Debug:", OUTPUT_DEBUG)


if __name__ == "__main__":
    main()
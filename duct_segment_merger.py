import json
import cv2
import numpy as np
import math

INPUT_JSON = "outputs/duct_centerlines_healed.json"

OUTPUT_JSON = "outputs/duct_merged.json"
OUTPUT_DEBUG = "outputs/duct_merged_debug.png"

CANVAS_W = 2200
CANVAS_H = 1200

ANGLE_THRESHOLD = 10
DIST_THRESHOLD = 20


def load_lines():
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    return data["lines"]


def line_angle(line):
    dx = line["x2"] - line["x1"]
    dy = line["y2"] - line["y1"]

    return math.degrees(math.atan2(dy, dx))


def point_distance(a, b):
    return math.hypot(
        a[0] - b[0],
        a[1] - b[1]
    )


def are_collinear(a1, a2):
    ang1 = line_angle(a1)
    ang2 = line_angle(a2)

    return abs(ang1 - ang2) < ANGLE_THRESHOLD


def close_lines(a, b):
    pts_a = [
        (a["x1"], a["y1"]),
        (a["x2"], a["y2"])
    ]

    pts_b = [
        (b["x1"], b["y1"]),
        (b["x2"], b["y2"])
    ]

    for pa in pts_a:
        for pb in pts_b:
            if point_distance(pa, pb) < DIST_THRESHOLD:
                return True

    return False


def merge_two(a, b):
    pts = [
        (a["x1"], a["y1"]),
        (a["x2"], a["y2"]),
        (b["x1"], b["y1"]),
        (b["x2"], b["y2"])
    ]

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    horizontal = abs(a["y1"] - a["y2"]) < abs(a["x1"] - a["x2"])

    if horizontal:
        return {
            "x1": min(xs),
            "y1": int(sum(ys) / len(ys)),
            "x2": max(xs),
            "y2": int(sum(ys) / len(ys))
        }

    else:
        return {
            "x1": int(sum(xs) / len(xs)),
            "y1": min(ys),
            "x2": int(sum(xs) / len(xs)),
            "y2": max(ys)
        }


def merge_lines(lines):
    merged = []

    used = [False] * len(lines)

    for i in range(len(lines)):
        if used[i]:
            continue

        current = lines[i]

        for j in range(i + 1, len(lines)):
            if used[j]:
                continue

            other = lines[j]

            if are_collinear(current, other) and close_lines(current, other):
                current = merge_two(current, other)
                used[j] = True

        merged.append(current)

    return merged


def draw(lines):
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    for line in lines:
        cv2.line(
            canvas,
            (line["x1"], line["y1"]),
            (line["x2"], line["y2"]),
            (0, 255, 0),
            3
        )

    cv2.imwrite(OUTPUT_DEBUG, canvas)


def main():
    lines = load_lines()

    print("Original:", len(lines))

    merged = merge_lines(lines)

    print("Merged:", len(merged))

    with open(OUTPUT_JSON, "w") as f:
        json.dump(
            {
                "lines": merged
            },
            f,
            indent=2
        )

    draw(merged)

    print("Debug:", OUTPUT_DEBUG)


if __name__ == "__main__":
    main()
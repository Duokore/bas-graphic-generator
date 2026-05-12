import json
import cv2
import numpy as np
import math

INPUT_JSON = "outputs/duct_merged.json"

OUTPUT_JSON = "outputs/duct_connected.json"
OUTPUT_DEBUG = "outputs/duct_connected_debug.png"

CANVAS_W = 2200
CANVAS_H = 1200

CONNECT_DISTANCE = 35


def load_lines():
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    return data["lines"]


def point_distance(a, b):
    return math.hypot(
        a[0] - b[0],
        a[1] - b[1]
    )


def endpoints(line):
    return [
        (line["x1"], line["y1"]),
        (line["x2"], line["y2"])
    ]


def connect_lines(lines):
    new_lines = list(lines)

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):

            a = lines[i]
            b = lines[j]

            pts_a = endpoints(a)
            pts_b = endpoints(b)

            for pa in pts_a:
                for pb in pts_b:

                    dist = point_distance(pa, pb)

                    if dist < CONNECT_DISTANCE:

                        new_lines.append({
                            "x1": int(pa[0]),
                            "y1": int(pa[1]),
                            "x2": int(pb[0]),
                            "y2": int(pb[1])
                        })

    return new_lines


def draw(lines):
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    for line in lines:
        cv2.line(
            canvas,
            (line["x1"], line["y1"]),
            (line["x2"], line["y2"]),
            (0, 255, 255),
            2
        )

    cv2.imwrite(OUTPUT_DEBUG, canvas)


def main():
    lines = load_lines()

    print("Original:", len(lines))

    connected = connect_lines(lines)

    print("Connected:", len(connected))

    with open(OUTPUT_JSON, "w") as f:
        json.dump(
            {
                "lines": connected
            },
            f,
            indent=2
        )

    draw(connected)

    print("Debug:", OUTPUT_DEBUG)


if __name__ == "__main__":
    main()
import cv2
import numpy as np
import json
import os

SKELETON_PATH = "outputs/duct_skeleton.png"
OUTPUT_JSON = "outputs/duct_centerlines.json"
OUTPUT_DEBUG = "outputs/duct_centerlines_debug.png"

os.makedirs("outputs", exist_ok=True)


def extract_centerlines():
    img = cv2.imread(SKELETON_PATH, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Could not load {SKELETON_PATH}")

    _, binary = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)

    lines = cv2.HoughLinesP(
        binary,
        1,
        np.pi / 180,
        threshold=20,
        minLineLength=20,
        maxLineGap=10
    )

    centerlines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

            if length < 20:
                continue

            centerlines.append({
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "length": round(float(length), 2)
            })

    return img, centerlines


def draw_debug(img, centerlines):
    debug = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for line in centerlines:
        cv2.line(
            debug,
            (line["x1"], line["y1"]),
            (line["x2"], line["y2"]),
            (0, 255, 0),
            2
        )

        cv2.circle(debug, (line["x1"], line["y1"]), 3, (0, 0, 255), -1)
        cv2.circle(debug, (line["x2"], line["y2"]), 3, (255, 0, 0), -1)

    cv2.imwrite(OUTPUT_DEBUG, debug)


def main():
    img, centerlines = extract_centerlines()

    data = {
        "total_centerlines": len(centerlines),
        "centerlines": centerlines
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=2)

    draw_debug(img, centerlines)

    print("DONE")
    print("Centerlines:", len(centerlines))
    print("Debug:", OUTPUT_DEBUG)
    print("JSON:", OUTPUT_JSON)


if __name__ == "__main__":
    main()
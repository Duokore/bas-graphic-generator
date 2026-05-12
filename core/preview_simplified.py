import json
import cv2
import numpy as np

INPUT_JSON = "../outputs/duct_graph_simplified.json"

OUTPUT_IMAGE = "../outputs/simplified_preview.png"

CANVAS_W = 2200
CANVAS_H = 1200


def load_lines():
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    return data["lines"]


def main():

    canvas = np.zeros(
        (CANVAS_H, CANVAS_W, 3),
        dtype=np.uint8
    )

    lines = load_lines()

    for line in lines:

        x1 = line["x1"]
        y1 = line["y1"]

        x2 = line["x2"]
        y2 = line["y2"]

        line_type = line.get("type", "unknown")

        color = (255, 255, 255)

        if line_type == "horizontal":
            color = (255, 255, 255)

        elif line_type == "vertical":
            color = (180, 180, 180)

        cv2.line(
            canvas,
            (x1, y1),
            (x2, y2),
            color,
            5
        )

        cv2.circle(
            canvas,
            (x1, y1),
            4,
            (0, 255, 0),
            -1
        )

        cv2.circle(
            canvas,
            (x2, y2),
            4,
            (0, 0, 255),
            -1
        )

    cv2.imwrite(OUTPUT_IMAGE, canvas)

    print("DONE")
    print("Lines:", len(lines))
    print("Saved:", OUTPUT_IMAGE)


if __name__ == "__main__":
    main()
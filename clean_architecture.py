import json
import cv2
import numpy as np

INPUT_WALLS = "outputs/walls.json"

OUTPUT = "outputs/clean_walls.png"

CANVAS_W = 2200
CANVAS_H = 1200


def load_walls():
    with open(INPUT_WALLS, "r") as f:
        data = json.load(f)

    return data["walls"]


def main():

    canvas = np.zeros(
        (CANVAS_H, CANVAS_W, 3),
        dtype=np.uint8
    )

    canvas[:] = (35, 35, 35)

    walls = load_walls()

    for w in walls:

        cv2.line(
            canvas,
            (w["x1"], w["y1"]),
            (w["x2"], w["y2"]),
            (120, 120, 120),
            4
        )

    cv2.imwrite(OUTPUT, canvas)

    print("DONE")
    print("Walls:", len(walls))
    print("Render:", OUTPUT)


if __name__ == "__main__":
    main()
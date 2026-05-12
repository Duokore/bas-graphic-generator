import os
import cv2
import numpy as np


def extract_cad_geometry(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Could not load CAD source")

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    thresh = cv2.threshold(
        blur,
        180,
        255,
        cv2.THRESH_BINARY_INV
    )[1]

    lines = cv2.HoughLinesP(
        thresh,
        1,
        np.pi / 180,
        threshold=70,
        minLineLength=40,
        maxLineGap=10
    )

    geometry = []

    if lines is None:
        return geometry

    for l in lines:

        x1, y1, x2, y2 = l[0]

        geometry.append({

            "x1": int(x1),
            "y1": int(y1),

            "x2": int(x2),
            "y2": int(y2)
        })

    return geometry
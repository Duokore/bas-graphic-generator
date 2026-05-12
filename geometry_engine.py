import cv2
import numpy as np


def isolate_floorplan(gray):

    h, w = gray.shape

    mask = np.ones_like(gray) * 255

    # REMOVE RIGHT TITLE BLOCK
    cv2.rectangle(
        mask,
        (int(w * 0.90), 0),
        (w, h),
        0,
        -1
    )

    # REMOVE BOTTOM TABLES
    cv2.rectangle(
        mask,
        (0, int(h * 0.82)),
        (w, h),
        0,
        -1
    )

    # REMOVE LEFT BORDER
    cv2.rectangle(
        mask,
        (0, 0),
        (int(w * 0.05), h),
        0,
        -1
    )

    cleaned = cv2.bitwise_and(gray, mask)

    return cleaned


def build_wall_mask(gray):

    gray = isolate_floorplan(gray)

    blur = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    thresh = cv2.threshold(
        blur,
        190,
        255,
        cv2.THRESH_BINARY_INV
    )[1]

    lines = cv2.HoughLinesP(
        thresh,
        1,
        np.pi / 180,
        threshold=70,
        minLineLength=55,
        maxLineGap=18
    )

    mask = np.zeros_like(gray)

    if lines is None:
        print("Walls detected: 0")
        return mask

    count = 0

    for l in lines:

        x1, y1, x2, y2 = l[0]

        dx = x2 - x1
        dy = y2 - y1

        length = np.hypot(dx, dy)

        if length < 55:
            continue

        angle = abs(
            np.degrees(
                np.arctan2(dy, dx)
            )
        )

        horizontal = angle < 10 or angle > 170
        vertical = 80 < angle < 100

        if not horizontal and not vertical:
            continue

        cv2.line(
            mask,
            (x1, y1),
            (x2, y2),
            255,
            10
        )

        count += 1

    print("Walls detected:", count)

    mask = cv2.dilate(
        mask,
        np.ones((5, 5), np.uint8),
        iterations=1
    )

    return mask


def draw_extruded_walls(canvas, wall_mask, ox, oy):

    contours, _ = cv2.findContours(
        wall_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for c in contours:

        area = cv2.contourArea(c)

        if area < 120:
            continue

        x, y, w, h = cv2.boundingRect(c)

        # IGNORE MASSIVE BLOCKS
        if w > 800 or h > 450:
            continue

        # IGNORE TINY NOISE
        if w < 20 or h < 20:
            continue

        # SHADOW
        cv2.rectangle(
            canvas,
            (ox + x + 7, oy + y + 7),
            (ox + x + w + 7, oy + y + h + 7),
            (165, 165, 165),
            -1
        )

        # WALL BODY
        cv2.rectangle(
            canvas,
            (ox + x, oy + y),
            (ox + x + w, oy + y + h),
            (255, 255, 255),
            -1
        )

        # OUTLINE
        cv2.rectangle(
            canvas,
            (ox + x, oy + y),
            (ox + x + w, oy + y + h),
            (120, 120, 120),
            2
        )
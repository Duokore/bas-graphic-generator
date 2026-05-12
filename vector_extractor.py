import fitz
import cv2
import numpy as np
import json
import os

PDF_PATH = "uploads/mechanical_upload.pdf"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEBUG_IMAGE = os.path.join(OUTPUT_DIR, "vector_debug.png")
JSON_OUTPUT = os.path.join(OUTPUT_DIR, "vector_geometry.json")


def pdf_to_image(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
    img_data = np.frombuffer(pix.samples, dtype=np.uint8)

    if pix.alpha:
        img = img_data.reshape(pix.height, pix.width, 4)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = img_data.reshape(pix.height, pix.width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    doc.close()
    return img


def extract_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur,
        210,
        255,
        cv2.THRESH_BINARY_INV
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    cleaned = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    lines = cv2.HoughLinesP(
        cleaned,
        1,
        np.pi / 180,
        threshold=120,
        minLineLength=120,
        maxLineGap=18
    )

    extracted = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            dx = x2 - x1
            dy = y2 - y1

            length = float((dx * dx + dy * dy) ** 0.5)

            if length < 120:
                continue

            # =========================
            # REGION FILTERING
            # =========================

            # Ignore title block right side
            if x1 > img.shape[1] * 0.78:
                continue

            # Ignore schedules / notes bottom
            if y1 > img.shape[0] * 0.80:
                continue

            # Ignore top legends / headers
            if y1 < img.shape[0] * 0.12:
                continue

            # Ignore very left border junk
            if x1 < img.shape[1] * 0.05:
                continue

            # Keep mostly horizontal / vertical lines
            angle = abs(np.degrees(np.arctan2(dy, dx)))

            if not (
                angle < 15 or
                angle > 75
            ):
                continue

            extracted.append({
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "length": round(length, 2),
                "angle": round(angle, 2)
            })

    return extracted


def build_duct_network(lines):
    network = []
    used = set()

    CONNECT_DISTANCE = 25

    for i, line in enumerate(lines):
        if i in used:
            continue

        path = [
            [line["x1"], line["y1"]],
            [line["x2"], line["y2"]]
        ]

        used.add(i)

        extended = True

        while extended:
            extended = False

            end_x, end_y = path[-1]

            for j, other in enumerate(lines):
                if j in used:
                    continue

                pts = [
                    (other["x1"], other["y1"]),
                    (other["x2"], other["y2"])
                ]

                for pt in pts:
                    dist = (
                        (pt[0] - end_x) ** 2 +
                        (pt[1] - end_y) ** 2
                    ) ** 0.5

                    if dist < CONNECT_DISTANCE:
                        if pt == pts[0]:
                            next_pt = pts[1]
                        else:
                            next_pt = pts[0]

                        path.append([
                            next_pt[0],
                            next_pt[1]
                        ])

                        used.add(j)
                        extended = True
                        break

                if extended:
                    break

        if len(path) >= 2:
            network.append({
                "path": path,
                "segments": len(path)
            })

    return network


def draw_debug(img, lines, network):
    debug = img.copy()

    # Raw extracted lines = red
    for line in lines:
        cv2.line(
            debug,
            (line["x1"], line["y1"]),
            (line["x2"], line["y2"]),
            (0, 0, 255),
            2
        )

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 0),
        (0, 128, 255),
    ]

    # Connected duct networks = different colors
    for idx, duct in enumerate(network):
        color = colors[idx % len(colors)]
        path = duct["path"]

        for i in range(len(path) - 1):
            p1 = tuple(path[i])
            p2 = tuple(path[i + 1])

            cv2.line(
                debug,
                p1,
                p2,
                color,
                5
            )

            cv2.circle(
                debug,
                p1,
                6,
                color,
                -1
            )

            cv2.circle(
                debug,
                p2,
                6,
                color,
                -1
            )

    cv2.imwrite(DEBUG_IMAGE, debug)


def main():
    img = pdf_to_image(PDF_PATH)

    lines = extract_lines(img)

    network = build_duct_network(lines)

    data = {
        "total_lines": len(lines),
        "total_networks": len(network),
        "duct_networks": network,
        "raw_lines": lines
    }

    with open(JSON_OUTPUT, "w") as f:
        json.dump(data, f, indent=2)

    draw_debug(img, lines, network)

    print("DONE")
    print("Lines detected:", len(lines))
    print("Networks detected:", len(network))
    print("Debug image:", DEBUG_IMAGE)
    print("JSON:", JSON_OUTPUT)


if __name__ == "__main__":
    main()
import json
import cv2
import numpy as np
import os

INPUT_LINES = "outputs/duct_connected.json"
INPUT_HVAC = "outputs/color_hvac.json"
MANUAL_JSON = "manual_overrides.json"

OUTPUT = "outputs/bas_clean_render.png"

CANVAS_W = 2200
CANVAS_H = 1200


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_lines():
    data = load_json(INPUT_LINES)
    return data.get("lines", [])


def load_hvac():
    hvac = load_json(INPUT_HVAC)

    vavs = hvac.get("vavs", [])
    ahus = hvac.get("ahus", [])

    if os.path.exists(MANUAL_JSON):
        manual = load_json(MANUAL_JSON)

        manual_vavs = manual.get("vavs", [])
        manual_ahus = manual.get("ahus", [])

        if manual.get("use_manual_vavs", False) and len(manual_vavs) > 0:
            vavs = manual_vavs
            print("Using MANUAL VAV positions")
        else:
            print("Manual VAV empty, using auto VAVs")

        if manual.get("use_manual_ahu", False) and len(manual_ahus) > 0:
            ahus = manual_ahus
            print("Using MANUAL AHU position")
        else:
            print("Manual AHU empty, using auto AHU")

    return {
        "vavs": vavs,
        "ahus": ahus
    }


def draw_ducts(canvas, lines):
    for line in lines:
        cv2.line(
            canvas,
            (line["x1"], line["y1"]),
            (line["x2"], line["y2"]),
            (235, 235, 235),
            6
        )


def draw_vavs(canvas, vavs):
    for i, v in enumerate(vavs, start=1):
        x = int(v["x"])
        y = int(v["y"])
        w = int(v.get("w", 30))
        h = int(v.get("h", 30))

        cv2.rectangle(canvas, (x, y), (x + w, y + h), (180, 80, 20), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 120, 40), 2)

        name = v.get("name", f"VAV-{i}")

        cv2.putText(
            canvas,
            name,
            (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 140, 50),
            1
        )


def draw_ahu(canvas, ahus):
    for i, a in enumerate(ahus, start=1):
        x = int(a["x"])
        y = int(a["y"])
        w = int(a.get("w", 45))
        h = int(a.get("h", 45))

        cv2.rectangle(canvas, (x, y), (x + w, y + h), (40, 180, 40), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (120, 255, 120), 3)

        name = a.get("name", f"AHU-{i}")

        cv2.putText(
            canvas,
            name,
            (x - 10, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (120, 255, 120),
            2
        )


def main():
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    lines = load_lines()
    hvac = load_hvac()

    draw_ducts(canvas, lines)
    draw_vavs(canvas, hvac["vavs"])
    draw_ahu(canvas, hvac["ahus"])

    cv2.imwrite(OUTPUT, canvas)

    print("DONE")
    print("Lines:", len(lines))
    print("VAVs:", len(hvac["vavs"]))
    print("AHUs:", len(hvac["ahus"]))
    print("Render:", OUTPUT)


if __name__ == "__main__":
    main()
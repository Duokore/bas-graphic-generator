import cv2
import json

IMAGE_PATH = "mechanical_upload.png"
OUTPUT_JSON = "manual_overrides.json"

points = []
labels = [
    "VAV-1", "VAV-2", "VAV-3", "VAV-4", "VAV-5",
    "VAV-6", "VAV-7", "VAV-8", "VAV-9", "AHU-1"
]


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < len(labels):
            label = labels[len(points)]
            points.append({
                "name": label,
                "x": x - 15,
                "y": y - 15,
                "w": 30 if "VAV" in label else 45,
                "h": 30 if "VAV" in label else 45
            })
            print(f"{label}: x={x}, y={y}")


img = cv2.imread(IMAGE_PATH)

if img is None:
    raise FileNotFoundError(f"Could not load {IMAGE_PATH}")

display = img.copy()

cv2.namedWindow("Manual Picker", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Manual Picker", mouse_callback)

while True:
    view = display.copy()

    for idx, p in enumerate(points):
        color = (255, 0, 0)

        if "AHU" in p["name"]:
            color = (0, 255, 0)

        cv2.rectangle(
            view,
            (p["x"], p["y"]),
            (p["x"] + p["w"], p["y"] + p["h"]),
            color,
            3
        )

        cv2.putText(
            view,
            p["name"],
            (p["x"], p["y"] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    next_label = labels[len(points)] if len(points) < len(labels) else "DONE"

    cv2.putText(
        view,
        f"Click: {next_label} | Press S to save | Q to quit",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow("Manual Picker", view)

    key = cv2.waitKey(20) & 0xFF

    if key == ord("s"):
        data = {
            "use_manual_vavs": True,
            "use_manual_ahu": True,
            "vavs": [p for p in points if "VAV" in p["name"]],
            "ahus": [p for p in points if "AHU" in p["name"]]
        }

        with open(OUTPUT_JSON, "w") as f:
            json.dump(data, f, indent=2)

        print("SAVED:", OUTPUT_JSON)
        break

    if key == ord("q"):
        print("QUIT WITHOUT SAVING")
        break

cv2.destroyAllWindows()
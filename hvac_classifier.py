import json
import cv2
import numpy as np

INPUT_JSON = "outputs/vector_geometry.json"
INPUT_IMAGE = "uploads/mechanical_upload.png"

OUTPUT_JSON = "outputs/hvac_classified.json"
OUTPUT_DEBUG = "outputs/hvac_classified_debug.png"


def calculate_angle(p1, p2):

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    angle = abs(np.degrees(np.arctan2(dy, dx)))

    return angle


def classify_networks(networks):

    main_ducts = []
    branch_ducts = []
    possible_noise = []

    for duct in networks:

        path = duct["path"]
        segments = duct["segments"]

        total_length = 0
        horizontal_score = 0
        vertical_score = 0

        for i in range(len(path) - 1):

            p1 = path[i]
            p2 = path[i + 1]

            dist = (
                (p2[0] - p1[0]) ** 2 +
                (p2[1] - p1[1]) ** 2
            ) ** 0.5

            total_length += dist

            angle = calculate_angle(p1, p2)

            # Horizontal
            if angle < 20:
                horizontal_score += dist

            # Vertical
            elif angle > 70:
                vertical_score += dist

        duct["total_length"] = round(total_length, 2)

        # ====================================
        # MAIN DUCTS
        # ====================================

        if (
            total_length > 700 and
            horizontal_score > vertical_score * 1.5
        ):

            main_ducts.append(duct)

        # ====================================
        # BRANCH DUCTS
        # ====================================

        elif (
            total_length > 120 and
            vertical_score < 500
        ):

            branch_ducts.append(duct)

        # ====================================
        # NOISE
        # ====================================

        else:
            possible_noise.append(duct)

    return {
        "main_ducts": main_ducts,
        "branch_ducts": branch_ducts,
        "possible_noise": possible_noise
    }


def draw_debug(img, classified):

    debug = img.copy()

    # =========================
    # MAIN DUCTS = RED
    # =========================

    for duct in classified["main_ducts"]:

        path = duct["path"]

        for i in range(len(path) - 1):

            p1 = tuple(path[i])
            p2 = tuple(path[i + 1])

            cv2.line(
                debug,
                p1,
                p2,
                (0, 0, 255),
                8
            )

    # =========================
    # BRANCH DUCTS = GREEN
    # =========================

    for duct in classified["branch_ducts"]:

        path = duct["path"]

        for i in range(len(path) - 1):

            p1 = tuple(path[i])
            p2 = tuple(path[i + 1])

            cv2.line(
                debug,
                p1,
                p2,
                (0, 255, 0),
                4
            )

    cv2.imwrite(OUTPUT_DEBUG, debug)


def main():

    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    networks = data["duct_networks"]

    classified = classify_networks(networks)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(classified, f, indent=2)

    img = cv2.imread(INPUT_IMAGE)

    draw_debug(img, classified)

    print("DONE")
    print("Main ducts:", len(classified["main_ducts"]))
    print("Branch ducts:", len(classified["branch_ducts"]))
    print("Noise:", len(classified["possible_noise"]))


if __name__ == "__main__":
    main()
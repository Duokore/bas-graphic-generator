import json
import math
import os

INPUT_JSON = "../outputs/duct_connected.json"
OUTPUT_JSON = "../outputs/duct_graph_simplified.json"

MIN_LENGTH = 18
SNAP_GRID = 10
MERGE_DISTANCE = 12


def load_lines():
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    return data.get("lines", [])


def length(line):
    return math.hypot(
        line["x2"] - line["x1"],
        line["y2"] - line["y1"]
    )


def snap(value):
    return int(round(value / SNAP_GRID) * SNAP_GRID)


def snap_line(line):
    return {
        "x1": snap(line["x1"]),
        "y1": snap(line["y1"]),
        "x2": snap(line["x2"]),
        "y2": snap(line["y2"])
    }


def normalize_line(line):
    p1 = (line["x1"], line["y1"])
    p2 = (line["x2"], line["y2"])

    if p2 < p1:
        p1, p2 = p2, p1

    return {
        "x1": p1[0],
        "y1": p1[1],
        "x2": p2[0],
        "y2": p2[1]
    }


def is_horizontal(line):
    return abs(line["y2"] - line["y1"]) <= MERGE_DISTANCE


def is_vertical(line):
    return abs(line["x2"] - line["x1"]) <= MERGE_DISTANCE


def simplify_orientation(line):
    if is_horizontal(line):
        y = int(round((line["y1"] + line["y2"]) / 2))
        return {
            "x1": line["x1"],
            "y1": y,
            "x2": line["x2"],
            "y2": y,
            "type": "horizontal"
        }

    if is_vertical(line):
        x = int(round((line["x1"] + line["x2"]) / 2))
        return {
            "x1": x,
            "y1": line["y1"],
            "x2": x,
            "y2": line["y2"],
            "type": "vertical"
        }

    return None


def remove_duplicates(lines):
    seen = set()
    unique = []

    for line in lines:
        key = (
            line["x1"],
            line["y1"],
            line["x2"],
            line["y2"],
            line.get("type", "")
        )

        if key not in seen:
            seen.add(key)
            unique.append(line)

    return unique


def merge_collinear(lines):
    horizontal = {}
    vertical = {}

    for line in lines:
        if line["type"] == "horizontal":
            y = line["y1"]
            horizontal.setdefault(y, []).append(line)

        elif line["type"] == "vertical":
            x = line["x1"]
            vertical.setdefault(x, []).append(line)

    merged = []

    # Merge horizontal lines by Y
    for y, group in horizontal.items():
        group = sorted(group, key=lambda l: min(l["x1"], l["x2"]))

        current_start = None
        current_end = None

        for line in group:
            x1 = min(line["x1"], line["x2"])
            x2 = max(line["x1"], line["x2"])

            if current_start is None:
                current_start = x1
                current_end = x2
                continue

            if x1 <= current_end + MERGE_DISTANCE:
                current_end = max(current_end, x2)
            else:
                merged.append({
                    "x1": current_start,
                    "y1": y,
                    "x2": current_end,
                    "y2": y,
                    "type": "horizontal"
                })

                current_start = x1
                current_end = x2

        if current_start is not None:
            merged.append({
                "x1": current_start,
                "y1": y,
                "x2": current_end,
                "y2": y,
                "type": "horizontal"
            })

    # Merge vertical lines by X
    for x, group in vertical.items():
        group = sorted(group, key=lambda l: min(l["y1"], l["y2"]))

        current_start = None
        current_end = None

        for line in group:
            y1 = min(line["y1"], line["y2"])
            y2 = max(line["y1"], line["y2"])

            if current_start is None:
                current_start = y1
                current_end = y2
                continue

            if y1 <= current_end + MERGE_DISTANCE:
                current_end = max(current_end, y2)
            else:
                merged.append({
                    "x1": x,
                    "y1": current_start,
                    "x2": x,
                    "y2": current_end,
                    "type": "vertical"
                })

                current_start = y1
                current_end = y2

        if current_start is not None:
            merged.append({
                "x1": x,
                "y1": current_start,
                "x2": x,
                "y2": current_end,
                "type": "vertical"
            })

    return merged


def main():
    raw_lines = load_lines()

    print("Raw lines:", len(raw_lines))

    cleaned = []

    for line in raw_lines:
        if length(line) < MIN_LENGTH:
            continue

        snapped = snap_line(line)
        normalized = normalize_line(snapped)
        oriented = simplify_orientation(normalized)

        if oriented is None:
            continue

        if length(oriented) < MIN_LENGTH:
            continue

        cleaned.append(oriented)

    print("After cleanup:", len(cleaned))

    unique = remove_duplicates(cleaned)
    print("After duplicates removed:", len(unique))

    merged = merge_collinear(unique)
    print("After merge:", len(merged))

    final = [
        line for line in merged
        if length(line) >= MIN_LENGTH
    ]

    print("Final simplified:", len(final))

    os.makedirs("../outputs", exist_ok=True)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(
            {
                "raw_lines": len(raw_lines),
                "simplified_lines": len(final),
                "lines": final
            },
            f,
            indent=2
        )

    print("Saved:", OUTPUT_JSON)


if __name__ == "__main__":
    main()
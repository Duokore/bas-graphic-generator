import json
import cv2
import numpy as np
import math

INPUT_JSON = "outputs/duct_centerlines_healed.json"

OUTPUT_JSON = "outputs/duct_network.json"
OUTPUT_DEBUG = "outputs/duct_network_debug.png"

CANVAS_W = 2200
CANVAS_H = 1200

NODE_DISTANCE = 18


def load_lines():
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    return data["lines"]


def point_distance(a, b):
    return math.hypot(
        a[0] - b[0],
        a[1] - b[1]
    )


def normalize_point(pt):
    return (
        int(round(pt[0] / NODE_DISTANCE) * NODE_DISTANCE),
        int(round(pt[1] / NODE_DISTANCE) * NODE_DISTANCE)
    )


def build_nodes(lines):
    nodes = {}
    edges = []

    for line in lines:
        p1 = normalize_point((line["x1"], line["y1"]))
        p2 = normalize_point((line["x2"], line["y2"]))

        nodes[p1] = True
        nodes[p2] = True

        edges.append({
            "start": p1,
            "end": p2,
            "length": round(point_distance(p1, p2), 2)
        })

    return list(nodes.keys()), edges


def node_degree(nodes, edges):
    degree = {}

    for n in nodes:
        degree[n] = 0

    for e in edges:
        degree[e["start"]] += 1
        degree[e["end"]] += 1

    return degree


def classify_nodes(nodes, degree):
    trunks = []
    branches = []
    terminals = []

    for n in nodes:
        d = degree[n]

        if d >= 3:
            trunks.append(n)

        elif d == 2:
            branches.append(n)

        else:
            terminals.append(n)

    return trunks, branches, terminals


def draw_debug(nodes, edges, trunks, branches, terminals):
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    # edges
    for e in edges:
        cv2.line(
            canvas,
            e["start"],
            e["end"],
            (180, 180, 180),
            2
        )

    # trunk nodes
    for n in trunks:
        cv2.circle(canvas, n, 8, (0, 0, 255), -1)

    # branch nodes
    for n in branches:
        cv2.circle(canvas, n, 6, (0, 255, 255), -1)

    # terminal nodes
    for n in terminals:
        cv2.circle(canvas, n, 5, (255, 0, 0), -1)

    cv2.imwrite(OUTPUT_DEBUG, canvas)


def main():
    lines = load_lines()

    nodes, edges = build_nodes(lines)

    degree = node_degree(nodes, edges)

    trunks, branches, terminals = classify_nodes(
        nodes,
        degree
    )

    data = {
        "nodes": [
            {
                "x": n[0],
                "y": n[1],
                "degree": degree[n]
            }
            for n in nodes
        ],

        "edges": edges,

        "summary": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "trunks": len(trunks),
            "branches": len(branches),
            "terminals": len(terminals)
        }
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=2)

    draw_debug(
        nodes,
        edges,
        trunks,
        branches,
        terminals
    )

    print("DONE")
    print("Nodes:", len(nodes))
    print("Edges:", len(edges))
    print("Trunks:", len(trunks))
    print("Branches:", len(branches))
    print("Terminals:", len(terminals))
    print("Debug:", OUTPUT_DEBUG)


if __name__ == "__main__":
    main()
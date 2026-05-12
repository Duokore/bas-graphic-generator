import json
import math

INPUT = "../outputs/duct_connected.json"

ANGLE_TOLERANCE = 12
NODE_SNAP_DISTANCE = 18


class Node:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)
        self.edges = []

    def pos(self):
        return (self.x, self.y)


class Edge:
    def __init__(self, start, end):
        self.start = start
        self.end = end

        self.length = math.hypot(
            end.x - start.x,
            end.y - start.y
        )

        self.angle = math.degrees(
            math.atan2(
                end.y - start.y,
                end.x - start.x
            )
        )


class HVACGraph:

    def __init__(self):
        self.nodes = []
        self.edges = []

    def distance(self, a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def find_or_create_node(self, x, y):

        candidate = Node(x, y)

        for node in self.nodes:
            if self.distance(node, candidate) <= NODE_SNAP_DISTANCE:
                return node

        self.nodes.append(candidate)

        return candidate

    def add_line(self, line):

        start = self.find_or_create_node(
            line["x1"],
            line["y1"]
        )

        end = self.find_or_create_node(
            line["x2"],
            line["y2"]
        )

        edge = Edge(start, end)

        start.edges.append(edge)
        end.edges.append(edge)

        self.edges.append(edge)

    def load_json(self):

        with open(INPUT, "r") as f:
            data = json.load(f)

        for line in data["lines"]:
            self.add_line(line)

    def classify_nodes(self):

        trunks = []
        terminals = []
        junctions = []

        for node in self.nodes:

            count = len(node.edges)

            if count == 1:
                terminals.append(node)

            elif count == 2:
                trunks.append(node)

            elif count >= 3:
                junctions.append(node)

        return {
            "trunks": trunks,
            "terminals": terminals,
            "junctions": junctions
        }

    def print_summary(self):

        summary = self.classify_nodes()

        print("\n===== HVAC GRAPH =====")

        print("Nodes:", len(self.nodes))
        print("Edges:", len(self.edges))

        print("Trunks:", len(summary["trunks"]))
        print("Terminals:", len(summary["terminals"]))
        print("Junctions:", len(summary["junctions"]))


if __name__ == "__main__":

    graph = HVACGraph()

    graph.load_json()

    graph.print_summary()
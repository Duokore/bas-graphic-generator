from flask import Flask, request, render_template, jsonify, send_file, session, redirect, url_for
import os
import base64
import json
import cv2
import numpy as np
import fitz
from geometry_engine import *

app = Flask(__name__)

app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-this")
BAS_PASSWORD = os.environ.get("BAS_PASSWORD", "admin123")

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

UPLOAD_IMAGE = os.path.join(UPLOAD_FOLDER, "mechanical_upload.png")
FINAL_RENDER = os.path.join(OUTPUT_FOLDER, "synchrony_static_render.png")
PROJECT_FILE = os.path.join(OUTPUT_FOLDER, "project_data.json")


LOGIN_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>BAS Login</title>
<style>
body{
    background:#0d1117;
    color:white;
    font-family:Arial;
    display:flex;
    justify-content:center;
    align-items:center;
    height:100vh;
}
.card{
    background:#181b24;
    padding:40px;
    border-radius:18px;
    width:350px;
    text-align:center;
}
input{
    width:90%;
    padding:14px;
    border-radius:10px;
    border:none;
    margin:15px 0;
}
button{
    background:#16a34a;
    color:white;
    border:none;
    padding:14px 28px;
    border-radius:10px;
    font-weight:bold;
    cursor:pointer;
}
.error{
    color:#ff6b6b;
}
</style>
</head>
<body>
<div class="card">
<h1>BAS Private Access</h1>
<p>Enter password to continue</p>

{% if error %}
<p class="error">{{ error }}</p>
{% endif %}

<form method="POST" action="/login">
<input type="password" name="password" placeholder="Password" required>
<br>
<button type="submit">Login</button>
</form>
</div>
</body>
</html>
"""


def login_required():
    return session.get("logged_in") is True


@app.before_request
def protect_routes():
    allowed_routes = ["login", "static"]

    if request.endpoint in allowed_routes:
        return

    if not login_required():
        return redirect(url_for("login"))


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def pdf_to_png(pdf_path, out_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
    pix.save(out_path)
    doc.close()


def draw_shadow_line(canvas, p1, p2, color, thickness):
    offset = 5

    cv2.line(
        canvas,
        (p1[0] + offset, p1[1] + offset),
        (p2[0] + offset, p2[1] + offset),
        (80, 80, 80),
        thickness + 4,
        cv2.LINE_AA
    )

    cv2.line(
        canvas,
        p1,
        p2,
        color,
        thickness,
        cv2.LINE_AA
    )


def render_static_graphic():
    img = cv2.imread(UPLOAD_IMAGE)

    if img is None:
        raise ValueError("Could not read upload image")

    with open(PROJECT_FILE, "r") as f:
        project = json.load(f)

    points = project["points"]
    ducts = project["ducts"]

    h, w = img.shape[:2]

    target_w = 1800
    target_h = int(h * (target_w / w))

    if target_h > 950:
        target_h = 950
        target_w = int(w * (target_h / h))

    scale_x = target_w / w
    scale_y = target_h / h

    canvas_w = target_w + 140
    canvas_h = target_h + 140

    canvas = np.full(
        (canvas_h, canvas_w, 3),
        (235, 235, 235),
        dtype=np.uint8
    )

    ox = 70
    oy = 70

    # GRID
    for x in range(ox, ox + target_w, 24):
        cv2.line(canvas, (x, oy), (x, oy + target_h), (215, 215, 215), 1)

    for y in range(oy, oy + target_h, 24):
        cv2.line(canvas, (ox, y), (ox + target_w, y), (215, 215, 215), 1)

    # FLOORPLAN OVERLAY
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    wall_mask = build_wall_mask(gray)

    wall_mask = cv2.resize(
        wall_mask,
        (target_w, target_h)
    )

    gray_resized = cv2.resize(
        gray,
        (target_w, target_h)
    )

    plan = cv2.cvtColor(
        gray_resized,
        cv2.COLOR_GRAY2BGR
    )

    plan = cv2.threshold(
        plan,
        215,
        255,
        cv2.THRESH_BINARY
    )[1]

    plan = 255 - plan

    plan = cv2.GaussianBlur(
        plan,
        (3, 3),
        0
    )

    roi = canvas[
        oy:oy + target_h,
        ox:ox + target_w
    ]

    roi = cv2.addWeighted(
        roi,
        1.0,
        plan,
        0.05,
        0
    )

    canvas[
        oy:oy + target_h,
        ox:ox + target_w
    ] = roi

    # WALLS
    draw_extruded_walls(
        canvas,
        wall_mask,
        ox,
        oy
    )

    # DUCTS
    for d in ducts:
        if d.get("type") == "curve":
            p1 = (
                int(d["x1"] * scale_x) + ox,
                int(d["y1"] * scale_y) + oy
            )

            pc = (
                int(d["cx"] * scale_x) + ox,
                int(d["cy"] * scale_y) + oy
            )

            p2 = (
                int(d["x2"] * scale_x) + ox,
                int(d["y2"] * scale_y) + oy
            )

            curve_points = []

            for t in np.linspace(0, 1, 40):
                x = int(
                    (1 - t) ** 2 * p1[0] +
                    2 * (1 - t) * t * pc[0] +
                    t ** 2 * p2[0]
                )

                y = int(
                    (1 - t) ** 2 * p1[1] +
                    2 * (1 - t) * t * pc[1] +
                    t ** 2 * p2[1]
                )

                curve_points.append([x, y])

            curve_points = np.array(
                curve_points,
                dtype=np.int32
            )

            shadow = curve_points + np.array([5, 5])

            cv2.polylines(
                canvas,
                [shadow],
                False,
                (80, 80, 80),
                14,
                cv2.LINE_AA
            )

            cv2.polylines(
                canvas,
                [curve_points],
                False,
                (250, 250, 250),
                10,
                cv2.LINE_AA
            )

        else:
            p1 = (
                int(d["x1"] * scale_x) + ox,
                int(d["y1"] * scale_y) + oy
            )

            p2 = (
                int(d["x2"] * scale_x) + ox,
                int(d["y2"] * scale_y) + oy
            )

            draw_shadow_line(
                canvas,
                p1,
                p2,
                (250, 250, 250),
                10
            )

    # EQUIPMENT
    for p in points:
        px = int(p["x"] * scale_x) + ox
        py = int(p["y"] * scale_y) + oy

        if p["type"] == "VAV":
            size = 34

            x1 = px - size // 2
            y1 = py - size // 2
            x2 = px + size // 2
            y2 = py + size // 2

            cv2.rectangle(canvas, (x1 + 5, y1 + 5), (x2 + 5, y2 + 5), (55, 55, 55), -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (95, 125, 190), -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (20, 55, 120), 3)

        elif p["type"] == "AHU":
            bw = 70
            bh = 55

            x1 = px - bw // 2
            y1 = py - bh // 2
            x2 = px + bw // 2
            y2 = py + bh // 2

            cv2.rectangle(canvas, (x1 + 6, y1 + 6), (x2 + 6, y2 + 6), (55, 55, 55), -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (35, 145, 55), -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (15, 90, 35), 4)

    cv2.imwrite(
        FINAL_RENDER,
        canvas
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        password = request.form.get("password")

        if password == BAS_PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("home"))

        return render_template_string(LOGIN_PAGE, error="Incorrect password")

    return render_template_string(LOGIN_PAGE, error=None)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
def home():
    image = None

    if os.path.exists(UPLOAD_IMAGE):
        image = image_to_base64(UPLOAD_IMAGE)

    return render_template(
        "index.html",
        image=image
    )


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]

    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        pdf_path = os.path.join(
            UPLOAD_FOLDER,
            "upload.pdf"
        )

        file.save(pdf_path)

        pdf_to_png(
            pdf_path,
            UPLOAD_IMAGE
        )

    else:
        file.save(UPLOAD_IMAGE)

    img = cv2.imread(UPLOAD_IMAGE)

    if img is None:
        return "Error loading image", 400

    h, w = img.shape[:2]

    if h > w:
        img = cv2.rotate(
            img,
            cv2.ROTATE_90_CLOCKWISE
        )

    cv2.imwrite(
        UPLOAD_IMAGE,
        img
    )

    return render_template(
        "index.html",
        image=image_to_base64(UPLOAD_IMAGE)
    )


@app.route("/generate_manual", methods=["POST"])
def generate_manual():
    data = request.get_json()

    points = data.get("points", [])
    ducts = data.get("ducts", [])

    vavs = [
        p for p in points
        if p["type"] == "VAV"
    ]

    ahus = [
        p for p in points
        if p["type"] == "AHU"
    ]

    if len(vavs) == 0:
        return jsonify({
            "success": False,
            "error": "Need at least one VAV."
        })

    if len(ahus) == 0:
        return jsonify({
            "success": False,
            "error": "Need at least one AHU."
        })

    with open(PROJECT_FILE, "w") as f:
        json.dump({
            "points": points,
            "ducts": ducts
        }, f, indent=2)

    try:
        render_static_graphic()

        return jsonify({
            "success": True
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })


@app.route("/result")
def result():
    if not os.path.exists(FINAL_RENDER):
        return "No render generated yet."

    image = image_to_base64(FINAL_RENDER)

    return f"""
    <html>
    <head>
        <title>BAS Render Result</title>

        <style>
            body {{
                background:#0d1117;
                color:white;
                font-family:Arial;
                text-align:center;
                padding:20px;
            }}

            img {{
                max-width:95%;
                border-radius:12px;
                border:2px solid #333;
                background:white;
            }}

            a {{
                display:inline-block;
                margin:15px;
                padding:14px 24px;
                background:#16a34a;
                color:white;
                border-radius:10px;
                text-decoration:none;
                font-weight:bold;
            }}
        </style>
    </head>

    <body>
        <h1>
            Generated BAS Graphic
        </h1>

        <img src="data:image/png;base64,{image}">

        <br>

        <a href="/download">
            Download PNG
        </a>

        <a href="/">
            Back
        </a>

        <a href="/logout">
            Logout
        </a>
    </body>
    </html>
    """


@app.route("/download")
def download():
    return send_file(
        FINAL_RENDER,
        as_attachment=True
    )


if __name__ == "__main__":
    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )

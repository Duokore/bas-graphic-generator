from flask import Flask, request, send_file
import cv2
import numpy as np
import os
import base64
import fitz
import math

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

UPLOAD_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, "mechanical_upload.png")
UPLOAD_PDF_PATH = os.path.join(UPLOAD_FOLDER, "mechanical_upload.pdf")
PNG_OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, "bas_graphic_transparent.png")
ISOMETRIC_OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, "bas_graphic_isometric.png")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def pdf_to_png(pdf_path, output_image_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    pix.save(output_image_path)
    doc.close()


def create_isometric_view(img):
    h, w = img.shape[:2]

    src = np.float32([
        [0, 0],
        [w, 0],
        [0, h],
        [w, h]
    ])

    dst = np.float32([
        [w * 0.12, h * 0.05],
        [w * 0.88, h * 0.12],
        [0, h * 0.92],
        [w, h * 0.84]
    ])

    matrix = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(
        img,
        matrix,
        (w, h),
        borderMode=cv2.BORDER_TRANSPARENT
    )

    shadow = cv2.GaussianBlur(warped, (31, 31), 0)
    iso = cv2.addWeighted(warped, 1, shadow, 0.14, 0)

    return iso


def filter_architectural_lines(lines):
    filtered = []

    if lines is None:
        return filtered

    for line in lines:
        x1, y1, x2, y2 = line[0]

        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)

        if length < 45:
            continue

        angle = abs(math.degrees(math.atan2(dy, dx)))

        is_horizontal = angle < 10 or angle > 170
        is_vertical = 80 < angle < 100

        if not (is_horizontal or is_vertical):
            continue

        filtered.append((x1, y1, x2, y2, length, angle))

    return filtered


def build_wall_mask(lines, height, width):
    wall_mask = np.zeros((height, width), dtype=np.uint8)

    for x1, y1, x2, y2, length, angle in lines:
        thickness = 7 if length > 120 else 5

        cv2.line(
            wall_mask,
            (x1, y1),
            (x2, y2),
            255,
            thickness
        )

    wall_mask = cv2.morphologyEx(
        wall_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    )

    wall_mask = cv2.dilate(
        wall_mask,
        np.ones((2, 2), np.uint8),
        iterations=1
    )

    return wall_mask


def remove_small_components(mask, min_area=180):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        if area >= min_area and (width > 30 or height > 30):
            cleaned[labels == i] = 255

    return cleaned


@app.route("/")
def home():
    return """
    <html>
    <head>
        <title>BAS Graphic Generator v6</title>
    </head>
    <body style="margin:0;background:#101218;color:white;font-family:Arial, Helvetica, sans-serif;">
        <div style="max-width:950px;margin:55px auto;background:#1b1d25;padding:42px;border-radius:24px;text-align:center;border:1px solid #303542;">
            <div style="font-size:42px;">⚙️ 🏢 📐</div>
            <h1>BAS Graphic Generator v6</h1>
            <p style="color:#b8bcc8;">Architectural Filter · Vector Wall Engine · BAS Detection</p>

            <form action="/generate" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/png,image/jpeg,application/pdf" required style="background:#222631;color:white;padding:14px;border-radius:10px;width:80%;border:1px solid #3b4050;">
                <br><br>
                <button style="font-size:20px;padding:15px 34px;background:#2d89ef;color:white;border:none;border-radius:12px;cursor:pointer;font-weight:bold;">
                    Generate BAS Graphic
                </button>
            </form>

            <p style="font-size:14px;color:#8d93a3;">Supports PNG, JPG, and PDF marked plans.</p>
            <p style="font-size:13px;color:#666;">Made by Paolo V. & Emmanuel R.</p>
        </div>
    </body>
    </html>
    """


@app.route("/generate", methods=["POST"])
def generate():
    file = request.files["file"]
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        file.save(UPLOAD_PDF_PATH)
        pdf_to_png(UPLOAD_PDF_PATH, UPLOAD_IMAGE_PATH)
    else:
        file.save(UPLOAD_IMAGE_PATH)

    img = cv2.imread(UPLOAD_IMAGE_PATH)

    if img is None:
        return "Error loading image"

    h, w = img.shape[:2]

    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h, w = img.shape[:2]

    target_w = 1600
    scale = target_w / w
    target_h = int(h * scale)

    img = cv2.resize(img, (target_w, target_h))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    canvas = np.zeros((target_h, target_w, 4), dtype=np.uint8)

    # =========================
    # ARCHITECTURAL FILTER ENGINE v1
    # =========================

    enhanced = cv2.equalizeHist(gray)

    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    edges = cv2.Canny(
        blur,
        70,
        190
    )

    raw_lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=75,
        minLineLength=45,
        maxLineGap=22
    )

    vector_lines = filter_architectural_lines(raw_lines)

    wall_mask = build_wall_mask(
        vector_lines,
        target_h,
        target_w
    )

    wall_mask = remove_small_components(
        wall_mask,
        min_area=250
    )

    # =========================
    # FLOOR AREA FROM WALL MASS
    # =========================

    floor_seed = cv2.GaussianBlur(wall_mask, (51, 51), 0)

    floor_seed = cv2.threshold(
        floor_seed,
        8,
        255,
        cv2.THRESH_BINARY
    )[1]

    floor_seed = cv2.morphologyEx(
        floor_seed,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (55, 55))
    )

    contours, _ = cv2.findContours(
        floor_seed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    floor_area = np.zeros_like(floor_seed)

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 25000:
            hull = cv2.convexHull(contour)
            cv2.drawContours(
                floor_area,
                [hull],
                -1,
                255,
                thickness=cv2.FILLED
            )

    floor_area = cv2.GaussianBlur(floor_area, (17, 17), 0)
    floor_area = cv2.threshold(floor_area, 10, 255, cv2.THRESH_BINARY)[1]

    # =========================
    # COLOR BAS DETECTION
    # =========================

    red1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([12, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([165, 70, 50]), np.array([180, 255, 255]))
    ducts = red1 + red2

    blue = cv2.inRange(hsv, np.array([90, 50, 40]), np.array([140, 255, 255]))
    green = cv2.inRange(hsv, np.array([40, 40, 40]), np.array([90, 255, 255]))
    purple = cv2.inRange(hsv, np.array([120, 30, 40]), np.array([170, 255, 255]))

    ducts = cv2.dilate(ducts, np.ones((5, 5), np.uint8), iterations=1)
    blue = cv2.dilate(blue, np.ones((4, 4), np.uint8), iterations=1)
    green = cv2.dilate(green, np.ones((4, 4), np.uint8), iterations=1)
    purple = cv2.dilate(purple, np.ones((3, 3), np.uint8), iterations=1)

    # =========================
    # RENDER ENGINE
    # =========================

    canvas[floor_area > 0] = (38, 42, 52, 105)
    canvas[wall_mask > 0] = (155, 155, 165, 255)

    canvas[ducts > 0] = (225, 225, 225, 255)
    canvas[blue > 0] = (255, 120, 0, 255)
    canvas[green > 0] = (80, 220, 80, 255)
    canvas[purple > 0] = (190, 110, 220, 255)

    cv2.imwrite(PNG_OUTPUT_PATH, canvas)
    cv2.imwrite(UPLOAD_IMAGE_PATH, img)

    iso = create_isometric_view(canvas)
    cv2.imwrite(ISOMETRIC_OUTPUT_PATH, iso)

    original_base64 = image_to_base64(UPLOAD_IMAGE_PATH)
    generated_base64 = image_to_base64(PNG_OUTPUT_PATH)
    iso_base64 = image_to_base64(ISOMETRIC_OUTPUT_PATH)

    return f"""
    <html>
    <head>
        <title>BAS Graphic Comparison</title>
        <style>
            body {{
                margin:0;
                background:#101218;
                color:white;
                font-family:Arial, Helvetica, sans-serif;
                padding:30px;
            }}

            .grid {{
                display:grid;
                grid-template-columns:1fr 1fr 1fr;
                gap:24px;
                margin-top:30px;
            }}

            .card {{
                background:#1b1d25;
                padding:18px;
                border-radius:18px;
                border:1px solid #333;
                box-shadow:0 0 30px rgba(0,0,0,0.6);
            }}

            .viewer {{
                width:100%;
                height:650px;
                overflow:hidden;
                background:#111;
                border:1px solid #333;
                border-radius:12px;
            }}

            .viewer img {{
                width:100%;
            }}

            button {{
                font-size:18px;
                padding:14px 30px;
                color:white;
                border:none;
                border-radius:10px;
                cursor:pointer;
                font-weight:bold;
            }}
        </style>
    </head>

    <body>
        <h1 style="text-align:center;">BAS Graphic Comparison</h1>
        <p style="text-align:center;color:#b8bcc8;">
            Architectural Filter Engine v1 · Text Reduction · Vector Walls
        </p>

        <div class="grid">
            <div class="card">
                <h2 style="text-align:center;">Original</h2>
                <div class="viewer">
                    <img src="data:image/png;base64,{original_base64}">
                </div>
            </div>

            <div class="card">
                <h2 style="text-align:center;">BAS Graphic</h2>
                <div class="viewer">
                    <img src="data:image/png;base64,{generated_base64}">
                </div>
            </div>

            <div class="card">
                <h2 style="text-align:center;">Isometric View</h2>
                <div class="viewer">
                    <img src="data:image/png;base64,{iso_base64}">
                </div>
            </div>
        </div>

        <div style="text-align:center;margin-top:35px;">
            <a href="/download_png">
                <button style="background:#28a745;">Download PNG</button>
            </a>

            <a href="/">
                <button style="background:#444;margin-left:10px;">Generate Another</button>
            </a>
        </div>

        <p style="font-size:13px;color:#666;text-align:center;margin-top:25px;">
            Made by Paolo V. & Emmanuel R.
        </p>
    </body>
    </html>
    """


@app.route("/download_png")
def download_png():
    return send_file(
        PNG_OUTPUT_PATH,
        mimetype="image/png",
        as_attachment=True,
        download_name="bas_graphic.png"
    )


if __name__ == "__main__":
    app.run(debug=True)
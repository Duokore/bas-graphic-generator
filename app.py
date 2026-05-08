from flask import Flask, request, send_file
import cv2
import numpy as np
import os
import base64
import fitz
import svgwrite

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

UPLOAD_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, "mechanical_upload.png")
UPLOAD_PDF_PATH = os.path.join(UPLOAD_FOLDER, "mechanical_upload.pdf")
PNG_OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, "bas_graphic_transparent.png")
SVG_OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, "bas_graphic.svg")

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


def mask_to_svg(dwg, mask, color):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if len(contour) < 3:
            continue

        points = [(int(point[0][0]), int(point[0][1])) for point in contour]

        dwg.add(
            dwg.polygon(
                points=points,
                fill=color,
                stroke=color,
                stroke_width=1
            )
        )


def create_svg(width, height, edges, ducts, blue, green, purple):
    dwg = svgwrite.Drawing(SVG_OUTPUT_PATH, size=(width, height))

    mask_to_svg(dwg, edges, "rgb(120,120,128)")
    mask_to_svg(dwg, ducts, "rgb(220,220,220)")
    mask_to_svg(dwg, blue, "rgb(255,120,0)")
    mask_to_svg(dwg, green, "rgb(80,220,80)")
    mask_to_svg(dwg, purple, "rgb(190,110,220)")

    dwg.save()


@app.route("/")
def home():
    return """
    <html>
    <head>
        <title>BAS Graphic Generator v1</title>
    </head>
    <body style="margin:0;background:#101218;color:white;font-family:Arial, Helvetica, sans-serif;">
        <div style="max-width:950px;margin:55px auto;background:#1b1d25;padding:42px;border-radius:24px;text-align:center;border:1px solid #303542;">
            <div style="font-size:42px;">⚙️ 🏢 📐</div>
            <h1>BAS Graphic Generator v1</h1>
            <p style="color:#b8bcc8;">HVAC Controls · Mechanical Plans · Synchrony Style Graphics</p>

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

    target_w = 1400
    scale = target_w / w
    target_h = int(h * scale)

    img = cv2.resize(img, (target_w, target_h))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    canvas = np.zeros((target_h, target_w, 4), dtype=np.uint8)

    # SMART CLEANUP MODE
    blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur_gray, 130, 230)

    cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, cleanup_kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges)

    clean_edges = np.zeros_like(edges)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area > 50:
            clean_edges[labels == i] = 255

    edges = cv2.dilate(clean_edges, np.ones((2, 2), np.uint8), iterations=1)

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

    canvas[edges > 0] = (120, 120, 128, 255)
    canvas[ducts > 0] = (220, 220, 220, 255)
    canvas[blue > 0] = (255, 120, 0, 255)
    canvas[green > 0] = (80, 220, 80, 255)
    canvas[purple > 0] = (190, 110, 220, 255)

    cv2.imwrite(PNG_OUTPUT_PATH, canvas)
    cv2.imwrite(UPLOAD_IMAGE_PATH, img)

    create_svg(target_w, target_h, edges, ducts, blue, green, purple)

    original_base64 = image_to_base64(UPLOAD_IMAGE_PATH)
    generated_base64 = image_to_base64(PNG_OUTPUT_PATH)

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
                grid-template-columns:1fr 1fr;
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
                height:620px;
                overflow:hidden;
                background:#111;
                border:1px solid #333;
                border-radius:12px;
                position:relative;
                cursor:grab;
            }}

            .viewer img {{
                position:absolute;
                top:0;
                left:0;
                width:100%;
                transform-origin:0 0;
                user-select:none;
                -webkit-user-drag:none;
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
            Smart Cleanup Mode · Scroll to zoom · Drag to pan
        </p>

        <div class="grid">
            <div class="card">
                <h2 style="text-align:center;">Original Marked Plan</h2>
                <div class="viewer">
                    <img class="zoomable" src="data:image/png;base64,{original_base64}">
                </div>
            </div>

            <div class="card">
                <h2 style="text-align:center;">Generated BAS Graphic</h2>
                <div class="viewer">
                    <img class="zoomable" src="data:image/png;base64,{generated_base64}">
                </div>
            </div>
        </div>

        <div style="text-align:center;margin-top:35px;">
            <a href="/download_png">
                <button style="background:#28a745;">
                    Download Transparent PNG
                </button>
            </a>

            <a href="/download_svg">
                <button style="background:#8e44ad;margin-left:10px;">
                    Download SVG
                </button>
            </a>

            <a href="/">
                <button style="background:#444;margin-left:10px;">
                    Generate Another
                </button>
            </a>
        </div>

        <p style="font-size:13px;color:#666;text-align:center;margin-top:25px;">
            Made by Paolo V. & Emmanuel R.
        </p>

        <script>
            document.querySelectorAll('.viewer').forEach(viewer => {{
                const img = viewer.querySelector('img');

                let scale = 1;
                let posX = 0;
                let posY = 0;
                let isDragging = false;
                let startX = 0;
                let startY = 0;

                function updateTransform() {{
                    img.style.transform = `translate(${{posX}}px, ${{posY}}px) scale(${{scale}})`;
                }}

                viewer.addEventListener('wheel', function(e) {{
                    e.preventDefault();

                    const zoomSpeed = 0.1;
                    const oldScale = scale;

                    if (e.deltaY < 0) {{
                        scale += zoomSpeed;
                    }} else {{
                        scale -= zoomSpeed;
                    }}

                    scale = Math.max(0.3, Math.min(scale, 6));

                    const rect = viewer.getBoundingClientRect();
                    const mouseX = e.clientX - rect.left;
                    const mouseY = e.clientY - rect.top;

                    posX = mouseX - (mouseX - posX) * (scale / oldScale);
                    posY = mouseY - (mouseY - posY) * (scale / oldScale);

                    updateTransform();
                }});

                viewer.addEventListener('mousedown', function(e) {{
                    isDragging = true;
                    viewer.style.cursor = 'grabbing';
                    startX = e.clientX - posX;
                    startY = e.clientY - posY;
                }});

                window.addEventListener('mouseup', function() {{
                    isDragging = false;
                    viewer.style.cursor = 'grab';
                }});

                window.addEventListener('mousemove', function(e) {{
                    if (!isDragging) return;

                    posX = e.clientX - startX;
                    posY = e.clientY - startY;

                    updateTransform();
                }});
            }});
        </script>
    </body>
    </html>
    """


@app.route("/download_png")
def download_png():
    return send_file(
        PNG_OUTPUT_PATH,
        mimetype="image/png",
        as_attachment=True,
        download_name="bas_graphic_transparent.png"
    )


@app.route("/download_svg")
def download_svg():
    return send_file(
        SVG_OUTPUT_PATH,
        mimetype="image/svg+xml",
        as_attachment=True,
        download_name="bas_graphic.svg"
    )


if __name__ == "__main__":
    app.run(debug=True)
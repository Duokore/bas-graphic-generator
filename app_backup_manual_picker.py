from flask import Flask, request, render_template_string, send_file
import os
import base64
import cv2
import numpy as np
import fitz

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

MARKED_IMAGE = os.path.join(
    UPLOAD_FOLDER,
    "mechanical_marked.png"
)

FINAL_RENDER = os.path.join(
    OUTPUT_FOLDER,
    "synchrony_static_render.png"
)

# =========================================================
# HTML
# =========================================================

HOME_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>BAS Static Graphic Generator</title>

<style>

body{
    background:#0d0f14;
    color:white;
    font-family:Arial;
    text-align:center;
    padding:50px;
}

.card{
    background:#181b24;
    border-radius:20px;
    padding:40px;
    max-width:700px;
    margin:auto;
}

button{
    background:#16a34a;
    color:white;
    border:0;
    padding:15px 30px;
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
    cursor:pointer;
}

input{
    margin:25px;
}

</style>

</head>

<body>

<div class="card">

<h1>BAS Static Graphic Generator</h1>

<p>
Upload your marked mechanical plan.
</p>

<p>
Red = Ducts | Blue = VAVs | Green = AHU
</p>

<form action="/upload" method="post" enctype="multipart/form-data">

<input
    type="file"
    name="file"
    accept="image/png,image/jpeg,application/pdf"
    required
>

<br>

<button type="submit">
Generate Graphic
</button>

</form>

</div>

</body>
</html>
"""

RESULT_PAGE = """
<!DOCTYPE html>
<html>

<head>

<title>Generated BAS Graphic</title>

<style>

body{
    background:#0d0f14;
    color:white;
    font-family:Arial;
    text-align:center;
    padding:20px;
}

img{
    max-width:95%;
    border-radius:12px;
    border:1px solid #333;
    background:white;
}

.btn{
    display:inline-block;
    margin:15px;
    padding:14px 25px;
    border-radius:10px;
    background:#16a34a;
    color:white;
    text-decoration:none;
    font-weight:bold;
}

.btn2{
    background:#333;
}

</style>

</head>

<body>

<h1>Generated BAS Static Graphic</h1>

<img src="data:image/png;base64,{{ image }}">

<br>

<a class="btn" href="/download">
Download PNG
</a>

<a class="btn btn2" href="/">
New Plan
</a>

</body>
</html>
"""

# =========================================================
# HELPERS
# =========================================================

def image_to_base64(path):

    with open(path, "rb") as f:
        return base64.b64encode(
            f.read()
        ).decode("utf-8")


def pdf_to_png(pdf_path, out_path):

    doc = fitz.open(pdf_path)

    page = doc[0]

    pix = page.get_pixmap(
        matrix=fitz.Matrix(2.5, 2.5)
    )

    pix.save(out_path)

    doc.close()


def clean_mask(mask, close_size=5, iterations=2):

    kernel = np.ones(
        (close_size, close_size),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=iterations
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    return mask


def get_color_masks(img):

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    # RED DUCTS

    red1 = cv2.inRange(
        hsv,
        np.array([0, 70, 60]),
        np.array([12, 255, 255])
    )

    red2 = cv2.inRange(
        hsv,
        np.array([168, 70, 60]),
        np.array([180, 255, 255])
    )

    duct_mask = cv2.bitwise_or(
        red1,
        red2
    )

    # BLUE VAVS

    blue_mask = cv2.inRange(
        hsv,
        np.array([88, 50, 40]),
        np.array([145, 255, 255])
    )

    # GREEN AHU

    green_mask = cv2.inRange(
        hsv,
        np.array([35, 45, 45]),
        np.array([90, 255, 255])
    )

    duct_mask = clean_mask(
        duct_mask,
        7,
        2
    )

    blue_mask = clean_mask(
        blue_mask,
        5,
        1
    )

    green_mask = clean_mask(
        green_mask,
        5,
        1
    )

    return duct_mask, blue_mask, green_mask


def extract_boxes(mask, min_area=80, max_area=100000):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for c in contours:

        area = cv2.contourArea(c)

        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(c)

        boxes.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "area": float(area),
            "cx": int(x + w / 2),
            "cy": int(y + h / 2)
        })

    return boxes


def draw_shadow_line(
    canvas,
    p1,
    p2,
    color,
    thickness
):

    shadow_offset = 5

    cv2.line(
        canvas,
        (p1[0] + shadow_offset, p1[1] + shadow_offset),
        (p2[0] + shadow_offset, p2[1] + shadow_offset),
        (90, 90, 90),
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

# =========================================================
# RENDER ENGINE
# =========================================================

def render_static_graphic(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Could not read image")

    h, w = img.shape[:2]

    target_w = 1800
    target_h = int(h * (target_w / w))

    if target_h > 950:

        target_h = 950

        target_w = int(
            w * (target_h / h)
        )

    scale_x = target_w / w
    scale_y = target_h / h

    duct_mask, blue_mask, green_mask = get_color_masks(img)

    # =====================================================
    # DUCTS
    # =====================================================

    ducts = extract_boxes(
        duct_mask,
        min_area=100
    )

    # =====================================================
    # VAVS
    # =====================================================

    vavs = extract_boxes(
        blue_mask,
        min_area=80,
        max_area=6000
    )

    vavs = sorted(
        vavs,
        key=lambda v: (
            -v["area"],
            v["y"]
        )
    )

    filtered_vavs = []

    for v in vavs:

        keep = True

        for existing in filtered_vavs:

            dx = abs(
                v["cx"] - existing["cx"]
            )

            dy = abs(
                v["cy"] - existing["cy"]
            )

            if dx < 45 and dy < 45:
                keep = False
                break

        if keep:
            filtered_vavs.append(v)

    vavs = filtered_vavs[:9]

    # =====================================================
    # AHU
    # =====================================================

    ahus = extract_boxes(
        green_mask,
        min_area=150
    )

    ahus = sorted(
        ahus,
        key=lambda a: a["area"],
        reverse=True
    )[:1]

    # =====================================================
    # CANVAS
    # =====================================================

    canvas_w = target_w + 140
    canvas_h = target_h + 140

    canvas = np.full(
        (canvas_h, canvas_w, 3),
        (235, 235, 235),
        dtype=np.uint8
    )

    ox = 70
    oy = 70

    # =====================================================
    # FLOOR GRID
    # =====================================================

    floor = canvas.copy()

    for x in range(ox, ox + target_w, 24):

        cv2.line(
            floor,
            (x, oy),
            (x, oy + target_h),
            (210, 210, 210),
            1
        )

    for y in range(oy, oy + target_h, 24):

        cv2.line(
            floor,
            (ox, y),
            (ox + target_w, y),
            (210, 210, 210),
            1
        )

    canvas = cv2.addWeighted(
        canvas,
        0.82,
        floor,
        0.18,
        0
    )

    # =====================================================
    # PLAN OVERLAY
    # =====================================================

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        (target_w, target_h)
    )

    plan = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR
    )

    plan = cv2.threshold(
        plan,
        210,
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
        0.10,
        0
    )

    canvas[
        oy:oy + target_h,
        ox:ox + target_w
    ] = roi

    # =====================================================
    # DUCTS
    # =====================================================

    duct_resized = cv2.resize(
        duct_mask,
        (target_w, target_h)
    )

    contours, _ = cv2.findContours(
        duct_resized,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for c in contours:

        area = cv2.contourArea(c)

        if area < 120:
            continue

        x, y, bw, bh = cv2.boundingRect(c)

        if bw >= bh:

            p1 = (
                ox + x,
                oy + y + bh // 2
            )

            p2 = (
                ox + x + bw,
                oy + y + bh // 2
            )

        else:

            p1 = (
                ox + x + bw // 2,
                oy + y
            )

            p2 = (
                ox + x + bw // 2,
                oy + y + bh
            )

        draw_shadow_line(
            canvas,
            p1,
            p2,
            (250,250,250),
            8
        )

    # =====================================================
    # VAVS
    # =====================================================

    for i, v in enumerate(vavs, start=1):

        x = int(v["x"] * scale_x) + ox
        y = int(v["y"] * scale_y) + oy

        bw = max(
            24,
            int(v["w"] * scale_x)
        )

        bh = max(
            24,
            int(v["h"] * scale_y)
        )

        cv2.rectangle(
            canvas,
            (x + 5, y + 5),
            (x + bw + 5, y + bh + 5),
            (55,55,55),
            -1
        )

        cv2.rectangle(
            canvas,
            (x, y),
            (x + bw, y + bh),
            (95,125,190),
            -1
        )

        cv2.rectangle(
            canvas,
            (x, y),
            (x + bw, y + bh),
            (20,55,120),
            3
        )

    # =====================================================
    # AHU
    # =====================================================

    for a in ahus:

        x = int(a["x"] * scale_x) + ox
        y = int(a["y"] * scale_y) + oy

        bw = max(
            55,
            int(a["w"] * scale_x)
        )

        bh = max(
            45,
            int(a["h"] * scale_y)
        )

        cv2.rectangle(
            canvas,
            (x + 6, y + 6),
            (x + bw + 6, y + bh + 6),
            (55,55,55),
            -1
        )

        cv2.rectangle(
            canvas,
            (x, y),
            (x + bw, y + bh),
            (35,145,55),
            -1
        )

        cv2.rectangle(
            canvas,
            (x, y),
            (x + bw, y + bh),
            (15,90,35),
            4
        )

    cv2.imwrite(
        FINAL_RENDER,
        canvas
    )

    print("Generated:", FINAL_RENDER)
    print("Duct pieces:", len(ducts))
    print("VAVs:", len(vavs))
    print("AHUs:", len(ahus))

# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def home():
    return HOME_PAGE


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
            MARKED_IMAGE
        )

    else:

        file.save(MARKED_IMAGE)

    render_static_graphic(
        MARKED_IMAGE
    )

    return render_template_string(
        RESULT_PAGE,
        image=image_to_base64(
            FINAL_RENDER
        )
    )


@app.route("/download")
def download():

    return send_file(
        FINAL_RENDER,
        as_attachment=True
    )

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )
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
UPLOAD_PDF_PATH   = os.path.join(UPLOAD_FOLDER, "mechanical_upload.pdf")
PNG_OUTPUT_PATH   = os.path.join(OUTPUT_FOLDER, "bas_graphic_transparent.png")
ISO_OUTPUT_PATH   = os.path.join(OUTPUT_FOLDER, "bas_graphic_isometric.png")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def pdf_to_png(pdf_path, out_path):
    doc  = fitz.open(pdf_path)
    page = doc[0]
    pix  = page.get_pixmap(matrix=fitz.Matrix(3, 3))
    pix.save(out_path)
    doc.close()


# ── 1. Clean — remove text/symbols, keep structural lines ──
def clean_plan_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    cleaned = np.zeros_like(binary)
    for i in range(1, num_labels):
        w    = stats[i, cv2.CC_STAT_WIDTH]
        h    = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        asp  = max(w, h) / (min(w, h) + 1)
        if (asp > 4 and area > 80) or (area > 500 and w > 40 and h > 40):
            cleaned[labels == i] = 255
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return gray, cleaned


# ── 2. Detect walls as horizontal/vertical vector lines ──
def detect_walls(cleaned, h, w):
    edges    = cv2.Canny(cleaned, 50, 150)
    raw      = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60,
                                minLineLength=50, maxLineGap=18)
    wall_lines = []
    if raw is not None:
        for ln in raw:
            x1, y1, x2, y2 = ln[0]
            dx  = x2 - x1; dy = y2 - y1
            length = math.sqrt(dx*dx + dy*dy)
            if length < 50: continue
            ang = abs(math.degrees(math.atan2(dy, dx)))
            if ang < 12 or ang > 168 or (78 < ang < 102):
                wall_lines.append((x1, y1, x2, y2, length))
    mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2, length in wall_lines:
        cv2.line(mask, (x1, y1), (x2, y2), 255, 10 if length > 150 else 6)
    k    = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask, wall_lines


# ── 3. Detect floor/room interiors ──
def detect_floor(wall_mask, h, w):
    inv = cv2.bitwise_not(wall_mask)
    nl, labels, stats, _ = cv2.connectedComponentsWithStats(inv)
    floor = np.zeros((h, w), dtype=np.uint8)
    img_area = h * w
    for i in range(1, nl):
        area = stats[i, cv2.CC_STAT_AREA]
        rw   = stats[i, cv2.CC_STAT_WIDTH]
        rh   = stats[i, cv2.CC_STAT_HEIGHT]
        if area < 2000 or area > img_area * 0.85: continue
        if rw < 40 or rh < 40: continue
        floor[labels == i] = 255
    k     = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    floor = cv2.morphologyEx(floor, cv2.MORPH_CLOSE, k)
    return floor


# ── 4. Detect HVAC equipment by color ──
def detect_hvac(img):
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    r1   = cv2.inRange(hsv, np.array([0,60,50]),   np.array([12,255,255]))
    r2   = cv2.inRange(hsv, np.array([165,60,50]), np.array([180,255,255]))
    ducts = cv2.dilate(r1 + r2, np.ones((6,6), np.uint8), iterations=2)
    blue  = cv2.dilate(cv2.inRange(hsv, np.array([95,50,40]),  np.array([135,255,255])),
                       np.ones((5,5), np.uint8), iterations=2)
    cyan  = cv2.dilate(cv2.inRange(hsv, np.array([85,40,40]),  np.array([100,255,255])),
                       np.ones((4,4), np.uint8), iterations=1)
    green = cv2.dilate(cv2.inRange(hsv, np.array([40,40,40]),  np.array([85,255,255])),
                       np.ones((4,4), np.uint8), iterations=1)
    return ducts, blue, cyan, green


# ── 5. Build flat 2D BAS canvas ──
def build_2d(h, w, floor, walls, ducts, blue, cyan, green):
    c = np.zeros((h, w, 4), dtype=np.uint8)
    c[floor  > 0] = (45,  50,  62,  120)
    c[walls  > 0] = (160, 162, 172, 255)
    c[ducts  > 0] = (210, 210, 220, 255)
    c[cyan   > 0] = (255, 200,   0, 255)
    c[green  > 0] = ( 80, 220,  80, 255)
    c[blue   > 0] = (255, 100,  30, 255)
    return c


# ── 6. True isometric 3D with wall extrusion + VAV cubes ──
def build_iso(canvas, wall_mask, floor_mask):
    h, w  = canvas.shape[:2]
    rad   = math.radians(26.565)

    def proj(x, y, z=0):
        return (x - y) * math.cos(rad), (x + y) * math.sin(rad) - z

    out_w, out_h = int(w * 1.4), int(h * 1.2)
    out     = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    ox, oy  = out_w * 0.5, out_h * 0.35
    WH      = 40   # wall height px
    FLOOR_C = (38, 44, 56, 200)
    WALL_S  = (100, 100, 110, 240)
    WALL_T  = (160, 162, 172, 255)

    # Floor tiles
    for y in range(0, h, 3):
        for x in range(0, w, 3):
            if floor_mask[y, x] > 0:
                sx, sy = proj(x, y)
                px, py = int(sx+ox), int(sy+oy)
                if 0 <= px < out_w-3 and 0 <= py < out_h-3:
                    cv2.rectangle(out, (px, py), (px+3, py+3), FLOOR_C, -1)

    # Wall sides (extrusion)
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            if wall_mask[y, x] > 0:
                bx, by = proj(x, y, 0)
                tx, ty = proj(x, y, WH)
                pbx, pby = int(bx+ox), int(by+oy)
                ptx, pty = int(tx+ox), int(ty+oy)
                if 0<=pbx 0:
                tx, ty = proj(x, y, WH)
                ptx, pty = int(tx+ox), int(ty+oy)
                if 0 <= ptx < out_w and 0 <= pty < out_h:
                    cv2.circle(out, (ptx, pty), 2, WALL_T, -1)

    # HVAC overlays (ducts, pipes)
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            p = canvas[y, x]
            if p[3] < 50: continue
            b, g, r, a = int(p[0]), int(p[1]), int(p[2]), int(p[3])
            if (145 200 and p[0] > 200 and p[1] < 120 and p[2] < 50:
                vav_mask[y, x] = 255

    nl, lmap, stats, cents = cv2.connectedComponentsWithStats(vav_mask)
    BS, BH = 12, 18
    CF = (220, 100, 30, 255)
    CT = (255, 140, 60, 255)
    CS = (160,  70, 20, 255)

    def pt(x, y, z):
        sx, sy = proj(x, y, z)
        return [int(sx+ox), int(sy+oy)]

    for i in range(1, nl):
        if stats[i, cv2.CC_STAT_AREA] < 30: continue
        cx, cy = int(cents[i][0]), int(cents[i][1])
        front = np.array([pt(cx-BS,cy+BS,0), pt(cx+BS,cy+BS,0),
                          pt(cx+BS,cy+BS,BH), pt(cx-BS,cy+BS,BH)], np.int32)
        top   = np.array([pt(cx-BS,cy-BS,BH), pt(cx+BS,cy-BS,BH),
                          pt(cx+BS,cy+BS,BH), pt(cx-BS,cy+BS,BH)], np.int32)
        side  = np.array([pt(cx+BS,cy-BS,0), pt(cx+BS,cy+BS,0),
                          pt(cx+BS,cy+BS,BH), pt(cx+BS,cy-BS,BH)], np.int32)
        cv2.fillPoly(out, [front], CF)
        cv2.fillPoly(out, [top],   CT)
        cv2.fillPoly(out, [side],  CS)

    shadow = cv2.GaussianBlur(out, (21, 21), 0)
    return cv2.addWeighted(out, 1.0, shadow, 0.08, 0)


HTML_STYLE = """

"""

HOME_HTML = """





  
⚙️ 🏢 📐

  
BAS Graphic Generator v8

  

HVAC Controls · Mechanical Plans · Synchrony Style Graphics


  

    True Isometric 3D
    Wall Extrusion
    VAV Cubes
    Duct Routing
  

  

    

      
      
Upload mechanical plan — PNG, JPG or PDF

    

    ⚡ Generate BAS Graphic
  

  

Made by Paolo V. & Emmanuel R.





"""


@app.route("/")
def home():
    return HOME_HTML


@app.route("/generate", methods=["POST"])
def generate():
    file     = request.files["file"]
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        file.save(UPLOAD_PDF_PATH)
        pdf_to_png(UPLOAD_PDF_PATH, UPLOAD_IMAGE_PATH)
    else:
        file.save(UPLOAD_IMAGE_PATH)

    img = cv2.imread(UPLOAD_IMAGE_PATH)
    if img is None:
        return "Error loading image", 400

    h, w = img.shape[:2]
    if h > w:
        img  = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h, w = img.shape[:2]

    target_w = 1800
    img = cv2.resize(img, (target_w, int(h * target_w / w)))
    h, w = img.shape[:2]
    cv2.imwrite(UPLOAD_IMAGE_PATH, img)

    gray, cleaned            = clean_plan_image(img)
    wall_mask, _             = detect_walls(cleaned, h, w)
    floor_mask               = detect_floor(wall_mask, h, w)
    ducts, blue, cyan, green = detect_hvac(img)
    canvas_2d                = build_2d(h, w, floor_mask, wall_mask, ducts, blue, cyan, green)
    cv2.imwrite(PNG_OUTPUT_PATH, canvas_2d)

    iso = build_iso(canvas_2d, wall_mask, floor_mask)
    cv2.imwrite(ISO_OUTPUT_PATH, iso)

    o_b64  = image_to_base64(UPLOAD_IMAGE_PATH)
    f_b64  = image_to_base64(PNG_OUTPUT_PATH)
    i_b64  = image_to_base64(ISO_OUTPUT_PATH)

    return f"""
    {HTML_STYLE}
    
      
BAS Graphic Result

      

True Isometric 3D · Wall Extrusion · VAV Cubes · Duct Routing


      
📄 ORIGINAL PLAN

          
🗺️ BAS FLAT VIEW

          
🏢 ISOMETRIC 3D

          

      
⬇️ Download Flat PNG
⬇️ Download 3D PNG
🔄 Generate Another

      
Made by Paolo V. & Emmanuel R.

    
    """


@app.route("/download_flat")
def download_flat():
    return send_file(PNG_OUTPUT_PATH, mimetype="image/png",
                     as_attachment=True, download_name="bas_graphic_flat.png")


@app.route("/download_iso")
def download_iso():
    return send_file(ISO_OUTPUT_PATH, mimetype="image/png",
                     as_attachment=True, download_name="bas_graphic_3d.png")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

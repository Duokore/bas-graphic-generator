from flask import Flask, request, render_template_string, jsonify, redirect, make_response
import os
import base64
import json
import math
import cv2
import numpy as np
import fitz

app = Flask(__name__)

APP_PASSWORD = os.environ.get("APP_PASSWORD", "1234")
SECRET_KEY = os.environ.get("SECRET_KEY", "change_this_secret")
app.secret_key = SECRET_KEY

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
UPLOAD_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, "mechanical_upload.png")
UPLOAD_PDF_PATH = os.path.join(UPLOAD_FOLDER, "mechanical_upload.pdf")
CLEAN_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, "mechanical_clean.png")
ISOLATED_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, "mechanical_isolated.png")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def pdf_to_png(pdf_path, out_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
    pix.save(out_path)
    doc.close()


# ============================================================
# SMART FLOORPLAN ISOLATION (v23.5 NEW)
# ============================================================

def smart_isolate_floorplan(img):
    """Detect and crop ONLY the building floorplan area, ignoring title blocks,
    tables, legends, logos, etc. Returns isolated image and crop bounds."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: Binary threshold to get all dark content
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Step 2: Heavy dilation to create blobs of related content
    # Title blocks, tables, and the floorplan each become separate blobs
    kernel = np.ones((25, 25), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=3)

    # Step 3: Find connected components (each blob = one region of content)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)

    # Step 4: Filter regions - we want the BIGGEST connected blob that's
    # NOT touching the page border (excludes the full page outline)
    candidates = []
    border_margin = 20
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]
        # Skip very small regions
        if area < (w * h * 0.05):
            continue
        # Calculate how "filled" the region is (density)
        density = area / max(ww * hh, 1)

        # Aspect ratio - floorplans are usually wider than tall, or roughly square
        aspect = ww / max(hh, 1)

        # Position score - floorplans are usually in the upper-left or center
        # Title blocks/legends are usually on right/bottom edges
        is_right_edge = (x + ww) > (w - border_margin * 5) and x > w * 0.6
        is_bottom_edge = (y + hh) > (h - border_margin * 5) and y > h * 0.7

        # Skip narrow strips (title blocks are usually tall and thin or wide and short)
        if aspect < 0.3 or aspect > 8:
            continue

        # Skip regions clearly in corner positions where title blocks live
        if is_right_edge and ww < w * 0.3:
            continue
        if is_bottom_edge and hh < h * 0.25:
            continue

        candidates.append({
            "x": x, "y": y, "w": ww, "h": hh,
            "area": area, "density": density,
            "aspect": aspect
        })

    if not candidates:
        # Fallback: use the full image
        return img, (0, 0, w, h)

    # Step 5: Pick the largest valid candidate (this is our floorplan)
    candidates.sort(key=lambda c: -c["area"])
    best = candidates[0]

    # Step 6: Crop with small padding
    pad = 20
    x = max(0, best["x"] - pad)
    y = max(0, best["y"] - pad)
    x2 = min(w, best["x"] + best["w"] + pad)
    y2 = min(h, best["y"] + best["h"] + pad)

    cropped = img[y:y2, x:x2]

    return cropped, (x, y, x2 - x, y2 - y)


# ============================================================
# ARCHITECTURE DETECTION ENGINE (improved)
# ============================================================

def remove_text_from_plan(img_gray):
    """Remove text and small annotations using connected components filter."""
    _, binary = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    cleaned = np.zeros_like(binary)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        aspect = max(w, h) / max(min(w, h), 1)
        # Keep only LONG lines or LARGE rectangles (walls), reject text/symbols
        if (aspect > 5 and area > 100) or (area > 800 and max(w, h) > 60):
            cleaned[labels == i] = 255

    return cleaned


def detect_walls_hough(binary_clean, min_line_length=60, max_line_gap=15):
    """Detect straight wall lines using Hough Transform."""
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))

    horizontal = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, kernel_h, iterations=1)
    vertical = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, kernel_v, iterations=1)

    combined = cv2.bitwise_or(horizontal, vertical)

    lines = cv2.HoughLinesP(
        combined,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    return lines if lines is not None else []


def snap_to_orthogonal(lines, angle_tolerance=5):
    """Snap lines to perfectly horizontal or vertical."""
    snapped = []
    for line in lines:
        try:
            if hasattr(line, '__len__') and len(line) == 4:
                x1, y1, x2, y2 = line
            elif hasattr(line, '__len__') and len(line) == 1:
                x1, y1, x2, y2 = line[0]
            else:
                continue
        except (TypeError, ValueError):
            continue

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        angle = math.degrees(math.atan2(dy, dx))

        if angle < 0:
            angle += 180

        if angle < angle_tolerance or angle > 180 - angle_tolerance:
            y_avg = (y1 + y2) // 2
            snapped.append([min(x1, x2), y_avg, max(x1, x2), y_avg])
        elif abs(angle - 90) < angle_tolerance:
            x_avg = (x1 + x2) // 2
            snapped.append([x_avg, min(y1, y2), x_avg, max(y1, y2)])

    return snapped


def merge_collinear_lines(lines, distance_threshold=12):
    """Merge lines that are collinear and close to each other."""
    if not lines:
        return []

    horizontal = [l for l in lines if l[1] == l[3]]
    vertical = [l for l in lines if l[0] == l[2]]

    merged = []

    # Group horizontal lines
    horizontal.sort(key=lambda l: (l[1], l[0]))
    h_groups = []
    for line in horizontal:
        added = False
        for group in h_groups:
            if abs(line[1] - group[0][1]) < distance_threshold:
                # Also check if x ranges overlap or are close
                gx_min = min(l[0] for l in group)
                gx_max = max(l[2] for l in group)
                if line[0] < gx_max + 50 and line[2] > gx_min - 50:
                    group.append(line)
                    added = True
                    break
        if not added:
            h_groups.append([line])

    for group in h_groups:
        y_avg = sum(l[1] for l in group) // len(group)
        x_min = min(l[0] for l in group)
        x_max = max(l[2] for l in group)
        merged.append([x_min, y_avg, x_max, y_avg])

    # Group vertical lines
    vertical.sort(key=lambda l: (l[0], l[1]))
    v_groups = []
    for line in vertical:
        added = False
        for group in v_groups:
            if abs(line[0] - group[0][0]) < distance_threshold:
                gy_min = min(l[1] for l in group)
                gy_max = max(l[3] for l in group)
                if line[1] < gy_max + 50 and line[3] > gy_min - 50:
                    group.append(line)
                    added = True
                    break
        if not added:
            v_groups.append([line])

    for group in v_groups:
        x_avg = sum(l[0] for l in group) // len(group)
        y_min = min(l[1] for l in group)
        y_max = max(l[3] for l in group)
        merged.append([x_avg, y_min, x_avg, y_max])

    return merged


def find_building_footprint(lines, img_shape):
    """Find the tight bounding box of the actual walls (not page borders)."""
    if not lines:
        return None

    h, w = img_shape[:2]

    # Filter out lines that are too close to image edges (likely page borders)
    edge_margin = 15
    valid_lines = []
    for l in lines:
        x1, y1, x2, y2 = l
        # Reject if line is on the very edge of the image
        if (x1 < edge_margin and x2 < edge_margin) or \
           (x1 > w - edge_margin and x2 > w - edge_margin) or \
           (y1 < edge_margin and y2 < edge_margin) or \
           (y1 > h - edge_margin and y2 > h - edge_margin):
            continue
        valid_lines.append(l)

    if not valid_lines:
        valid_lines = lines

    # Find tight bounding box
    all_x = [l[0] for l in valid_lines] + [l[2] for l in valid_lines]
    all_y = [l[1] for l in valid_lines] + [l[3] for l in valid_lines]

    return {
        "min_x": min(all_x),
        "max_x": max(all_x),
        "min_y": min(all_y),
        "max_y": max(all_y)
    }


def classify_exterior_vs_interior(lines, footprint):
    """Identify which lines form the exterior perimeter vs interior walls."""
    if not lines or not footprint:
        return [], []

    edge_tolerance = 30

    exterior_lines = []
    interior_lines = []

    for line in lines:
        x1, y1, x2, y2 = line

        # Check if line is near the footprint boundary
        near_top = max(y1, y2) < footprint["min_y"] + edge_tolerance
        near_bottom = min(y1, y2) > footprint["max_y"] - edge_tolerance
        near_left = max(x1, x2) < footprint["min_x"] + edge_tolerance
        near_right = min(x1, x2) > footprint["max_x"] - edge_tolerance

        near_edge = near_top or near_bottom or near_left or near_right

        length = math.hypot(x2 - x1, y2 - y1)

        if near_edge and length > 80:
            exterior_lines.append(line)
        else:
            interior_lines.append(line)

    return exterior_lines, interior_lines


def build_exterior_polygon(footprint):
    """Construct a closed polygon from the footprint bounds."""
    if not footprint:
        return None
    return [
        {"x": int(footprint["min_x"]), "y": int(footprint["min_y"])},
        {"x": int(footprint["max_x"]), "y": int(footprint["min_y"])},
        {"x": int(footprint["max_x"]), "y": int(footprint["max_y"])},
        {"x": int(footprint["min_x"]), "y": int(footprint["max_y"])}
    ]


def detect_architecture(image_path):
    """Main detection pipeline with smart floorplan isolation."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    # === STEP 0: Smart Floorplan Isolation ===
    isolated, crop_box = smart_isolate_floorplan(img)
    cv2.imwrite(ISOLATED_IMAGE_PATH, isolated)

    # Replace the upload image with the isolated version so it shows correctly in editor
    cv2.imwrite(image_path, isolated)

    img_h, img_w = isolated.shape[:2]
    gray = cv2.cvtColor(isolated, cv2.COLOR_BGR2GRAY)

    # Step 1: Remove text and small components
    binary_clean = remove_text_from_plan(gray)
    cv2.imwrite(CLEAN_IMAGE_PATH, binary_clean)

    # Step 2: Detect lines with Hough
    raw_lines = detect_walls_hough(binary_clean)

    if len(raw_lines) == 0:
        return {
            "image_width": img_w,
            "image_height": img_h,
            "elements": [],
            "stats": {"lines_raw": 0, "lines_merged": 0, "exterior": 0, "interior": 0,
                      "crop": crop_box}
        }

    # Step 3: Snap to orthogonal
    snapped = snap_to_orthogonal(raw_lines)

    # Step 4: Merge collinear lines
    merged = merge_collinear_lines(snapped, distance_threshold=15)

    # Step 5: Filter very short lines
    merged = [l for l in merged if math.hypot(l[2] - l[0], l[3] - l[1]) > 60]

    # Step 6: Find building footprint
    footprint = find_building_footprint(merged, (img_h, img_w))

    # Step 7: Classify exterior vs interior
    exterior_lines, interior_lines = classify_exterior_vs_interior(merged, footprint)

    # Step 8: Build elements
    elements = []
    ext_polygon = build_exterior_polygon(footprint)
    if ext_polygon:
        elements.append({
            "type": "extwall",
            "points": ext_polygon,
            "closed": True,
            "detected": True
        })

    # Step 9: Add interior walls
    for line in interior_lines:
        elements.append({
            "type": "intwall",
            "points": [
                {"x": int(line[0]), "y": int(line[1])},
                {"x": int(line[2]), "y": int(line[3])}
            ],
            "detected": True
        })

    return {
        "image_width": img_w,
        "image_height": img_h,
        "elements": elements,
        "stats": {
            "lines_raw": len(raw_lines),
            "lines_merged": len(merged),
            "exterior": len(exterior_lines),
            "interior": len(interior_lines),
            "crop": crop_box
        }
    }


# ============================================================
# COLOR DETECTION (kept from previous version)
# ============================================================

def clean_mask(mask, iterations=2):
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def get_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def auto_detect_colors(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elements = []

    blue_mask = cv2.inRange(hsv, np.array([90, 60, 60]), np.array([140, 255, 255]))
    blue_mask = clean_mask(blue_mask, 1)
    for cnt in get_contours(blue_mask):
        area = cv2.contourArea(cnt)
        if area < 40 or area > 5000:
            continue
        x, y, ww, hh = cv2.boundingRect(cnt)
        elements.append({"type": "vav", "x": int(x + ww / 2), "y": int(y + hh / 2)})

    green_mask = cv2.inRange(hsv, np.array([40, 60, 60]), np.array([85, 255, 255]))
    green_mask = clean_mask(green_mask, 1)
    candidates = []
    for cnt in get_contours(green_mask):
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        x, y, ww, hh = cv2.boundingRect(cnt)
        candidates.append({"area": area, "x": int(x + ww / 2), "y": int(y + hh / 2)})
    candidates.sort(key=lambda a: -a["area"])
    for a in candidates[:1]:
        elements.append({"type": "ahu", "x": a["x"], "y": a["y"]})

    red1 = cv2.inRange(hsv, np.array([0, 60, 60]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([170, 60, 60]), np.array([180, 255, 255]))
    red_mask = clean_mask(cv2.bitwise_or(red1, red2), 1)
    for cnt in get_contours(red_mask):
        area = cv2.contourArea(cnt)
        if area < 80:
            continue
        x, y, ww, hh = cv2.boundingRect(cnt)
        if ww > hh:
            elements.append({
                "type": "duct",
                "points": [{"x": x, "y": int(y + hh / 2)}, {"x": x + ww, "y": int(y + hh / 2)}]
            })
        else:
            elements.append({
                "type": "duct",
                "points": [{"x": int(x + ww / 2), "y": y}, {"x": int(x + ww / 2), "y": y + hh}]
            })
    return {"image_width": w, "image_height": h, "elements": elements}


# ============================================================
# HTML PAGES
# ============================================================

LOGIN_PAGE = '''<!DOCTYPE html>
<html><head><title>Login</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{background:#0d0f14;color:white;font-family:'Segoe UI',Arial,sans-serif;min-height:100vh;display:flex;align-items:center;justify-content:center;}
.card{background:#181b24;border:1px solid #2a3050;border-radius:24px;padding:45px;width:420px;text-align:center;box-shadow:0 0 60px rgba(0,0,0,0.65);}
.logo{font-size:48px;margin-bottom:12px;}
h1{font-size:26px;margin-bottom:6px;background:linear-gradient(135deg,#2d89ef,#b388ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.sub{color:#8b93ad;font-size:14px;margin-bottom:24px;}
input{width:100%;padding:15px;border-radius:12px;border:1px solid #2a3050;background:#10131a;color:white;font-size:16px;outline:none;}
input:focus{border-color:#2d89ef;}
button{width:100%;padding:15px;border:none;border-radius:12px;margin-top:18px;background:linear-gradient(135deg,#1a6fd4,#2d89ef);color:white;font-size:16px;font-weight:700;cursor:pointer;}
.error{margin-top:14px;color:#ff6b6b;font-size:13px;}
.footer{color:#3a4060;font-size:11px;margin-top:24px;}
</style></head><body>
<div class="card">
<div class="logo">&#128274;</div>
<h1>BAS Generator v23.5</h1>
<p class="sub">Private Access</p>
<form method="POST" action="/login">
<input type="password" name="password" placeholder="Enter password" required autofocus>
<button type="submit">Login</button>
</form>
{% if error %}<div class="error">Invalid password. Try again.</div>{% endif %}
<div class="footer">Made by Paolo V. R.</div>
</div></body></html>'''


HOME_PAGE = '''<!DOCTYPE html>
<html><head><title>BAS Generator v23.5</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{background:#0d0f14;color:white;font-family:'Segoe UI',Arial,sans-serif;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px;}
.card{background:#181b24;border:1px solid #2a2f3e;border-radius:24px;padding:36px;text-align:center;max-width:780px;width:100%;box-shadow:0 0 60px rgba(0,0,0,0.6);}
.logo{font-size:48px;margin-bottom:12px;}
h1{font-size:28px;margin-bottom:6px;background:linear-gradient(135deg,#2d89ef,#b388ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.sub{color:#7a8099;font-size:13px;margin-bottom:18px;}
.zone{border:2px dashed #2d3348;border-radius:14px;padding:24px;margin-bottom:14px;background:#13151d;}
.zone:hover{border-color:#2d89ef;}
input[type=file]{background:transparent;color:#aab0c4;border:none;font-size:13px;width:100%;cursor:pointer;}
.lbl{display:block;font-size:11px;color:#5a6280;margin-top:6px;}
.option-row{display:flex;gap:10px;margin-bottom:14px;}
.option-btn{flex:1;padding:14px 10px;background:#13151d;border:2px solid #2a3050;border-radius:10px;color:white;font-size:12px;font-weight:700;cursor:pointer;transition:all 0.2s;line-height:1.3;}
.option-btn small{display:block;font-weight:400;font-size:10px;color:#8a92a8;margin-top:4px;}
.option-btn:hover{border-color:#2d89ef;}
.option-btn.active{background:linear-gradient(135deg,#1a6fd4,#2d89ef);border-color:#2d89ef;}
.option-btn.active small{color:#bcdaff;}
.btn{background:linear-gradient(135deg,#1a6fd4,#2d89ef);color:white;border:none;border-radius:12px;padding:15px 40px;font-size:15px;font-weight:700;cursor:pointer;width:100%;}
.feature-row{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:14px;}
.feature{background:#13151d;border:1px solid #2a3050;border-radius:8px;padding:8px;text-align:left;font-size:10px;color:#aab0c4;display:flex;gap:6px;align-items:center;}
.color-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;}
.footer{color:#3a4060;font-size:10px;margin-top:14px;}
.badge{display:inline-block;background:linear-gradient(135deg,#ff9800,#ff5722);color:white;padding:2px 9px;font-size:10px;border-radius:6px;margin-left:6px;}
.tip{background:#1a1d28;border-left:3px solid #16a34a;padding:8px 12px;margin-bottom:12px;font-size:11px;color:#aab0c4;text-align:left;border-radius:5px;}
</style></head><body>
<div class="card">
<div class="logo">&#127970;</div>
<h1>BAS Generator v23.5 <span class="badge">SMART ISOLATION</span></h1>
<p class="sub">Auto-detect building floorplan + HVAC</p>

<div class="tip">
<b>v23.5 NEW:</b> Smart Floorplan Isolation removes title blocks, tables, and legends before detecting walls.
</div>

<form action="/upload" method="post" enctype="multipart/form-data" id="uploadForm">
<input type="hidden" name="mode" id="modeInput" value="manual">

<div class="option-row">
<button type="button" class="option-btn active" id="manualBtn" onclick="setMode('manual')">
Manual Editor
<small>Draw everything yourself</small>
</button>
<button type="button" class="option-btn" id="archBtn" onclick="setMode('arch')">
Smart Auto-Detect
<small>Isolates floorplan + detects walls</small>
</button>
<button type="button" class="option-btn" id="colorBtn" onclick="setMode('color')">
Auto-Detect Colors
<small>Detects HVAC if pre-marked</small>
</button>
</div>

<div class="feature-row">
<div class="feature"><div class="color-dot" style="background:#1e40af"></div> Blue = VAVs</div>
<div class="feature"><div class="color-dot" style="background:#16a34a"></div> Green = AHU</div>
<div class="feature"><div class="color-dot" style="background:#dc2626"></div> Red = Ducts</div>
<div class="feature"><div class="color-dot" style="background:#9333ea"></div> Purple = Ext walls</div>
</div>

<div class="zone">
<input type="file" name="file" accept="image/png,image/jpeg,application/pdf" required>
<span class="lbl">Upload your plan (PDF or PNG)</span>
</div>
<button class="btn" type="submit">Open Editor</button>
</form>

<div class="footer">Made by Paolo V. R.</div>
</div>

<script>
function setMode(mode){
    document.getElementById('manualBtn').classList.toggle('active', mode==='manual');
    document.getElementById('archBtn').classList.toggle('active', mode==='arch');
    document.getElementById('colorBtn').classList.toggle('active', mode==='color');
    document.getElementById('modeInput').value = mode;
}
</script>
</body></html>'''


EDITOR_PAGE = '''<!DOCTYPE html>
<html><head><title>CAD Editor v23.5</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{background:#0d0f14;color:white;font-family:'Segoe UI',Arial,sans-serif;padding:8px;height:100vh;display:flex;flex-direction:column;overflow:hidden;}
.topbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;}
h1{font-size:16px;background:linear-gradient(135deg,#2d89ef,#b388ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.toolbar{background:#181b24;border:1px solid #252a38;border-radius:10px;padding:8px;display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-bottom:8px;}
.tool-btn{padding:8px 12px;border:2px solid transparent;background:#1e2233;color:white;border-radius:8px;cursor:pointer;font-size:12px;font-weight:600;display:flex;align-items:center;gap:6px;white-space:nowrap;}
.tool-btn:hover{background:#252a38;}
.tool-btn.active{border-color:#fff;background:#2d3348;}
.color-swatch{width:14px;height:14px;border-radius:3px;border:1px solid rgba(255,255,255,0.3);}
.divider{width:1px;background:#333;height:24px;margin:0 3px;}
.canvas-wrap{flex:1;position:relative;background:#1a1a1a;border-radius:10px;border:1px solid #2a3050;overflow:hidden;}
#canvasContainer{width:100%;height:100%;position:relative;overflow:auto;}
canvas{display:block;}
.action-btn{padding:8px 16px;border:none;border-radius:8px;font-size:13px;font-weight:700;cursor:pointer;}
.btn-green{background:#16a34a;color:white;}
.btn-red{background:#dc2626;color:white;}
.btn-gray{background:#333;color:white;}
.btn-purple{background:#9333ea;color:white;}
.spinner{display:inline-block;width:20px;height:20px;border:3px solid #fff;border-top-color:transparent;border-radius:50%;animation:spin 1s linear infinite;}
@keyframes spin{to{transform:rotate(360deg);}}
.loading-overlay{position:fixed;inset:0;background:rgba(0,0,0,0.85);display:none;align-items:center;justify-content:center;z-index:100;flex-direction:column;gap:16px;}
.loading-overlay.active{display:flex;}
.status{padding:4px 12px;background:#1e2233;border-radius:6px;font-size:11px;color:#aab0c4;min-width:200px;text-align:center;}
.auto-banner{background:linear-gradient(135deg,#16a34a22,#16a34a44);border:1px solid #16a34a;border-radius:8px;padding:8px 14px;font-size:12px;color:#4ade80;margin-bottom:8px;}
.cursor-cross{cursor:crosshair;}
.cursor-move{cursor:move;}
</style></head><body>

{% if detected_message %}
<div class="auto-banner">&#10004; {{ detected_message }}</div>
{% endif %}

<div class="topbar">
<h1>CAD Editor v23.5</h1>
<div style="display:flex;gap:6px;">
<button onclick="undo()" class="action-btn btn-gray">&#8617; Undo</button>
<button onclick="clearAll()" class="action-btn btn-red">Clear</button>
<button onclick="autoBranchDiffusers()" class="action-btn btn-purple">Auto-Connect Diffusers</button>
<button onclick="generate()" class="action-btn btn-green">Generate &rarr;</button>
</div>
</div>

<div class="toolbar">
<button class="tool-btn active" data-tool="extwall" onclick="selectTool(this)">
<div class="color-swatch" style="background:#9333ea"></div> Ext Wall
</button>
<button class="tool-btn" data-tool="intwall" onclick="selectTool(this)">
<div class="color-swatch" style="background:#ea580c"></div> Int Wall
</button>
<button class="tool-btn" data-tool="duct" onclick="selectTool(this)">
<div class="color-swatch" style="background:#fff;border:1px solid #888"></div> Duct
</button>

<div class="divider"></div>

<button class="tool-btn" data-tool="vav" onclick="selectTool(this)">
<div class="color-swatch" style="background:#1e40af"></div> VAV
</button>
<button class="tool-btn" data-tool="ahu" onclick="selectTool(this)">
<div class="color-swatch" style="background:#16a34a"></div> AHU
</button>
<button class="tool-btn" data-tool="diffuser" onclick="selectTool(this)">
<div class="color-swatch" style="background:#fff;border:1px solid #888"></div> Diffuser
</button>

<div class="divider"></div>

<button class="tool-btn" data-tool="move" onclick="selectTool(this)">&#9874; Move</button>
<button class="tool-btn" data-tool="delete" onclick="selectTool(this)">&#128465; Delete</button>

<div class="divider"></div>

<span class="status" id="statusBar">Click corners to draw walls. Double-click to finish.</span>
</div>

<div class="canvas-wrap">
<div id="canvasContainer">
<canvas id="bgCanvas" style="position:absolute;top:0;left:0;"></canvas>
<canvas id="drawCanvas" class="cursor-cross" style="position:absolute;top:0;left:0;"></canvas>
</div>
</div>

<div class="loading-overlay" id="loading">
<div class="spinner"></div>
<div style="color:white;font-size:14px;">Processing...</div>
</div>

<script>
const imgB64 = '{{ image_b64 }}';
const initialElements = {{ initial_elements | safe }};

let bgCanvas = document.getElementById('bgCanvas');
let drawCanvas = document.getElementById('drawCanvas');
let bgCtx = bgCanvas.getContext('2d');
let drawCtx = drawCanvas.getContext('2d');

let currentTool = 'extwall';
let elements = initialElements;
let history = [];
let currentPolyline = null;
let hoverPoint = null;
let selectedElement = null;
let dragOffset = null;

const COLORS = {
    extwall:'#9333ea', intwall:'#ea580c', duct:'#dcdce0',
    vav:'#1e40af', ahu:'#16a34a', diffuser:'#ffffff'
};

const STATUS_TEXTS = {
    extwall:'Click corners of building PERIMETER. Double-click to close.',
    intwall:'Click corners of an INTERIOR WALL. Double-click to finish.',
    duct:'Click TWO points for a straight duct line.',
    vav:'Click to place a VAV.',
    ahu:'Click to place the AHU.',
    diffuser:'Click to place a diffuser.',
    move:'Click and drag any element to move it.',
    delete:'Click any element to delete it.'
};

const img = new Image();
img.onload = function(){
    bgCanvas.width = img.width;
    bgCanvas.height = img.height;
    drawCanvas.width = img.width;
    drawCanvas.height = img.height;
    bgCtx.drawImage(img, 0, 0);
    saveState();
    redraw();
};
img.src = 'data:image/png;base64,' + imgB64;

function selectTool(btn){
    document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentTool = btn.dataset.tool;
    document.getElementById('statusBar').textContent = STATUS_TEXTS[currentTool] || '';
    if(currentPolyline){
        if(currentPolyline.points.length >= 2) elements.push(currentPolyline);
        currentPolyline = null;
        saveState();
    }
    drawCanvas.className = '';
    if(currentTool === 'move') drawCanvas.classList.add('cursor-move');
    else drawCanvas.classList.add('cursor-cross');
    redraw();
}

function getMousePos(e){
    const rect = drawCanvas.getBoundingClientRect();
    const sx = drawCanvas.width / rect.width;
    const sy = drawCanvas.height / rect.height;
    return { x: (e.clientX - rect.left) * sx, y: (e.clientY - rect.top) * sy };
}

drawCanvas.addEventListener('click', function(e){
    const pos = getMousePos(e);
    if(currentTool === 'delete'){
        const idx = findElementAt(pos);
        if(idx !== -1){ elements.splice(idx, 1); saveState(); redraw(); }
        return;
    }
    if(currentTool === 'move') return;
    if(currentTool === 'vav' || currentTool === 'ahu' || currentTool === 'diffuser'){
        elements.push({ type: currentTool, x: pos.x, y: pos.y });
        saveState(); redraw(); return;
    }
    if(currentTool === 'extwall' || currentTool === 'intwall'){
        if(!currentPolyline){
            currentPolyline = { type: currentTool, points: [{ x: pos.x, y: pos.y }] };
        } else {
            currentPolyline.points.push({ x: pos.x, y: pos.y });
        }
        redraw(); return;
    }
    if(currentTool === 'duct'){
        if(!currentPolyline){
            currentPolyline = { type: 'duct', points: [{ x: pos.x, y: pos.y }] };
        } else {
            currentPolyline.points.push({ x: pos.x, y: pos.y });
            elements.push(currentPolyline);
            currentPolyline = null;
            saveState();
        }
        redraw(); return;
    }
});

drawCanvas.addEventListener('dblclick', function(e){
    if(currentPolyline && currentPolyline.points && currentPolyline.points.length >= 2){
        if(currentPolyline.type === 'extwall' && currentPolyline.points.length >= 3){
            currentPolyline.closed = true;
        }
        elements.push(currentPolyline);
        currentPolyline = null;
        saveState();
        redraw();
    }
});

drawCanvas.addEventListener('mousemove', function(e){
    const pos = getMousePos(e);
    hoverPoint = pos;
    if(currentTool === 'move' && selectedElement && dragOffset){
        moveElement(selectedElement, pos.x - dragOffset.x, pos.y - dragOffset.y);
        const c = getElementCenter(selectedElement);
        dragOffset = { x: pos.x - c.x, y: pos.y - c.y };
        redraw();
        return;
    }
    if(currentPolyline) redraw();
});

drawCanvas.addEventListener('mousedown', function(e){
    if(currentTool !== 'move') return;
    const pos = getMousePos(e);
    const idx = findElementAt(pos);
    if(idx !== -1){
        selectedElement = elements[idx];
        const c = getElementCenter(selectedElement);
        dragOffset = { x: pos.x - c.x, y: pos.y - c.y };
    }
});

drawCanvas.addEventListener('mouseup', function(){
    if(selectedElement){ saveState(); selectedElement = null; dragOffset = null; }
});

document.addEventListener('keydown', function(e){
    if(e.key === 'Escape' && currentPolyline){ currentPolyline = null; redraw(); }
});

function findElementAt(pos){
    for(let i = elements.length - 1; i >= 0; i--){
        const el = elements[i];
        if(el.type === 'vav' || el.type === 'ahu' || el.type === 'diffuser'){
            if(Math.hypot(pos.x - el.x, pos.y - el.y) < 25) return i;
        } else if(el.points){
            for(const p of el.points){
                if(Math.hypot(pos.x - p.x, pos.y - p.y) < 15) return i;
            }
        }
    }
    return -1;
}

function getElementCenter(el){
    if(el.type === 'vav' || el.type === 'ahu' || el.type === 'diffuser'){
        return { x: el.x, y: el.y };
    }
    if(!el.points) return { x: 0, y: 0 };
    let sx = 0, sy = 0;
    for(const p of el.points){ sx += p.x; sy += p.y; }
    return { x: sx / el.points.length, y: sy / el.points.length };
}

function moveElement(el, dx, dy){
    if(el.type === 'vav' || el.type === 'ahu' || el.type === 'diffuser'){
        el.x += dx; el.y += dy;
    } else if(el.points){
        for(const p of el.points){ p.x += dx; p.y += dy; }
    }
}

function autoBranchDiffusers(){
    const ducts = elements.filter(e => e.type === 'duct' && e.points && e.points.length >= 2);
    const diffusers = elements.filter(e => e.type === 'diffuser');
    if(ducts.length === 0){ alert('Draw ducts first!'); return; }
    if(diffusers.length === 0){ alert('Place diffusers first!'); return; }
    elements = elements.filter(e => e.type !== 'branch');
    diffusers.forEach(diff => {
        let bestDist = Infinity, bestPoint = null;
        ducts.forEach(duct => {
            for(let i = 0; i < duct.points.length - 1; i++){
                const np = nearestPointOnSegment(diff, duct.points[i], duct.points[i+1]);
                const d = Math.hypot(np.x - diff.x, np.y - diff.y);
                if(d < bestDist){ bestDist = d; bestPoint = np; }
            }
        });
        if(bestPoint && bestDist < 200){
            elements.push({ type: 'branch', points: [{ x: diff.x, y: diff.y }, bestPoint] });
        }
    });
    saveState();
    redraw();
}

function nearestPointOnSegment(p, a, b){
    const dx = b.x - a.x, dy = b.y - a.y;
    const ls = dx*dx + dy*dy;
    if(ls < 0.01) return { x: a.x, y: a.y };
    let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / ls;
    t = Math.max(0, Math.min(1, t));
    return { x: a.x + t*dx, y: a.y + t*dy };
}

function redraw(){
    drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    for(const el of elements) drawElement(el);
    if(currentPolyline){
        drawElement(currentPolyline, true);
        if(hoverPoint && currentPolyline.points && currentPolyline.points.length > 0){
            const last = currentPolyline.points[currentPolyline.points.length - 1];
            drawCtx.strokeStyle = COLORS[currentPolyline.type] || '#fff';
            drawCtx.lineWidth = 3;
            drawCtx.setLineDash([8, 6]);
            drawCtx.beginPath();
            drawCtx.moveTo(last.x, last.y);
            drawCtx.lineTo(hoverPoint.x, hoverPoint.y);
            drawCtx.stroke();
            drawCtx.setLineDash([]);
        }
    }
}

function drawElement(el, inProgress = false){
    const color = COLORS[el.type] || '#fff';
    const detectedAlpha = el.detected ? 0.7 : 1.0;

    if(el.type === 'vav'){
        drawCtx.fillStyle = color;
        drawCtx.strokeStyle = '#fff';
        drawCtx.lineWidth = 2;
        drawCtx.beginPath();
        drawCtx.arc(el.x, el.y, 10, 0, Math.PI * 2);
        drawCtx.fill();
        drawCtx.stroke();
        return;
    }
    if(el.type === 'ahu'){
        drawCtx.fillStyle = color;
        drawCtx.strokeStyle = '#fff';
        drawCtx.lineWidth = 2;
        drawCtx.fillRect(el.x - 20, el.y - 15, 40, 30);
        drawCtx.strokeRect(el.x - 20, el.y - 15, 40, 30);
        return;
    }
    if(el.type === 'diffuser'){
        drawCtx.fillStyle = color;
        drawCtx.strokeStyle = '#666';
        drawCtx.lineWidth = 1.5;
        drawCtx.fillRect(el.x - 5, el.y - 5, 10, 10);
        drawCtx.strokeRect(el.x - 5, el.y - 5, 10, 10);
        return;
    }
    if(el.type === 'branch'){
        if(!el.points || el.points.length < 2) return;
        drawCtx.strokeStyle = '#888';
        drawCtx.lineWidth = 2;
        drawCtx.setLineDash([3, 3]);
        drawCtx.beginPath();
        drawCtx.moveTo(el.points[0].x, el.points[0].y);
        drawCtx.lineTo(el.points[1].x, el.points[1].y);
        drawCtx.stroke();
        drawCtx.setLineDash([]);
        return;
    }
    if(!el.points || el.points.length === 0) return;

    drawCtx.strokeStyle = color;
    drawCtx.globalAlpha = detectedAlpha;
    drawCtx.lineWidth = el.type === 'duct' ? 4 : 5;
    drawCtx.lineCap = 'round';
    drawCtx.lineJoin = 'round';
    drawCtx.beginPath();
    drawCtx.moveTo(el.points[0].x, el.points[0].y);
    for(let i = 1; i < el.points.length; i++){
        drawCtx.lineTo(el.points[i].x, el.points[i].y);
    }
    if(el.closed) drawCtx.closePath();
    drawCtx.stroke();
    drawCtx.globalAlpha = 1.0;

    drawCtx.fillStyle = color;
    for(const p of el.points){
        drawCtx.beginPath();
        drawCtx.arc(p.x, p.y, 5, 0, Math.PI * 2);
        drawCtx.fill();
    }
    if(inProgress && el.points.length > 0){
        const first = el.points[0];
        drawCtx.strokeStyle = '#fff';
        drawCtx.lineWidth = 2;
        drawCtx.beginPath();
        drawCtx.arc(first.x, first.y, 8, 0, Math.PI * 2);
        drawCtx.stroke();
    }
}

function saveState(){
    history.push(JSON.stringify(elements));
    if(history.length > 40) history.shift();
}

function undo(){
    if(history.length < 2) return;
    history.pop();
    elements = JSON.parse(history[history.length - 1]);
    currentPolyline = null;
    redraw();
}

function clearAll(){
    if(!confirm('Clear everything?')) return;
    elements = [];
    currentPolyline = null;
    saveState();
    redraw();
}

async function generate(){
    if(currentPolyline && currentPolyline.points && currentPolyline.points.length >= 2){
        elements.push(currentPolyline);
        currentPolyline = null;
    }
    document.getElementById('loading').classList.add('active');
    try {
        const response = await fetch('/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                elements: elements,
                imageWidth: drawCanvas.width,
                imageHeight: drawCanvas.height
            })
        });
        const result = await response.json();
        if(result.success){ window.location.href = '/result'; }
        else { alert('Error: ' + result.error); document.getElementById('loading').classList.remove('active'); }
    } catch (err) {
        alert('Error: ' + err.message);
        document.getElementById('loading').classList.remove('active');
    }
}
</script>
</body></html>'''


RESULT_PAGE = '''<!DOCTYPE html>
<html><head><title>BAS Graphic v23.5</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{background:#0d0f14;color:white;font-family:'Segoe UI',Arial,sans-serif;padding:12px;}
h1{text-align:center;font-size:22px;margin-bottom:4px;background:linear-gradient(135deg,#2d89ef,#b388ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.sub{text-align:center;color:#6878a8;font-size:12px;margin-bottom:10px;}
.stats{display:flex;justify-content:center;gap:10px;margin:8px 0 12px;flex-wrap:wrap;}
.stat{background:#1e2233;padding:5px 12px;border-radius:8px;font-size:12px;color:#aab0c4;border:1px solid #2a3050;}
.stat b{color:#fff;}
.viewer-svg{width:100%;height:78vh;background:#1a1d24;border-radius:12px;border:1px solid #2a3050;overflow:auto;display:flex;align-items:center;justify-content:center;padding:20px;}
.viewer-svg svg{max-width:100%;height:auto;}
.actions{text-align:center;margin-top:12px;display:flex;justify-content:center;gap:8px;flex-wrap:wrap;}
.btn{padding:10px 18px;border:none;border-radius:10px;font-size:13px;font-weight:700;cursor:pointer;text-decoration:none;display:inline-block;}
.btn-blue{background:#1a6fd4;color:white;}
.btn-green{background:#1a9e4a;color:white;}
.btn-gray{background:#252a38;color:#aab0c4;}
.footer{text-align:center;color:#3a4060;font-size:11px;margin-top:10px;}
</style></head><body>
<h1>Synchrony BAS Graphic v23.5</h1>
<p class="sub">More horizontal cabinet projection - Ready for Tracer Synchrony / Niagara</p>
<div class="stats">
<div class="stat">VAVs: <b>{{ n_vavs }}</b></div>
<div class="stat">AHUs: <b>{{ n_ahus }}</b></div>
<div class="stat">Ducts: <b>{{ n_ducts }}</b></div>
<div class="stat">Diffusers: <b>{{ n_diffs }}</b></div>
<div class="stat">Walls: <b>{{ n_walls }}</b></div>
</div>
<div class="viewer-svg" id="svgViewer"></div>
<div class="actions">
<button onclick="downloadSVG()" class="btn btn-green">Download SVG</button>
<button onclick="downloadPNG()" class="btn btn-blue">Download PNG</button>
<a href="/editor" class="btn btn-gray">Edit Markings</a>
<a href="/" class="btn btn-gray">New Plan</a>
</div>
<div class="footer">Made by Paolo V. R.</div>

<script>
const data = {{ detection_json | safe }};

// === MORE HORIZONTAL PROJECTION (v23.5) ===
// Cabinet-like projection: minimal Y-axis skew, slight angle for depth
// X stays horizontal, Y moves slightly down, Z (height) goes up
const SKEW_ANGLE = Math.PI / 14;  // ~12.8 degrees (much less than iso 30 deg)
const COS_SK = Math.cos(SKEW_ANGLE);
const SIN_SK = Math.sin(SKEW_ANGLE);

function cabinetProject(x, y, z){
    // Cabinet projection: X stays full-scale horizontal,
    // Y is projected at angle (compressed), Z is vertical
    const sx = x + y * COS_SK * 0.5;
    const sy = y * SIN_SK - z;
    return [sx, sy];
}

function generateSVG(){
    const elements = data.elements || [];
    const extWall = elements.find(e => e.type === 'extwall' && e.points && e.points.length >= 3);
    let minX = 0, maxX = data.image_width, minY = 0, maxY = data.image_height;
    if(extWall){
        const xs = extWall.points.map(p => p.x);
        const ys = extWall.points.map(p => p.y);
        minX = Math.min(...xs); maxX = Math.max(...xs);
        minY = Math.min(...ys); maxY = Math.max(...ys);
    }
    const bcx = (minX + maxX) / 2;
    const bcy = (minY + maxY) / 2;
    const WALL_HEIGHT = 45;
    function toLocal(p){ return { x: p.x - bcx, y: p.y - bcy }; }
    function proj(x, y, z = 0){ return cabinetProject(x, y, z); }

    let svgMinX = 0, svgMaxX = 0, svgMinY = 0, svgMaxY = 0;
    const corners = [
        toLocal({ x: minX, y: minY }),
        toLocal({ x: maxX, y: minY }),
        toLocal({ x: maxX, y: maxY }),
        toLocal({ x: minX, y: maxY })
    ];
    for(const c of corners){
        for(const z of [0, WALL_HEIGHT + 15]){
            const [sx, sy] = proj(c.x, c.y, z);
            svgMinX = Math.min(svgMinX, sx); svgMaxX = Math.max(svgMaxX, sx);
            svgMinY = Math.min(svgMinY, sy); svgMaxY = Math.max(svgMaxY, sy);
        }
    }
    const padding = 60;
    const svgW = svgMaxX - svgMinX + padding * 2;
    const svgH = svgMaxY - svgMinY + padding * 2;
    const offsetX = -svgMinX + padding;
    const offsetY = -svgMinY + padding;

    function projSVG(x, y, z = 0){
        const [sx, sy] = proj(x, y, z);
        return [sx + offsetX, sy + offsetY];
    }

    let svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${svgW} ${svgH}" width="${svgW}" height="${svgH}">`;
    svg += `<rect width="${svgW}" height="${svgH}" fill="#0a0a0d"/>`;
    svg += `<defs>`;

    // Floor pattern - subtle grid for more horizontal cabinet projection
    const tileSize = 45;
    svg += `<pattern id="floorGrid" width="${tileSize}" height="${tileSize * SIN_SK * 1.5}" patternUnits="userSpaceOnUse">`;
    svg += `<rect width="${tileSize}" height="${tileSize * SIN_SK * 1.5}" fill="#dcdce0"/>`;
    svg += `<rect width="${tileSize}" height="${tileSize * SIN_SK * 1.5}" fill="none" stroke="#cfcfd3" stroke-width="0.5" opacity="0.7"/>`;
    svg += `</pattern>`;

    svg += `<linearGradient id="extSide" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#c8c8ce"/><stop offset="100%" stop-color="#929298"/></linearGradient>`;
    svg += `<linearGradient id="extTop" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#e8e8ec"/><stop offset="100%" stop-color="#b8b8bc"/></linearGradient>`;
    svg += `<linearGradient id="intSide" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#d4d4d8"/><stop offset="100%" stop-color="#a8a8ac"/></linearGradient>`;
    svg += `<linearGradient id="intTop" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#ebebef"/><stop offset="100%" stop-color="#c4c4c8"/></linearGradient>`;
    svg += `<linearGradient id="wallEnd" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#bababf"/><stop offset="100%" stop-color="#9b9ba0"/></linearGradient>`;
    svg += `<linearGradient id="ductTop" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#ffffff"/><stop offset="100%" stop-color="#d8d8dc"/></linearGradient>`;
    svg += `<linearGradient id="ductSide" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#c8c8cc"/><stop offset="100%" stop-color="#a0a0a4"/></linearGradient>`;
    svg += `<linearGradient id="vavTop" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#3b6df0"/><stop offset="100%" stop-color="#1e40af"/></linearGradient>`;
    svg += `<linearGradient id="vavFront" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#1e3a8a"/><stop offset="100%" stop-color="#152a6e"/></linearGradient>`;
    svg += `<linearGradient id="vavRight" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#1e40af"/><stop offset="100%" stop-color="#0c1f5c"/></linearGradient>`;
    svg += `<linearGradient id="ahuTop" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#34d365"/><stop offset="100%" stop-color="#16a34a"/></linearGradient>`;
    svg += `<linearGradient id="ahuFront" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#15803d"/><stop offset="100%" stop-color="#0a5828"/></linearGradient>`;
    svg += `<linearGradient id="ahuRight" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#16a34a"/><stop offset="100%" stop-color="#0c5a26"/></linearGradient>`;
    svg += `</defs>`;

    if(extWall){
        const pts = extWall.points.map(p => toLocal(p));
        let path = '';
        for(let i = 0; i < pts.length; i++){
            const [sx, sy] = projSVG(pts[i].x, pts[i].y, 0);
            path += (i === 0 ? 'M' : 'L') + sx + ',' + sy + ' ';
        }
        path += 'Z';
        svg += `<path d="${path}" fill="url(#floorGrid)" stroke="#a0a0a4" stroke-width="0.4"/>`;
    }

    function drawThickWall(p1, p2, height, thickness, sideG, topG, stroke){
        const dx = p2.x - p1.x, dy = p2.y - p1.y;
        const len = Math.sqrt(dx*dx + dy*dy);
        if(len < 1) return '';
        const nx = -dy / len * thickness / 2;
        const ny = dx / len * thickness / 2;
        const p1a = { x: p1.x + nx, y: p1.y + ny };
        const p1b = { x: p1.x - nx, y: p1.y - ny };
        const p2a = { x: p2.x + nx, y: p2.y + ny };
        const p2b = { x: p2.x - nx, y: p2.y - ny };
        const [b1ax, b1ay] = projSVG(p1a.x, p1a.y, 0);
        const [b1bx, b1by] = projSVG(p1b.x, p1b.y, 0);
        const [b2bx, b2by] = projSVG(p2b.x, p2b.y, 0);
        const [t1ax, t1ay] = projSVG(p1a.x, p1a.y, height);
        const [t2ax, t2ay] = projSVG(p2a.x, p2a.y, height);
        const [t1bx, t1by] = projSVG(p1b.x, p1b.y, height);
        const [t2bx, t2by] = projSVG(p2b.x, p2b.y, height);
        let w = '';
        w += `<path d="M ${b1bx},${b1by} L ${b2bx},${b2by} L ${t2bx},${t2by} L ${t1bx},${t1by} Z" fill="${sideG}" stroke="${stroke}" stroke-width="0.5"/>`;
        w += `<path d="M ${t1ax},${t1ay} L ${t2ax},${t2ay} L ${t2bx},${t2by} L ${t1bx},${t1by} Z" fill="${topG}" stroke="${stroke}" stroke-width="0.5"/>`;
        w += `<path d="M ${b1ax},${b1ay} L ${b1bx},${b1by} L ${t1bx},${t1by} L ${t1ax},${t1ay} Z" fill="url(#wallEnd)" stroke="${stroke}" stroke-width="0.5"/>`;
        return w;
    }

    if(extWall && extWall.points.length >= 2){
        const pts = extWall.points.map(p => toLocal(p));
        for(let i = 0; i < pts.length - 1; i++){
            svg += drawThickWall(pts[i], pts[i+1], WALL_HEIGHT, 14, 'url(#extSide)', 'url(#extTop)', '#5a5d63');
        }
        if(pts.length >= 3){
            svg += drawThickWall(pts[pts.length-1], pts[0], WALL_HEIGHT, 14, 'url(#extSide)', 'url(#extTop)', '#5a5d63');
        }
    }

    elements.forEach(el => {
        if(el.type === 'intwall' && el.points && el.points.length >= 2){
            const pts = el.points.map(p => toLocal(p));
            for(let i = 0; i < pts.length - 1; i++){
                svg += drawThickWall(pts[i], pts[i+1], WALL_HEIGHT * 0.92, 8, 'url(#intSide)', 'url(#intTop)', '#6a6d73');
            }
        }
    });

    const ductElements = elements.filter(e => e.type === 'duct' && e.points && e.points.length >= 2);
    ductElements.forEach(el => {
        const pts = el.points.map(p => toLocal(p));
        for(let i = 0; i < pts.length - 1; i++){
            const p1 = pts[i], p2 = pts[i+1];
            const dx = p2.x - p1.x, dy = p2.y - p1.y;
            const len = Math.sqrt(dx*dx + dy*dy);
            if(len < 1) continue;
            const ductW = 13, ductH = 9;
            const nx = -dy / len * ductW / 2;
            const ny = dx / len * ductW / 2;
            const p1a = { x: p1.x + nx, y: p1.y + ny };
            const p1b = { x: p1.x - nx, y: p1.y - ny };
            const p2a = { x: p2.x + nx, y: p2.y + ny };
            const p2b = { x: p2.x - nx, y: p2.y - ny };
            const zLevel = WALL_HEIGHT - 8;
            const [t1ax, t1ay] = projSVG(p1a.x, p1a.y, zLevel + ductH);
            const [t2ax, t2ay] = projSVG(p2a.x, p2a.y, zLevel + ductH);
            const [t1bx, t1by] = projSVG(p1b.x, p1b.y, zLevel + ductH);
            const [t2bx, t2by] = projSVG(p2b.x, p2b.y, zLevel + ductH);
            const [b1bx, b1by] = projSVG(p1b.x, p1b.y, zLevel);
            const [b2bx, b2by] = projSVG(p2b.x, p2b.y, zLevel);
            const [b1ax, b1ay] = projSVG(p1a.x, p1a.y, zLevel);
            const [b2ax, b2ay] = projSVG(p2a.x, p2a.y, zLevel);
            svg += `<path d="M ${t1ax},${t1ay} L ${t2ax},${t2ay} L ${t2bx},${t2by} L ${t1bx},${t1by} Z" fill="url(#ductTop)" stroke="#888" stroke-width="0.4"/>`;
            svg += `<path d="M ${b1bx},${b1by} L ${b2bx},${b2by} L ${t2bx},${t2by} L ${t1bx},${t1by} Z" fill="url(#ductSide)" stroke="#888" stroke-width="0.4"/>`;
            svg += `<path d="M ${b1ax},${b1ay} L ${b1bx},${b1by} L ${t1bx},${t1by} L ${t1ax},${t1ay} Z" fill="#bababf" stroke="#888" stroke-width="0.4"/>`;
            svg += `<path d="M ${b2ax},${b2ay} L ${b2bx},${b2by} L ${t2bx},${t2by} L ${t2ax},${t2ay} Z" fill="#bababf" stroke="#888" stroke-width="0.4"/>`;
        }
    });

    elements.forEach(el => {
        if(el.type === 'branch' && el.points && el.points.length === 2){
            const p1 = toLocal(el.points[0]);
            const p2 = toLocal(el.points[1]);
            const z = WALL_HEIGHT - 4;
            const [s1x, s1y] = projSVG(p1.x, p1.y, z);
            const [s2x, s2y] = projSVG(p2.x, p2.y, z);
            svg += `<line x1="${s1x}" y1="${s1y}" x2="${s2x}" y2="${s2y}" stroke="#444" stroke-width="2"/>`;
        }
    });

    elements.forEach(el => {
        if(el.type === 'diffuser'){
            const p = toLocal({ x: el.x, y: el.y });
            const size = 6;
            const z = WALL_HEIGHT - 1;
            const c = [
                { x: p.x - size, y: p.y - size }, { x: p.x + size, y: p.y - size },
                { x: p.x + size, y: p.y + size }, { x: p.x - size, y: p.y + size }
            ];
            const proj4 = c.map(cp => projSVG(cp.x, cp.y, z));
            svg += `<path d="M ${proj4[0][0]},${proj4[0][1]} L ${proj4[1][0]},${proj4[1][1]} L ${proj4[2][0]},${proj4[2][1]} L ${proj4[3][0]},${proj4[3][1]} Z" fill="#ffffff" stroke="#666" stroke-width="0.4"/>`;
        }
    });

    function drawCube(cx, cy, hs, h, bz, topG, frontG, rightG, stroke){
        const c = [
            { x: cx - hs, y: cy - hs }, { x: cx + hs, y: cy - hs },
            { x: cx + hs, y: cy + hs }, { x: cx - hs, y: cy + hs }
        ];
        const b = c.map(p => projSVG(p.x, p.y, bz));
        const t = c.map(p => projSVG(p.x, p.y, bz + h));
        let cube = '';
        const sh = c.map(p => projSVG(p.x + 3, p.y + 3, 0.4));
        cube += `<path d="M ${sh[0][0]},${sh[0][1]} L ${sh[1][0]},${sh[1][1]} L ${sh[2][0]},${sh[2][1]} L ${sh[3][0]},${sh[3][1]} Z" fill="#000" opacity="0.2"/>`;
        cube += `<path d="M ${t[0][0]},${t[0][1]} L ${t[1][0]},${t[1][1]} L ${t[2][0]},${t[2][1]} L ${t[3][0]},${t[3][1]} Z" fill="${topG}" stroke="${stroke}" stroke-width="0.5"/>`;
        cube += `<path d="M ${b[3][0]},${b[3][1]} L ${b[2][0]},${b[2][1]} L ${t[2][0]},${t[2][1]} L ${t[3][0]},${t[3][1]} Z" fill="${frontG}" stroke="${stroke}" stroke-width="0.5"/>`;
        cube += `<path d="M ${b[1][0]},${b[1][1]} L ${b[2][0]},${b[2][1]} L ${t[2][0]},${t[2][1]} L ${t[1][0]},${t[1][1]} Z" fill="${rightG}" stroke="${stroke}" stroke-width="0.5"/>`;
        return cube;
    }

    elements.forEach(el => {
        if(el.type === 'vav'){
            const p = toLocal({ x: el.x, y: el.y });
            svg += drawCube(p.x, p.y, 8, 18, WALL_HEIGHT - 18, 'url(#vavTop)', 'url(#vavFront)', 'url(#vavRight)', '#0c1c5c');
        }
    });

    elements.forEach(el => {
        if(el.type === 'ahu'){
            const p = toLocal({ x: el.x, y: el.y });
            svg += drawCube(p.x, p.y, 18, 26, WALL_HEIGHT - 26, 'url(#ahuTop)', 'url(#ahuFront)', 'url(#ahuRight)', '#0a4220');
        }
    });

    svg += '</svg>';
    return svg;
}

const svgContent = generateSVG();
document.getElementById('svgViewer').innerHTML = svgContent;

function downloadSVG(){
    const blob = new Blob([svgContent], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.download = 'bas_graphic.svg';
    link.href = url;
    link.click();
    URL.revokeObjectURL(url);
}

function downloadPNG(){
    const svgBlob = new Blob([svgContent], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(svgBlob);
    const img = new Image();
    img.onload = function(){
        const canvas = document.createElement('canvas');
        canvas.width = img.width * 2;
        canvas.height = img.height * 2;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#0a0a0d';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(function(blob){
            const purl = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.download = 'bas_graphic.png';
            link.href = purl;
            link.click();
            URL.revokeObjectURL(purl);
        }, 'image/png');
        URL.revokeObjectURL(url);
    };
    img.src = url;
}
</script>
</body></html>'''


# ============================================================
# ROUTES
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        password = request.form.get("password", "")
        if password == APP_PASSWORD:
            response = make_response(redirect("/"))
            response.set_cookie("bas_auth", APP_PASSWORD, max_age=60*60*24*7, httponly=True, secure=True, samesite="Lax")
            return response
        return render_template_string(LOGIN_PAGE, error=True)
    return render_template_string(LOGIN_PAGE, error=False)


@app.route("/logout")
def logout():
    response = make_response(redirect("/login"))
    response.delete_cookie("bas_auth")
    return response


@app.before_request
def require_login():
    if request.endpoint in ["login", "static"]:
        return
    if request.cookies.get("bas_auth") == APP_PASSWORD:
        return
    return redirect("/login")


@app.route("/")
def home():
    return HOME_PAGE


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    filename = file.filename.lower()
    mode = request.form.get("mode", "manual")

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
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    max_dim = 1400
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    cv2.imwrite(UPLOAD_IMAGE_PATH, img)

    initial_elements = []
    detected_message = ""

    if mode == "arch":
        try:
            result = detect_architecture(UPLOAD_IMAGE_PATH)
            if result:
                initial_elements = result["elements"]
                stats = result["stats"]
                n_walls = sum(1 for e in initial_elements if e.get("type") in ("extwall", "intwall"))
                crop = stats.get("crop", (0, 0, 0, 0))
                detected_message = f"Smart isolation: cropped to {crop[2]}x{crop[3]}. Detected {n_walls} walls from {stats['lines_raw']} raw lines. Review and adjust!"
            else:
                detected_message = "Detection ran but found nothing. Try Manual mode."
        except Exception as e:
            detected_message = f"Detection error: {str(e)}. Continuing in manual mode."
            initial_elements = []
    elif mode == "color":
        try:
            result = auto_detect_colors(UPLOAD_IMAGE_PATH)
            if result:
                initial_elements = result["elements"]
                detected_message = f"Auto-detected by colors: {len(initial_elements)} HVAC elements."
        except Exception as e:
            detected_message = f"Color detection error: {str(e)}."
            initial_elements = []

    return render_template_string(
        EDITOR_PAGE,
        image_b64=image_to_base64(UPLOAD_IMAGE_PATH),
        initial_elements=json.dumps(initial_elements),
        detected_message=detected_message
    )


@app.route("/editor")
def editor():
    if not os.path.exists(UPLOAD_IMAGE_PATH):
        return "<h2 style='color:white;background:#0d0f14;padding:30px;'>No plan uploaded. <a href='/' style='color:#2d89ef'>Upload one</a></h2>"
    return render_template_string(
        EDITOR_PAGE,
        image_b64=image_to_base64(UPLOAD_IMAGE_PATH),
        initial_elements=json.dumps([]),
        detected_message=""
    )


@app.route("/process", methods=["POST"])
def process():
    try:
        data = request.get_json()
        detection = {
            "image_width": data["imageWidth"],
            "image_height": data["imageHeight"],
            "elements": data["elements"]
        }
        with open(os.path.join(OUTPUT_FOLDER, "detection.json"), "w") as f:
            json.dump(detection, f)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/result")
def result():
    try:
        with open(os.path.join(OUTPUT_FOLDER, "detection.json"), "r") as f:
            detection = json.load(f)
    except FileNotFoundError:
        return "<h2 style='color:white;background:#0d0f14;padding:30px;'>No data. <a href='/' style='color:#2d89ef'>Start over</a></h2>"

    elements = detection.get("elements", [])
    n_vavs = sum(1 for e in elements if e.get("type") == "vav")
    n_ahus = sum(1 for e in elements if e.get("type") == "ahu")
    n_ducts = sum(1 for e in elements if e.get("type") == "duct")
    n_diffs = sum(1 for e in elements if e.get("type") == "diffuser")
    n_walls = sum(1 for e in elements if e.get("type") in ("extwall", "intwall"))

    return render_template_string(
        RESULT_PAGE,
        detection_json=json.dumps(detection),
        n_vavs=n_vavs, n_ahus=n_ahus, n_ducts=n_ducts,
        n_diffs=n_diffs, n_walls=n_walls
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
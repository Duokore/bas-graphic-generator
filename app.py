from flask import Flask, request, send_file, render_template_string
import os
import base64
import json
import math
import cv2
import numpy as np
import fitz
import anthropic

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
UPLOAD_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, "mechanical_upload.png")
UPLOAD_PDF_PATH = os.path.join(UPLOAD_FOLDER, "mechanical_upload.pdf")
ISO_OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, "bas_graphic_3d.png")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def pdf_to_png(pdf_path, out_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
    pix.save(out_path)
    doc.close()


def analyze_plan_with_claude(image_path):
    img_b64 = image_to_base64(image_path)

    prompt_text = (
        "You are an expert HVAC engineer analyzing a mechanical floor plan. "
        "Look at this plan carefully and extract data in JSON format. "
        "Return ONLY valid JSON with these fields: "
        "building_outline (polygon points 0-1000), "
        "rooms (with name and bbox [x,y,w,h] 0-1000), "
        "vavs (with tag and pos [x,y] 0-1000), "
        "ahus (with tag and pos), "
        "ducts (with path of points), "
        "diffusers (with pos). "
        "Example: {\"building_outline\":[[100,100],[900,100],[900,900],[100,900]],"
        "\"rooms\":[{\"name\":\"Office 101\",\"bbox\":[120,120,200,150]}],"
        "\"vavs\":[{\"tag\":\"VAV-1\",\"pos\":[200,180]}],"
        "\"ahus\":[{\"tag\":\"AHU-1\",\"pos\":[500,500]}],"
        "\"ducts\":[{\"path\":[[500,500],[400,500],[200,200]]}],"
        "\"diffusers\":[{\"pos\":[200,200]}]}"
    )

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                {"type": "text", "text": prompt_text}
            ]
        }]
    )

    response_text = message.content[0].text.strip()
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
        response_text = response_text.strip()

    return json.loads(response_text)


def build_clean_iso(plan_data, canvas_w=1600, canvas_h=1100):
    rad = math.radians(26.565)

    def proj(x, y, z=0):
        return (x - y) * math.cos(rad), (x + y) * math.sin(rad) - z

    out_w, out_h = canvas_w, canvas_h
    out = np.full((out_h, out_w, 4), (15, 18, 25, 255), dtype=np.uint8)
    ox, oy = out_w * 0.5, out_h * 0.3
    SCALE = 0.7

    def to_screen(p, z=0):
        x = p[0] * SCALE
        y = p[1] * SCALE
        sx, sy = proj(x, y, z)
        return [int(sx + ox), int(sy + oy)]

    WALL_HEIGHT = 50
    FLOOR_FILL = (50, 58, 75, 255)
    FLOOR_LINE = (75, 85, 105, 255)
    WALL_TOP = (170, 175, 188, 255)
    WALL_SIDE = (105, 110, 125, 255)
    DUCT_COLOR = (210, 215, 225, 255)
    GRILL_COLOR = (80, 220, 180, 255)

    if "building_outline" in plan_data and plan_data["building_outline"]:
        floor_pts = np.array([to_screen(p, 0) for p in plan_data["building_outline"]], np.int32)
        cv2.fillPoly(out, [floor_pts], FLOOR_FILL)
        cv2.polylines(out, [floor_pts], True, FLOOR_LINE, 2)

    for room in plan_data.get("rooms", []):
        x, y, w, h = room["bbox"]
        corners_floor = [to_screen([x, y], 0), to_screen([x + w, y], 0),
                         to_screen([x + w, y + h], 0), to_screen([x, y + h], 0)]
        cv2.polylines(out, [np.array(corners_floor, np.int32)], True, FLOOR_LINE, 1)

    for room in plan_data.get("rooms", []):
        x, y, w, h = room["bbox"]
        wall_segs = [
            ([x, y], [x + w, y]),
            ([x, y + h], [x + w, y + h]),
            ([x, y], [x, y + h]),
            ([x + w, y], [x + w, y + h])
        ]
        for p1, p2 in wall_segs:
            b1 = to_screen(p1, 0)
            b2 = to_screen(p2, 0)
            t1 = to_screen(p1, WALL_HEIGHT)
            t2 = to_screen(p2, WALL_HEIGHT)
            wall_quad = np.array([b1, b2, t2, t1], np.int32)
            cv2.fillPoly(out, [wall_quad], WALL_SIDE)
            cv2.polylines(out, [wall_quad], True, (60, 65, 80, 255), 1)
            cv2.line(out, tuple(t1), tuple(t2), WALL_TOP, 2)

    for room in plan_data.get("rooms", []):
        if not room.get("name"):
            continue
        x, y, w, h = room["bbox"]
        cx, cy = x + w / 2, y + h / 2
        label_pos = to_screen([cx, cy], WALL_HEIGHT + 10)
        cv2.putText(out, room["name"], (label_pos[0] - 30, label_pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 210, 230, 255), 1, cv2.LINE_AA)

    for duct in plan_data.get("ducts", []):
        path = duct.get("path", [])
        if len(path) < 2:
            continue
        screen_path = [to_screen(p, WALL_HEIGHT - 5) for p in path]
        for i in range(len(screen_path) - 1):
            cv2.line(out, tuple(screen_path[i]), tuple(screen_path[i + 1]), DUCT_COLOR, 4)

    for diff in plan_data.get("diffusers", []):
        p = diff["pos"]
        center = to_screen(p, WALL_HEIGHT - 3)
        cv2.rectangle(out, (center[0] - 4, center[1] - 4),
                      (center[0] + 4, center[1] + 4), GRILL_COLOR, -1)

    BS = 18
    BH = 30
    VAV_FRONT = (240, 130, 60, 255)
    VAV_TOP = (255, 165, 95, 255)
    VAV_SIDE = (180, 85, 30, 255)

    for vav in plan_data.get("vavs", []):
        cx, cy = vav["pos"]
        front = np.array([to_screen([cx - BS, cy + BS], 0), to_screen([cx + BS, cy + BS], 0),
                          to_screen([cx + BS, cy + BS], BH), to_screen([cx - BS, cy + BS], BH)], np.int32)
        top = np.array([to_screen([cx - BS, cy - BS], BH), to_screen([cx + BS, cy - BS], BH),
                        to_screen([cx + BS, cy + BS], BH), to_screen([cx - BS, cy + BS], BH)], np.int32)
        side = np.array([to_screen([cx + BS, cy - BS], 0), to_screen([cx + BS, cy + BS], 0),
                         to_screen([cx + BS, cy + BS], BH), to_screen([cx + BS, cy - BS], BH)], np.int32)
        cv2.fillPoly(out, [front], VAV_FRONT)
        cv2.fillPoly(out, [top], VAV_TOP)
        cv2.fillPoly(out, [side], VAV_SIDE)
        if vav.get("tag"):
            label_pos = to_screen([cx, cy], BH + 8)
            cv2.putText(out, vav["tag"], (label_pos[0] - 18, label_pos[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255, 255), 1, cv2.LINE_AA)

    AHU_S = 35
    AHU_H = 45
    AHU_FRONT = (130, 140, 155, 255)
    AHU_TOP = (160, 170, 185, 255)
    AHU_SIDE = (95, 105, 120, 255)

    for ahu in plan_data.get("ahus", []):
        cx, cy = ahu["pos"]
        front = np.array([to_screen([cx - AHU_S, cy + AHU_S], 0), to_screen([cx + AHU_S, cy + AHU_S], 0),
                          to_screen([cx + AHU_S, cy + AHU_S], AHU_H), to_screen([cx - AHU_S, cy + AHU_S], AHU_H)], np.int32)
        top = np.array([to_screen([cx - AHU_S, cy - AHU_S], AHU_H), to_screen([cx + AHU_S, cy - AHU_S], AHU_H),
                        to_screen([cx + AHU_S, cy + AHU_S], AHU_H), to_screen([cx - AHU_S, cy + AHU_S], AHU_H)], np.int32)
        side = np.array([to_screen([cx + AHU_S, cy - AHU_S], 0), to_screen([cx + AHU_S, cy + AHU_S], 0),
                         to_screen([cx + AHU_S, cy + AHU_S], AHU_H), to_screen([cx + AHU_S, cy - AHU_S], AHU_H)], np.int32)
        cv2.fillPoly(out, [front], AHU_FRONT)
        cv2.fillPoly(out, [top], AHU_TOP)
        cv2.fillPoly(out, [side], AHU_SIDE)
        if ahu.get("tag"):
            label_pos = to_screen([cx, cy], AHU_H + 8)
            cv2.putText(out, ahu["tag"], (label_pos[0] - 25, label_pos[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255, 255), 1, cv2.LINE_AA)

    return out


HOME_PAGE = '''<!DOCTYPE html>
<html>
<head>
<title>BAS Generator v9</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d0f14; color: white; font-family: Arial, sans-serif; min-height: 100vh; display: flex; align-items: center; justify-content: center; }
.card { background: #181b24; border: 1px solid #2a2f3e; border-radius: 24px; padding: 50px; text-align: center; max-width: 680px; width: 90%; }
.logo { font-size: 48px; margin-bottom: 16px; }
h1 { font-size: 28px; margin-bottom: 8px; }
.sub { color: #7a8099; font-size: 14px; margin-bottom: 36px; }
.zone { border: 2px dashed #2d3348; border-radius: 16px; padding: 32px; margin-bottom: 24px; background: #13151d; }
input[type=file] { background: transparent; color: #aab0c4; border: none; font-size: 14px; width: 100%; }
.lbl { display: block; font-size: 13px; color: #5a6280; margin-top: 10px; }
.btn { background: linear-gradient(135deg, #1a6fd4, #2d89ef); color: white; border: none; border-radius: 14px; padding: 16px 40px; font-size: 18px; font-weight: 700; cursor: pointer; width: 100%; }
.badge { display: inline-block; background: #1e2233; border: 1px solid #2a3050; border-radius: 8px; padding: 4px 12px; font-size: 12px; color: #6878a8; margin: 4px; }
.ai { background: linear-gradient(135deg, #7c4dff, #b388ff); color: white; }
.note { font-size: 12px; color: #5a6280; margin-top: 14px; font-style: italic; }
.footer { color: #3a4060; font-size: 12px; margin-top: 28px; }
</style>
</head>
<body>
<div class="card">
<div class="logo">&#9881;&#65039; &#127970; &#129302;</div>
<h1>BAS Graphic Generator v9</h1>
<p class="sub">AI-Powered HVAC Graphics - Powered by Claude</p>
<div style="margin-bottom:20px;">
<span class="badge ai">Claude AI</span>
<span class="badge">Auto Detection</span>
<span class="badge">VAV Tagging</span>
<span class="badge">Duct Routing</span>
</div>
<form action="/generate" method="post" enctype="multipart/form-data">
<div class="zone">
<input type="file" name="file" accept="image/png,image/jpeg,application/pdf" required>
<span class="lbl">Upload mechanical plan - PNG, JPG or PDF</span>
</div>
<button class="btn" type="submit">Generate AI Graphic</button>
<p class="note">AI analysis takes 20-40 seconds per plan</p>
</form>
<div class="footer">Made by Paolo V. and Emmanuel R.</div>
</div>
</body>
</html>'''


RESULT_PAGE = '''<!DOCTYPE html>
<html>
<head>
<title>AI Result</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d0f14; color: white; font-family: Arial, sans-serif; padding: 30px; }
h1 { text-align: center; font-size: 26px; margin-bottom: 6px; }
.sub { text-align: center; color: #6878a8; font-size: 13px; margin-bottom: 30px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.card { background: #181b24; border: 1px solid #252a38; border-radius: 18px; padding: 18px; }
.card h2 { text-align: center; font-size: 14px; margin-bottom: 12px; color: #9aa0b8; text-transform: uppercase; }
.viewer { width: 100%; height: 560px; overflow: auto; background: #111318; border-radius: 10px; border: 1px solid #1e2230; }
.viewer img { width: 100%; display: block; }
.actions { text-align: center; margin-top: 28px; display: flex; justify-content: center; gap: 12px; }
.btn { padding: 13px 28px; border: none; border-radius: 12px; font-size: 15px; font-weight: 700; cursor: pointer; text-decoration: none; display: inline-block; }
.btn-blue { background: #1a6fd4; color: white; }
.btn-gray { background: #252a38; color: #aab0c4; }
.stats { display: flex; justify-content: center; gap: 30px; margin: 14px 0; flex-wrap: wrap; }
.stat { background: #1e2233; padding: 8px 18px; border-radius: 8px; font-size: 13px; color: #aab0c4; border: 1px solid #2a3050; }
.stat b { color: #fff; }
.footer { text-align: center; color: #3a4060; font-size: 12px; margin-top: 24px; }
</style>
</head>
<body>
<h1>AI-Generated BAS Graphic</h1>
<p class="sub">Claude analyzed your plan and rendered the 3D view</p>
<div class="stats">
<div class="stat">Rooms: <b>{{ n_rooms }}</b></div>
<div class="stat">VAVs: <b>{{ n_vavs }}</b></div>
<div class="stat">AHUs: <b>{{ n_ahus }}</b></div>
<div class="stat">Ducts: <b>{{ n_ducts }}</b></div>
</div>
<div class="grid">
<div class="card"><h2>Original Plan</h2>
<div class="viewer"><img src="data:image/png;base64,{{ original_b64 }}"></div></div>
<div class="card"><h2>AI Isometric 3D</h2>
<div class="viewer"><img src="data:image/png;base64,{{ iso_b64 }}"></div></div>
</div>
<div class="actions">
<a href="/download_iso" class="btn btn-blue">Download 3D PNG</a>
<a href="/" class="btn btn-gray">Generate Another</a>
</div>
<div class="footer">Made by Paolo V. and Emmanuel R.</div>
</body>
</html>'''


@app.route("/")
def home():
    return HOME_PAGE


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
        return "Error loading image", 400

    h, w = img.shape[:2]
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(UPLOAD_IMAGE_PATH, img)

    try:
        plan_data = analyze_plan_with_claude(UPLOAD_IMAGE_PATH)
    except Exception as e:
        error_msg = str(e)
        return "<h2 style='color:white;background:#0d0f14;padding:30px;'>AI analysis failed: " + error_msg + "</h2>", 500

    iso = build_clean_iso(plan_data)
    cv2.imwrite(ISO_OUTPUT_PATH, iso)

    return render_template_string(
        RESULT_PAGE,
        original_b64=image_to_base64(UPLOAD_IMAGE_PATH),
        iso_b64=image_to_base64(ISO_OUTPUT_PATH),
        n_rooms=len(plan_data.get("rooms", [])),
        n_vavs=len(plan_data.get("vavs", [])),
        n_ahus=len(plan_data.get("ahus", [])),
        n_ducts=len(plan_data.get("ducts", []))
    )


@app.route("/download_iso")
def download_iso():
    return send_file(ISO_OUTPUT_PATH, mimetype="image/png",
                     as_attachment=True, download_name="bas_graphic_3d.png")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

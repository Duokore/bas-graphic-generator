from flask import Flask, request, send_file
import os, base64, json, math
import cv2
import numpy as np
import fitz
import anthropic

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
UPLOAD_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, "mechanical_upload.png")
UPLOAD_PDF_PATH   = os.path.join(UPLOAD_FOLDER, "mechanical_upload.pdf")
ISO_OUTPUT_PATH   = os.path.join(OUTPUT_FOLDER, "bas_graphic_3d.png")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def pdf_to_png(pdf_path, out_path):
    doc  = fitz.open(pdf_path)
    page = doc[0]
    pix  = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
    pix.save(out_path)
    doc.close()


# ─────────────────────────────────────────────
# CLAUDE AI — Analyze the mechanical plan
# ─────────────────────────────────────────────
def analyze_plan_with_claude(image_path):
    img_b64 = image_to_base64(image_path)

    prompt = """You are an expert HVAC engineer analyzing a mechanical floor plan.
Look at this plan carefully and extract the following in JSON format:

1. The overall building outline as a polygon (list of [x,y] points in image coordinates, normalized 0-1000)
2. Each room with: name (if visible), and bounding box [x, y, width, height] (normalized 0-1000)
3. Each VAV box with: tag (like VAV-101 if visible, or VAV-1, VAV-2 if not numbered), and position [x, y] (normalized 0-1000)
4. Each AHU (Air Handling Unit) with: tag and position [x, y]
5. Main duct routes as polylines (list of [x,y] points)
6. Diffusers/grills as small points

Return ONLY valid JSON, no other text. Example format:
{
  "building_outline": [[100,100],[900,100],[900,900],[100,900]],
  "rooms": [
    {"name": "Office 101", "bbox": [120,120,200,150]},
    {"name": "Conference", "bbox": [340,120,180,150]}
  ],
  "vavs": [
    {"tag": "VAV-1", "pos": [200,180]},
    {"tag": "VAV-2", "pos": [420,180]}
  ],
  "ahus": [
    {"tag": "AHU-1", "pos": [500,500]}
  ],
  "ducts": [
    {"path": [[500,500],[400,500],[400,200],[200,200]]}
  ],
  "diffusers": [
    {"pos": [200,200]},
    {"pos": [420,200]}
  ]
}

If you can't see something clearly, make your best estimate. Always return valid JSON."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                {"type": "text", "text": prompt}
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


# ─────────────────────────────────────────────
# Build clean isometric 3D from AI data
# ─────────────────────────────────────────────
def build_clean_iso(plan_data, canvas_w=1600, canvas_h=1100):
    rad = math.radians(26.565)

    def proj(x, y, z=0):
        return (x - y) * math.cos(rad), (x + y) * math.sin(rad) - z

    out_w, out_h = canvas_w, canvas_h
    out = np.full((out_h, out_w, 4), (15, 18, 25, 255), dtype=np.uint8)
    ox, oy = out_w * 0.5, out_h * 0.3

    # Scale normalized coords (0-1000) to drawing space
    SCALE = 0.7

    def to_screen(p, z=0):
        x = p[0] * SCALE
        y = p[1] * SCALE
        sx, sy = proj(x, y, z)
        return [int(sx + ox), int(sy + oy)]

    WALL_HEIGHT = 50
    FLOOR_FILL  = (50, 58, 75, 255)
    FLOOR_LINE  = (75, 85, 105, 255)
    WALL_TOP    = (170, 175, 188, 255)
    WALL_SIDE   = (105, 110, 125, 255)
    DUCT_COLOR  = (210, 215, 225, 255)
    GRILL_COLOR = ( 80, 220, 180, 255)

    # Draw floor (building outline filled)
    if "building_outline" in plan_data and plan_data["building_outline"]:
        floor_pts = np.array([to_screen(p, 0) for p in plan_data["building_outline"]], np.int32)
        cv2.fillPoly(out, [floor_pts], FLOOR_FILL)
        cv2.polylines(out, [floor_pts], True, FLOOR_LINE, 2)

    # Draw rooms (subtle interior dividers)
    for room in plan_data.get("rooms", []):
        x, y, w, h = room["bbox"]
        corners_floor = [to_screen([x,y],0), to_screen([x+w,y],0),
                          to_screen([x+w,y+h],0), to_screen([x,y+h],0)]
        cv2.polylines(out, [np.array(corners_floor, np.int32)], True, FLOOR_LINE, 1)

    # Draw walls — extruded (room boundaries become walls)
    for room in plan_data.get("rooms", []):
        x, y, w, h = room["bbox"]
        # 4 wall segments: top, bottom, left, right
        wall_segs = [
            ([x, y], [x+w, y]),
            ([x, y+h], [x+w, y+h]),
            ([x, y], [x, y+h]),
            ([x+w, y], [x+w, y+h])
        ]
        for p1, p2 in wall_segs:
            b1 = to_screen(p1, 0); b2 = to_screen(p2, 0)
            t1 = to_screen(p1, WALL_HEIGHT); t2 = to_screen(p2, WALL_HEIGHT)
            wall_quad = np.array([b1, b2, t2, t1], np.int32)
            cv2.fillPoly(out, [wall_quad], WALL_SIDE)
            cv2.polylines(out, [wall_quad], True, (60, 65, 80, 255), 1)
            # Top edge highlight
            cv2.line(out, tuple(t1), tuple(t2), WALL_TOP, 2)

    # Draw room labels on top
    for room in plan_data.get("rooms", []):
        if not room.get("name"): continue
        x, y, w, h = room["bbox"]
        cx, cy = x + w/2, y + h/2
        label_pos = to_screen([cx, cy], WALL_HEIGHT + 10)
        cv2.putText(out, room["name"], (label_pos[0]-30, label_pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 210, 230, 255), 1, cv2.LINE_AA)

    # Draw ducts — as thick lines floating above floor
    for duct in plan_data.get("ducts", []):
        path = duct.get("path", [])
        if len(path) < 2: continue
        screen_path = [to_screen(p, WALL_HEIGHT - 5) for p in path]
        for i in range(len(screen_path) - 1):
            cv2.line(out, tuple(screen_path[i]), tuple(screen_path[i+1]), DUCT_COLOR, 4)

    # Draw diffusers as small green squares on ceiling
    for diff in plan_data.get("diffusers", []):
        p = diff["pos"]
        center = to_screen(p, WALL_HEIGHT - 3)
        cv2.rectangle(out, (center[0]-4, center[1]-4),
                      (center[0]+4, center[1]+4), GRILL_COLOR, -1)

    # Draw VAV boxes as 3D blue cubes
    BS = 18  # box half-size in plan coords
    BH = 30  # height
    VAV_FRONT = (240, 130, 60, 255)
    VAV_TOP   = (255, 165, 95, 255)
    VAV_SIDE  = (180,  85, 30, 255)

    for vav in plan_data.get("vavs", []):
        cx, cy = vav["pos"]
        front = np.array([to_screen([cx-BS, cy+BS], 0), to_screen([cx+BS, cy+BS], 0),
                          to_screen([cx+BS, cy+BS], BH), to_screen([cx-BS, cy+BS], BH)], np.int32)
        top   = np.array([to_screen([cx-BS, cy-BS], BH), to_screen([cx+BS, cy-BS], BH),
                          to_screen([cx+BS, cy+BS], BH), to_screen([cx-BS, cy+BS], BH)], np.int32)
        side  = np.array([to_screen([cx+BS, cy-BS], 0), to_screen([cx+BS, cy+BS], 0),
                          to_screen([cx+BS, cy+BS], BH), to_screen([cx+BS, cy-BS], BH)], np.int32)
        cv2.fillPoly(out, [front], VAV_FRONT)
        cv2.fillPoly(out, [top],   VAV_TOP)
        cv2.fillPoly(out, [side],  VAV_SIDE)
        cv2.polylines(out, [front], True, (100,40,10,255), 1)
        cv2.polylines(out, [top],   True, (100,40,10,255), 1)
        cv2.polylines(out, [side],  True, (100,40,10,255), 1)

        # VAV label
        if vav.get("tag"):
            label_pos = to_screen([cx, cy], BH + 8)
            cv2.putText(out, vav["tag"],
                        (label_pos[0]-18, label_pos[1]-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255,255), 1, cv2.LINE_AA)

    # Draw AHUs as larger gray boxes
    AHU_S = 35
    AHU_H = 45
    AHU_FRONT = (130, 140, 155, 255)
    AHU_TOP   = (160, 170, 185, 255)
    AHU_SIDE  = ( 95, 105, 120, 255)

    for ahu in plan_data.get("ahus", []):
        cx, cy = ahu["pos"]
        front = np.array([to_screen([cx-AHU_S, cy+AHU_S], 0), to_screen([cx+AHU_S, cy+AHU_S], 0),
                          to_screen([cx+AHU_S, cy+AHU_S], AHU_H), to_screen([cx-AHU_S, cy+AHU_S], AHU_H)], np.int32)
        top   = np.array([to_screen([cx-AHU_S, cy-AHU_S], AHU_H), to_screen([cx+AHU_S, cy-AHU_S], AHU_H),
                          to_screen([cx+AHU_S, cy+AHU_S], AHU_H), to_screen([cx-AHU_S, cy+AHU_S], AHU_H)], np.int32)
        side  = np.array([to_screen([cx+AHU_S, cy-AHU_S], 0), to_screen([cx+AHU_S, cy+AHU_S], 0),
                          to_screen([cx+AHU_S, cy+AHU_S], AHU_H), to_screen([cx+AHU_S, cy-AHU_S], AHU_H)], np.int32)
        cv2.fillPoly(out, [front], AHU_FRONT)
        cv2.fillPoly(out, [top],   AHU_TOP)
        cv2.fillPoly(out, [side],  AHU_SIDE)
        if ahu.get("tag"):
            label_pos = to_screen([cx, cy], AHU_H + 8)
            cv2.putText(out, ahu["tag"], (label_pos[0]-25, label_pos[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255,255), 1, cv2.LINE_AA)

    return out


HOME_HTML = """





  
⚙️ 🏢 🤖

  
BAS Graphic Generator v9

  

AI-Powered HVAC Graphics · Powered by Claude


  

    🧠 Claude AI
    Auto Room Detection
    VAV Tagging
    Duct Routing
  

  

    

      
      
Upload mechanical plan — PNG, JPG or PDF

    

    ⚡ Generate AI Graphic
    

⏳ AI analysis takes 20-40 seconds per plan


  

  

Made by Paolo V. & Emmanuel R.





"""

RESULT_STYLE = """

"""


@app.route("/")
def home():
    return HOME_HTML


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

    # Call Claude AI to analyze the plan
    try:
        plan_data = analyze_plan_with_claude(UPLOAD_IMAGE_PATH)
    except Exception as e:
        return f"
AI analysis failed: {str(e)}
", 500

    # Build clean 3D from AI data
    iso = build_clean_iso(plan_data)
    cv2.imwrite(ISO_OUTPUT_PATH, iso)

    o_b64 = image_to_base64(UPLOAD_IMAGE_PATH)
    i_b64 = image_to_base64(ISO_OUTPUT_PATH)

    n_rooms = len(plan_data.get("rooms", []))
    n_vavs  = len(plan_data.get("vavs", []))
    n_ahus  = len(plan_data.get("ahus", []))
    n_ducts = len(plan_data.get("ducts", []))

    return f"""
    {RESULT_STYLE}
    
      
🤖 AI-Generated BAS Graphic

      

Claude analyzed your plan and rendered the 3D view


      
Rooms: {n_rooms}
VAVs: {n_vavs}
AHUs: {n_ahus}
Duct routes: {n_ducts}

      
📄 ORIGINAL PLAN

          
🏢 AI ISOMETRIC 3D

          

      
⬇️ Download 3D PNG
🔄 Generate Another

      
Made by Paolo V. & Emmanuel R.

    
    """


@app.route("/download_iso")
def download_iso():
    return send_file(ISO_OUTPUT_PATH, mimetype="image/png",
                     as_attachment=True, download_name="bas_graphic_3d.png")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

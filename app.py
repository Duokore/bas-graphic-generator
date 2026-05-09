from flask import Flask, request, render_template_string
import os
import base64
import json
import math
import cv2
import numpy as np
import fitz

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
UPLOAD_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, "mechanical_upload.png")
UPLOAD_PDF_PATH = os.path.join(UPLOAD_FOLDER, "mechanical_upload.pdf")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Detection parameters
MIN_VAV_AREA = 40
MAX_VAV_AREA = 1200
MIN_AHU_AREA = 200
MAX_DISTANCE_TO_DUCT = 120
TARGET_VAVS = 9


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def pdf_to_png(pdf_path, out_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
    pix.save(out_path)
    doc.close()


def clean_mask(mask, iterations=2):
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def get_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def contour_center(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return (int(x + w / 2), int(y + h / 2))


def distance_to_nearest_duct(center, ducts):
    cx, cy = center
    best = 999999
    for d in ducts:
        x, y, w, h = d["x"], d["y"], d["w"], d["h"]
        pts = [(x, y), (x + w, y), (x, y + h), (x + w, y + h), (x + w // 2, y + h // 2)]
        for px, py in pts:
            dist = math.hypot(cx - px, cy - py)
            if dist < best:
                best = dist
    return best


def detect_hvac_components(image_path):
    """Color-based HVAC detection - precise, no AI"""
    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # === RED DUCTS ===
    red1 = cv2.inRange(hsv, np.array([0, 60, 60]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([170, 60, 60]), np.array([180, 255, 255]))
    red_mask = clean_mask(cv2.bitwise_or(red1, red2), 1)

    ducts = []
    for cnt in get_contours(red_mask):
        area = cv2.contourArea(cnt)
        if area < 40:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = max(w, h) / max(min(w, h), 1)
        if ratio > 2.2:
            ducts.append({
                "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                "area": round(float(area), 2)
            })

    # === BLUE VAVs ===
    blue_mask = cv2.inRange(hsv, np.array([90, 60, 60]), np.array([140, 255, 255]))
    blue_mask = clean_mask(blue_mask, 1)

    vavs = []
    for cnt in get_contours(blue_mask):
        area = cv2.contourArea(cnt)
        if area < MIN_VAV_AREA or area > MAX_VAV_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / max(h, 1)
        if ratio < 0.35 or ratio > 3.2:
            continue
        center = contour_center(cnt)
        dist = distance_to_nearest_duct(center, ducts)
        if dist > MAX_DISTANCE_TO_DUCT:
            continue
        vavs.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "area": round(float(area), 2),
            "distance_to_duct": round(float(dist), 2),
            "cx": center[0], "cy": center[1]
        })

    # Keep best VAVs sorted by closeness to ducts
    vavs = sorted(vavs, key=lambda v: (v["distance_to_duct"], -v["area"]))
    vavs = vavs[:TARGET_VAVS]

    # === GREEN AHU ===
    green_mask = cv2.inRange(hsv, np.array([40, 40, 40]), np.array([90, 255, 255]))
    green_mask = clean_mask(green_mask, 1)

    ahus = []
    for cnt in get_contours(green_mask):
        area = cv2.contourArea(cnt)
        if area < MIN_AHU_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        ahus.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "area": round(float(area), 2),
            "cx": cx, "cy": cy
        })

    ahus = sorted(ahus, key=lambda a: a["area"], reverse=True)[:1]

    return {
        "image_width": img_w,
        "image_height": img_h,
        "ducts": ducts,
        "vavs": vavs,
        "ahus": ahus
    }


def detect_walls(image_path):
    """Detect building walls and outline using edge detection"""
    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to binary
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove small components (text, symbols)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    cleaned = np.zeros_like(binary)
    for i in range(1, num_labels):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if (max(w, h) / max(min(w, h), 1) > 4 and area > 100) or (area > 800 and w > 50 and h > 50):
            cleaned[labels == i] = 255

    # Detect lines via Hough
    edges = cv2.Canny(cleaned, 50, 150)
    raw_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=70, minLineLength=60, maxLineGap=15)

    walls = []
    if raw_lines is not None:
        for line in raw_lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            length = math.sqrt(dx * dx + dy * dy)
            if length < 60:
                continue
            angle = abs(math.degrees(math.atan2(dy, dx)))
            if angle < 12 or angle > 168 or (78 < angle < 102):
                walls.append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)})

    # Find building outline (largest contour)
    wall_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for w in walls:
        cv2.line(wall_mask, (w["x1"], w["y1"]), (w["x2"], w["y2"]), 255, 8)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40)))

    contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    building_outline = []
    if contours:
        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.005 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        building_outline = [[int(p[0][0]), int(p[0][1])] for p in approx]

    # Detect rooms (interior spaces)
    inv = cv2.bitwise_not(wall_mask)
    nl, lbl, st, _ = cv2.connectedComponentsWithStats(inv)
    rooms = []
    img_area = img_h * img_w
    for i in range(1, nl):
        area = st[i, cv2.CC_STAT_AREA]
        rw = st[i, cv2.CC_STAT_WIDTH]
        rh = st[i, cv2.CC_STAT_HEIGHT]
        rx = st[i, cv2.CC_STAT_LEFT]
        ry = st[i, cv2.CC_STAT_TOP]
        if area < 3000 or area > img_area * 0.7:
            continue
        if rw < 50 or rh < 50:
            continue
        rooms.append({"x": int(rx), "y": int(ry), "w": int(rw), "h": int(rh)})

    return {"walls": walls, "rooms": rooms, "building_outline": building_outline}


def build_duct_paths(ducts):
    """Convert duct rectangles into path segments for 3D rendering"""
    paths = []
    for d in ducts:
        # Treat each duct rectangle as a line segment along its longest dimension
        x, y, w, h = d["x"], d["y"], d["w"], d["h"]
        if w >= h:
            # Horizontal duct
            paths.append({"path": [[x, y + h // 2], [x + w, y + h // 2]]})
        else:
            # Vertical duct
            paths.append({"path": [[x + w // 2, y], [x + w // 2, y + h]]})
    return paths


HOME_PAGE = '''<!DOCTYPE html>
<html>
<head>
<title>BAS Generator v14 - Color Detection + 3D</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d0f14; color: white; font-family: 'Segoe UI', Arial, sans-serif; min-height: 100vh; display: flex; align-items: center; justify-content: center; }
.card { background: #181b24; border: 1px solid #2a2f3e; border-radius: 24px; padding: 50px; text-align: center; max-width: 720px; width: 90%; box-shadow: 0 0 60px rgba(0,0,0,0.6); }
.logo { font-size: 56px; margin-bottom: 16px; }
h1 { font-size: 32px; margin-bottom: 8px; background: linear-gradient(135deg, #2d89ef, #b388ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.sub { color: #7a8099; font-size: 14px; margin-bottom: 16px; }
.notice { background: #1e3a5f; border: 1px solid #2d5a8f; border-radius: 12px; padding: 14px; margin-bottom: 24px; font-size: 13px; color: #b8d4f0; text-align: left; }
.notice b { color: #ffd54f; }
.zone { border: 2px dashed #2d3348; border-radius: 16px; padding: 36px; margin-bottom: 24px; background: #13151d; }
.zone:hover { border-color: #2d89ef; }
input[type=file] { background: transparent; color: #aab0c4; border: none; font-size: 14px; width: 100%; cursor: pointer; }
.lbl { display: block; font-size: 13px; color: #5a6280; margin-top: 10px; }
.btn { background: linear-gradient(135deg, #1a6fd4, #2d89ef); color: white; border: none; border-radius: 14px; padding: 18px 40px; font-size: 18px; font-weight: 700; cursor: pointer; width: 100%; }
.badge { display: inline-block; background: #1e2233; border: 1px solid #2a3050; border-radius: 8px; padding: 6px 14px; font-size: 12px; color: #6878a8; margin: 4px; }
.green { background: linear-gradient(135deg, #1a9e4a, #16a34a); color: white; border: none; }
.pro { background: linear-gradient(135deg, #ff9800, #ff5722); color: white; border: none; }
.note { font-size: 12px; color: #5a6280; margin-top: 14px; font-style: italic; }
.footer { color: #3a4060; font-size: 12px; margin-top: 28px; }
</style>
</head>
<body>
<div class="card">
<div class="logo">&#127970;</div>
<h1>BAS Graphic Generator v14</h1>
<p class="sub">Color Detection + 3D Render | No AI Cost</p>
<div class="notice">
<b>How to prep your plan:</b><br>
Mark VAVs in <span style="color:#5b8def">BLUE</span>,
AHU in <span style="color:#4ade80">GREEN</span>,
Ducts in <span style="color:#ef4444">RED</span><br>
Then upload the marked image.
</div>
<div style="margin-bottom:24px;">
<span class="badge green">Color Detection</span>
<span class="badge pro">Precise (No AI)</span>
<span class="badge">3D Engine</span>
<span class="badge">Free Forever</span>
</div>
<form action="/generate" method="post" enctype="multipart/form-data">
<div class="zone">
<input type="file" name="file" accept="image/png,image/jpeg,application/pdf" required>
<span class="lbl">Upload your color-marked plan</span>
</div>
<button class="btn" type="submit">Generate 3D Graphic</button>
<p class="note">Processing takes ~5 seconds</p>
</form>
<div class="footer">Made by Paolo V. and Emmanuel R.</div>
</div>
</body>
</html>'''


RESULT_PAGE = '''<!DOCTYPE html>
<html>
<head>
<title>3D BAS Graphic</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d0f14; color: white; font-family: 'Segoe UI', Arial, sans-serif; padding: 12px; }
h1 { text-align: center; font-size: 22px; margin-bottom: 4px; background: linear-gradient(135deg, #2d89ef, #b388ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.sub { text-align: center; color: #6878a8; font-size: 12px; margin-bottom: 10px; }
.stats { display: flex; justify-content: center; gap: 10px; margin: 8px 0 12px; flex-wrap: wrap; }
.stat { background: #1e2233; padding: 5px 12px; border-radius: 8px; font-size: 12px; color: #aab0c4; border: 1px solid #2a3050; }
.stat b { color: #fff; }
.viewer-3d { width: 100%; height: 80vh; background: #000000; border-radius: 12px; border: 1px solid #2a3050; overflow: hidden; position: relative; }
.controls-help { position: absolute; bottom: 10px; left: 10px; background: rgba(0,0,0,0.7); color: white; padding: 6px 12px; border-radius: 8px; font-size: 11px; z-index: 10; border: 1px solid #333; }
.actions { text-align: center; margin-top: 12px; display: flex; justify-content: center; gap: 8px; flex-wrap: wrap; }
.btn { padding: 10px 18px; border: none; border-radius: 10px; font-size: 13px; font-weight: 700; cursor: pointer; text-decoration: none; display: inline-block; }
.btn-blue { background: #1a6fd4; color: white; }
.btn-green { background: #1a9e4a; color: white; }
.btn-gray { background: #252a38; color: #aab0c4; }
.btn-orange { background: #ff7e1a; color: white; }
.footer { text-align: center; color: #3a4060; font-size: 11px; margin-top: 10px; }
</style>
</head>
<body>
<h1>Professional 3D BAS Graphic</h1>
<p class="sub">Drag = Rotate | Scroll = Zoom | Right-click = Pan</p>
<div class="stats">
<div class="stat">VAVs: <b>{{ n_vavs }}</b></div>
<div class="stat">AHUs: <b>{{ n_ahus }}</b></div>
<div class="stat">Ducts: <b>{{ n_ducts }}</b></div>
<div class="stat">Rooms: <b>{{ n_rooms }}</b></div>
</div>
<div class="viewer-3d" id="viewer">
<div class="controls-help"><b>Drag</b> rotate | <b>Scroll</b> zoom | <b>Right-click</b> pan</div>
</div>
<div class="actions">
<button onclick="screenshot()" class="btn btn-green">Download PNG</button>
<button onclick="topView()" class="btn btn-orange">Top View</button>
<button onclick="frontView()" class="btn btn-blue">Synchrony View</button>
<button onclick="resetView()" class="btn btn-blue">Reset</button>
<a href="/" class="btn btn-gray">New Plan</a>
</div>
<div class="footer">Made by Paolo V. and Emmanuel R.</div>

<script>
const detectionData = {{ detection_json | safe }};

let scene, camera, renderer, controls;
let initialCameraPos, initialTarget;
let buildingCenter = { x: 0, z: 0, sizeX: 1000, sizeZ: 1000 };

function init() {
    const container = document.getElementById('viewer');
    const width = container.clientWidth;
    const height = container.clientHeight;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    camera = new THREE.PerspectiveCamera(28, width / height, 0.1, 20000);

    renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputEncoding = THREE.sRGBEncoding;
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 50;
    controls.maxDistance = 8000;

    setupLighting();
    buildScene();

    window.addEventListener('resize', onResize);
    animate();
}

function setupLighting() {
    const ambient = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambient);

    const hemi = new THREE.HemisphereLight(0xffffff, 0x404040, 0.55);
    hemi.position.set(0, 800, 0);
    scene.add(hemi);

    const key = new THREE.DirectionalLight(0xffffff, 0.85);
    key.position.set(300, 1500, 600);
    key.castShadow = true;
    key.shadow.mapSize.width = 4096;
    key.shadow.mapSize.height = 4096;
    key.shadow.camera.left = -2000;
    key.shadow.camera.right = 2000;
    key.shadow.camera.top = 2000;
    key.shadow.camera.bottom = -2000;
    key.shadow.camera.near = 100;
    key.shadow.camera.far = 5000;
    key.shadow.bias = -0.0008;
    key.shadow.radius = 6;
    scene.add(key);

    const fill = new THREE.DirectionalLight(0xffffff, 0.35);
    fill.position.set(-500, 800, -300);
    scene.add(fill);
}

function buildScene() {
    const imgW = detectionData.image_width;
    const imgH = detectionData.image_height;

    // Use building outline if detected, otherwise full image bounds
    let outline = detectionData.building_outline;
    if (!outline || outline.length < 3) {
        outline = [[0, 0], [imgW, 0], [imgW, imgH], [0, imgH]];
    }

    const xs = outline.map(p => p[0]);
    const ys = outline.map(p => p[1]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
    const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const sizeX = Math.max(...xs) - Math.min(...xs);
    const sizeZ = Math.max(...ys) - Math.min(...ys);

    buildingCenter = { x: cx, z: cy, sizeX, sizeZ };

    const WALL_HEIGHT = 75;
    const EXT_WALL_THICKNESS = 12;
    const INT_WALL_THICKNESS = 7;

    function toWorld(p) {
        return [p[0] - cx, p[1] - cy];
    }

    function pointToWorld(x, y) {
        return [x - cx, y - cy];
    }

    // === FLOOR with TILE pattern ===
    const floorShape = new THREE.Shape();
    const outlineWorld = outline.map(toWorld);
    floorShape.moveTo(outlineWorld[0][0], outlineWorld[0][1]);
    for (let i = 1; i < outlineWorld.length; i++) {
        floorShape.lineTo(outlineWorld[i][0], outlineWorld[i][1]);
    }
    floorShape.lineTo(outlineWorld[0][0], outlineWorld[0][1]);

    // Procedural tile texture
    const tileCanvas = document.createElement('canvas');
    tileCanvas.width = 512;
    tileCanvas.height = 512;
    const tctx = tileCanvas.getContext('2d');
    tctx.fillStyle = '#a8a8ac';
    tctx.fillRect(0, 0, 512, 512);
    tctx.strokeStyle = '#ffffff';
    tctx.lineWidth = 2;
    for (let i = 0; i <= 512; i += 64) {
        tctx.beginPath();
        tctx.moveTo(i, 0);
        tctx.lineTo(i, 512);
        tctx.stroke();
        tctx.beginPath();
        tctx.moveTo(0, i);
        tctx.lineTo(512, i);
        tctx.stroke();
    }
    for (let i = 0; i < 512; i += 64) {
        for (let j = 0; j < 512; j += 64) {
            const shade = 168 + Math.random() * 12 - 6;
            tctx.fillStyle = `rgba(${shade},${shade},${shade+2},0.3)`;
            tctx.fillRect(i+1, j+1, 62, 62);
        }
    }

    const tileTexture = new THREE.CanvasTexture(tileCanvas);
    tileTexture.wrapS = THREE.RepeatWrapping;
    tileTexture.wrapT = THREE.RepeatWrapping;
    tileTexture.repeat.set(sizeX / 80, sizeZ / 80);

    const floorGeo = new THREE.ShapeGeometry(floorShape);
    const floorMat = new THREE.MeshStandardMaterial({
        map: tileTexture,
        roughness: 0.85,
        metalness: 0.05
    });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = 0.5;
    floor.receiveShadow = true;
    scene.add(floor);

    // === EXTERIOR WALLS (building outline) ===
    const extWallMat = new THREE.MeshStandardMaterial({
        color: 0x6a6e76,
        roughness: 0.9,
        metalness: 0.05
    });
    for (let i = 0; i < outlineWorld.length; i++) {
        const p1 = outlineWorld[i];
        const p2 = outlineWorld[(i + 1) % outlineWorld.length];
        buildWall(p1, p2, WALL_HEIGHT, EXT_WALL_THICKNESS, extWallMat);
    }

    // === INTERIOR WALLS from detected wall lines ===
    const intWallMat = new THREE.MeshStandardMaterial({
        color: 0x848890,
        roughness: 0.88,
        metalness: 0.03
    });

    (detectionData.walls || []).forEach(w => {
        const p1 = pointToWorld(w.x1, w.y1);
        const p2 = pointToWorld(w.x2, w.y2);
        buildWall(p1, p2, WALL_HEIGHT * 0.92, INT_WALL_THICKNESS, intWallMat);
    });

    // === DUCTS - White metal sheet style ===
    const ductMat = new THREE.MeshStandardMaterial({
        color: 0xf8f8f8,
        roughness: 0.35,
        metalness: 0.55,
        emissive: 0xffffff,
        emissiveIntensity: 0.08
    });

    (detectionData.ducts || []).forEach(d => {
        // Each duct is a rectangle - turn into 3D box
        const px = (d.x + d.w / 2) - cx;
        const pz = (d.y + d.h / 2) - cy;
        const length = Math.max(d.w, d.h);
        const width = Math.min(d.w, d.h);
        const angle = d.w >= d.h ? 0 : Math.PI / 2;

        const geo = new THREE.BoxGeometry(length, 14, Math.max(width, 16));
        const mesh = new THREE.Mesh(geo, ductMat);
        mesh.position.set(px, WALL_HEIGHT - 12, pz);
        mesh.rotation.y = angle;
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        scene.add(mesh);
    });

    // === VAV BOXES - Bright blue cubes ===
    const vavMat = new THREE.MeshStandardMaterial({
        color: 0x1e40af,
        roughness: 0.45,
        metalness: 0.35,
        emissive: 0x1e3a8a,
        emissiveIntensity: 0.18
    });

    (detectionData.vavs || []).forEach(v => {
        const px = v.cx - cx;
        const pz = v.cy - cy;
        const size = Math.max(v.w, v.h, 28);
        const geo = new THREE.BoxGeometry(size, size * 0.85, size);
        const mesh = new THREE.Mesh(geo, vavMat);
        mesh.position.set(px, WALL_HEIGHT - 22, pz);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        scene.add(mesh);
    });

    // === AHU - Green main unit ===
    (detectionData.ahus || []).forEach((a, idx) => {
        const px = a.cx - cx;
        const pz = a.cy - cy;
        const ahuMat = idx === 0
            ? new THREE.MeshStandardMaterial({
                color: 0x16a34a,
                roughness: 0.5,
                metalness: 0.35,
                emissive: 0x14532d,
                emissiveIntensity: 0.18
            })
            : new THREE.MeshStandardMaterial({
                color: 0x9ca3af,
                roughness: 0.6,
                metalness: 0.4
            });
        const w = Math.max(a.w, 50);
        const d = Math.max(a.h, 40);
        const geo = new THREE.BoxGeometry(w, 50, d);
        const mesh = new THREE.Mesh(geo, ahuMat);
        mesh.position.set(px, WALL_HEIGHT - 30, pz);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        scene.add(mesh);
    });

    // === Camera setup - low angle like Synchrony render ===
    const maxSize = Math.max(sizeX, sizeZ);
    const distance = maxSize * 1.3;
    camera.position.set(0, distance * 0.45, distance * 0.95);
    initialCameraPos = camera.position.clone();
    initialTarget = new THREE.Vector3(0, WALL_HEIGHT * 0.4, 0);
    controls.target.copy(initialTarget);
    controls.update();
}

function buildWall(p1, p2, height, thickness, material) {
    const dx = p2[0] - p1[0];
    const dz = p2[1] - p1[1];
    const length = Math.sqrt(dx * dx + dz * dz);
    if (length < 5) return;

    const angle = Math.atan2(dz, dx);
    const cx = (p1[0] + p2[0]) / 2;
    const cz = (p1[1] + p2[1]) / 2;

    const geo = new THREE.BoxGeometry(length, height, thickness);
    const mesh = new THREE.Mesh(geo, material);
    mesh.position.set(cx, height / 2, cz);
    mesh.rotation.y = -angle;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    scene.add(mesh);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function onResize() {
    const container = document.getElementById('viewer');
    const width = container.clientWidth;
    const height = container.clientHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
}

function resetView() {
    camera.position.copy(initialCameraPos);
    controls.target.copy(initialTarget);
    controls.update();
}

function topView() {
    const maxSize = Math.max(buildingCenter.sizeX, buildingCenter.sizeZ);
    camera.position.set(0, maxSize * 1.4, 0.1);
    controls.target.set(0, 0, 0);
    controls.update();
}

function frontView() {
    const maxSize = Math.max(buildingCenter.sizeX, buildingCenter.sizeZ);
    const distance = maxSize * 1.5;
    camera.position.set(0, distance * 0.35, distance);
    controls.target.set(0, 30, 0);
    controls.update();
}

function screenshot() {
    renderer.render(scene, camera);
    const dataURL = renderer.domElement.toDataURL('image/png');
    const link = document.createElement('a');
    link.download = 'bas_graphic_3d.png';
    link.href = dataURL;
    link.click();
}

init();
</script>
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
        # Color-based HVAC detection (no AI - free, fast, precise)
        hvac_data = detect_hvac_components(UPLOAD_IMAGE_PATH)

        # Wall and room detection
        arch_data = detect_walls(UPLOAD_IMAGE_PATH)

        # Combine into single detection result
        detection = {
            "image_width": hvac_data["image_width"],
            "image_height": hvac_data["image_height"],
            "vavs": hvac_data["vavs"],
            "ahus": hvac_data["ahus"],
            "ducts": hvac_data["ducts"],
            "walls": arch_data["walls"],
            "rooms": arch_data["rooms"],
            "building_outline": arch_data["building_outline"]
        }

    except Exception as e:
        error_msg = str(e)
        return "<h2 style='color:white;background:#0d0f14;padding:30px;'>Detection failed: " + error_msg + "</h2>", 500

    return render_template_string(
        RESULT_PAGE,
        detection_json=json.dumps(detection),
        n_vavs=len(detection["vavs"]),
        n_ahus=len(detection["ahus"]),
        n_ducts=len(detection["ducts"]),
        n_rooms=len(detection["rooms"])
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

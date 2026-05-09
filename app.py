from flask import Flask, request, render_template_string, jsonify
import os
import base64
import json
import math
import cv2
import numpy as np
import fitz
from io import BytesIO

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
UPLOAD_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, "mechanical_upload.png")
UPLOAD_PDF_PATH = os.path.join(UPLOAD_FOLDER, "mechanical_upload.pdf")
MARKED_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, "mechanical_marked.png")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

MIN_VAV_AREA = 40
MAX_VAV_AREA = 4000
MIN_AHU_AREA = 200
MAX_DISTANCE_TO_DUCT = 200
TARGET_VAVS = 30


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def pdf_to_png(pdf_path, out_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
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


def detect_all_components(image_path):
    """Detect VAVs (blue), AHU (green), Ducts (red), Exterior walls (purple/magenta), Interior walls (orange)"""
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
        if ratio > 1.8:
            ducts.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})

    # === BLUE VAVs ===
    blue_mask = cv2.inRange(hsv, np.array([90, 60, 60]), np.array([140, 255, 255]))
    blue_mask = clean_mask(blue_mask, 1)

    vavs = []
    for cnt in get_contours(blue_mask):
        area = cv2.contourArea(cnt)
        if area < MIN_VAV_AREA or area > MAX_VAV_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        center = contour_center(cnt)
        dist = distance_to_nearest_duct(center, ducts) if ducts else 0
        if ducts and dist > MAX_DISTANCE_TO_DUCT:
            continue
        vavs.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "cx": center[0], "cy": center[1],
            "distance_to_duct": dist
        })

    vavs = sorted(vavs, key=lambda v: v["distance_to_duct"])[:TARGET_VAVS]

    # === GREEN AHU ===
    green_mask = cv2.inRange(hsv, np.array([40, 60, 60]), np.array([85, 255, 255]))
    green_mask = clean_mask(green_mask, 1)

    ahus = []
    for cnt in get_contours(green_mask):
        area = cv2.contourArea(cnt)
        if area < MIN_AHU_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx_a = int(x + w / 2)
        cy_a = int(y + h / 2)
        ahus.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "cx": cx_a, "cy": cy_a, "area": float(area)})

    ahus = sorted(ahus, key=lambda a: a["area"], reverse=True)[:1]

    # === PURPLE/MAGENTA EXTERIOR WALLS ===
    purple_mask = cv2.inRange(hsv, np.array([140, 60, 60]), np.array([170, 255, 255]))
    purple_mask = clean_mask(purple_mask, 1)

    ext_walls = extract_wall_lines(purple_mask)

    # === ORANGE INTERIOR WALLS ===
    orange_mask = cv2.inRange(hsv, np.array([10, 100, 100]), np.array([25, 255, 255]))
    orange_mask = clean_mask(orange_mask, 1)

    int_walls = extract_wall_lines(orange_mask)

    return {
        "image_width": img_w,
        "image_height": img_h,
        "ducts": ducts,
        "vavs": vavs,
        "ahus": ahus,
        "exterior_walls": ext_walls,
        "interior_walls": int_walls
    }


def extract_wall_lines(mask):
    """Convert wall mask into line segments using skeleton + contour analysis"""
    # Dilate to connect close strokes
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    walls = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue
        # Approximate contour to a polyline
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        points = [[int(p[0][0]), int(p[0][1])] for p in approx]
        if len(points) >= 2:
            walls.append({"points": points})
    return walls


HOME_PAGE = '''<!DOCTYPE html>
<html>
<head>
<title>BAS Generator v15 - Visual Editor</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d0f14; color: white; font-family: 'Segoe UI', Arial, sans-serif; min-height: 100vh; display: flex; align-items: center; justify-content: center; }
.card { background: #181b24; border: 1px solid #2a2f3e; border-radius: 24px; padding: 50px; text-align: center; max-width: 720px; width: 90%; box-shadow: 0 0 60px rgba(0,0,0,0.6); }
.logo { font-size: 56px; margin-bottom: 16px; }
h1 { font-size: 32px; margin-bottom: 8px; background: linear-gradient(135deg, #2d89ef, #b388ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.sub { color: #7a8099; font-size: 14px; margin-bottom: 16px; }
.zone { border: 2px dashed #2d3348; border-radius: 16px; padding: 36px; margin-bottom: 24px; background: #13151d; }
.zone:hover { border-color: #2d89ef; }
input[type=file] { background: transparent; color: #aab0c4; border: none; font-size: 14px; width: 100%; cursor: pointer; }
.lbl { display: block; font-size: 13px; color: #5a6280; margin-top: 10px; }
.btn { background: linear-gradient(135deg, #1a6fd4, #2d89ef); color: white; border: none; border-radius: 14px; padding: 18px 40px; font-size: 18px; font-weight: 700; cursor: pointer; width: 100%; }
.feature { background: #13151d; border: 1px solid #2a3050; border-radius: 12px; padding: 14px; margin: 8px 0; text-align: left; font-size: 13px; color: #aab0c4; display: flex; gap: 10px; align-items: center; }
.color-dot { width: 14px; height: 14px; border-radius: 50%; flex-shrink: 0; }
.footer { color: #3a4060; font-size: 12px; margin-top: 28px; }
</style>
</head>
<body>
<div class="card">
<div class="logo">&#127970;</div>
<h1>BAS Graphic Generator v15</h1>
<p class="sub">Visual Editor + 3D Render Engine</p>
<div style="text-align: left; margin-bottom: 24px;">
<div class="feature"><div class="color-dot" style="background:#1e40af"></div> Mark VAVs in BLUE</div>
<div class="feature"><div class="color-dot" style="background:#16a34a"></div> Mark AHU in GREEN</div>
<div class="feature"><div class="color-dot" style="background:#dc2626"></div> Mark Ducts in RED</div>
<div class="feature"><div class="color-dot" style="background:#9333ea"></div> Trace EXTERIOR walls in PURPLE</div>
<div class="feature"><div class="color-dot" style="background:#ea580c"></div> Trace INTERIOR walls in ORANGE</div>
</div>
<form action="/upload" method="post" enctype="multipart/form-data">
<div class="zone">
<input type="file" name="file" accept="image/png,image/jpeg,application/pdf" required>
<span class="lbl">Upload original mechanical plan (no marking needed)</span>
</div>
<button class="btn" type="submit">Open Visual Editor</button>
</form>
<div class="footer">Made by Paolo V. and Emmanuel R.</div>
</div>
</body>
</html>'''


EDITOR_PAGE = '''<!DOCTYPE html>
<html>
<head>
<title>Visual Editor</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d0f14; color: white; font-family: 'Segoe UI', Arial, sans-serif; padding: 12px; height: 100vh; display: flex; flex-direction: column; }
.topbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
h1 { font-size: 18px; background: linear-gradient(135deg, #2d89ef, #b388ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.toolbar { background: #181b24; border: 1px solid #252a38; border-radius: 12px; padding: 12px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 12px; }
.tool-btn { padding: 8px 14px; border: 2px solid transparent; background: #1e2233; color: white; border-radius: 8px; cursor: pointer; font-size: 13px; font-weight: 600; display: flex; align-items: center; gap: 6px; }
.tool-btn:hover { background: #252a38; }
.tool-btn.active { border-color: #fff; }
.color-swatch { width: 16px; height: 16px; border-radius: 4px; border: 1px solid rgba(255,255,255,0.3); }
.divider { width: 1px; background: #333; height: 28px; margin: 0 4px; }
.size-control { display: flex; align-items: center; gap: 8px; padding: 4px 12px; background: #1e2233; border-radius: 8px; }
.size-control input { width: 80px; }
.canvas-wrap { flex: 1; position: relative; background: #000; border-radius: 12px; border: 1px solid #2a3050; overflow: hidden; }
#canvasContainer { width: 100%; height: 100%; position: relative; overflow: auto; cursor: crosshair; }
canvas { display: block; }
.action-btn { padding: 10px 22px; border: none; border-radius: 10px; font-size: 14px; font-weight: 700; cursor: pointer; }
.btn-green { background: #16a34a; color: white; }
.btn-blue { background: #1a6fd4; color: white; }
.btn-red { background: #dc2626; color: white; }
.btn-gray { background: #333; color: white; }
.spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid #fff; border-top-color: transparent; border-radius: 50%; animation: spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.loading-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.85); display: none; align-items: center; justify-content: center; z-index: 100; flex-direction: column; gap: 16px; }
.loading-overlay.active { display: flex; }
.hint { font-size: 12px; color: #888; padding: 6px 12px; }
</style>
</head>
<body>
<div class="topbar">
<h1>Visual Editor - Mark your plan</h1>
<div style="display: flex; gap: 8px;">
<button onclick="undo()" class="action-btn btn-gray">&#9100; Undo</button>
<button onclick="clearAll()" class="action-btn btn-red">Clear All</button>
<button onclick="generate3D()" class="action-btn btn-green">Generate 3D &rarr;</button>
</div>
</div>

<div class="toolbar">
<button class="tool-btn active" data-color="#1e40af" data-name="vav" onclick="selectTool(this)">
<div class="color-swatch" style="background:#1e40af"></div> VAV (Blue)
</button>
<button class="tool-btn" data-color="#16a34a" data-name="ahu" onclick="selectTool(this)">
<div class="color-swatch" style="background:#16a34a"></div> AHU (Green)
</button>
<button class="tool-btn" data-color="#dc2626" data-name="duct" onclick="selectTool(this)">
<div class="color-swatch" style="background:#dc2626"></div> Duct (Red)
</button>
<button class="tool-btn" data-color="#9333ea" data-name="extwall" onclick="selectTool(this)">
<div class="color-swatch" style="background:#9333ea"></div> Ext Wall (Purple)
</button>
<button class="tool-btn" data-color="#ea580c" data-name="intwall" onclick="selectTool(this)">
<div class="color-swatch" style="background:#ea580c"></div> Int Wall (Orange)
</button>

<div class="divider"></div>

<div class="size-control">
<span style="font-size:12px;color:#aab0c4;">Brush:</span>
<input type="range" id="brushSize" min="3" max="40" value="10">
<span id="brushVal" style="font-size:12px;color:white;width:20px;">10</span>
</div>

<div class="divider"></div>

<span class="hint">&#128161; Click+drag to draw | For VAV/AHU click once</span>
</div>

<div class="canvas-wrap">
<div id="canvasContainer">
<canvas id="bgCanvas" style="position:absolute;top:0;left:0;"></canvas>
<canvas id="drawCanvas" style="position:absolute;top:0;left:0;"></canvas>
</div>
</div>

<div class="loading-overlay" id="loading">
<div class="spinner"></div>
<div style="color:white;font-size:14px;">Detecting components and building 3D...</div>
</div>

<script>
const imgB64 = '{{ image_b64 }}';
let bgCanvas = document.getElementById('bgCanvas');
let drawCanvas = document.getElementById('drawCanvas');
let bgCtx = bgCanvas.getContext('2d');
let drawCtx = drawCanvas.getContext('2d');

let currentTool = 'vav';
let currentColor = '#1e40af';
let brushSize = 10;
let drawing = false;
let lastX = 0, lastY = 0;
let history = [];

const img = new Image();
img.onload = function() {
    bgCanvas.width = img.width;
    bgCanvas.height = img.height;
    drawCanvas.width = img.width;
    drawCanvas.height = img.height;
    bgCtx.drawImage(img, 0, 0);
    saveState();
};
img.src = 'data:image/png;base64,' + imgB64;

function selectTool(btn) {
    document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentTool = btn.dataset.name;
    currentColor = btn.dataset.color;
}

document.getElementById('brushSize').addEventListener('input', function(e) {
    brushSize = parseInt(e.target.value);
    document.getElementById('brushVal').textContent = brushSize;
});

function getMousePos(e) {
    const rect = drawCanvas.getBoundingClientRect();
    const scaleX = drawCanvas.width / rect.width;
    const scaleY = drawCanvas.height / rect.height;
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
    };
}

drawCanvas.addEventListener('mousedown', function(e) {
    drawing = true;
    const pos = getMousePos(e);
    lastX = pos.x;
    lastY = pos.y;

    // Single click for VAV/AHU - draw filled circle
    if (currentTool === 'vav') {
        drawCtx.fillStyle = currentColor;
        drawCtx.beginPath();
        drawCtx.arc(pos.x, pos.y, 10, 0, Math.PI * 2);
        drawCtx.fill();
        drawing = false;
        saveState();
    } else if (currentTool === 'ahu') {
        drawCtx.fillStyle = currentColor;
        drawCtx.fillRect(pos.x - 18, pos.y - 14, 36, 28);
        drawing = false;
        saveState();
    } else {
        // Walls and ducts - start a stroke
        drawCtx.strokeStyle = currentColor;
        drawCtx.lineWidth = brushSize;
        drawCtx.lineCap = 'round';
        drawCtx.lineJoin = 'round';
        drawCtx.beginPath();
        drawCtx.moveTo(pos.x, pos.y);
    }
});

drawCanvas.addEventListener('mousemove', function(e) {
    if (!drawing) return;
    const pos = getMousePos(e);
    drawCtx.lineTo(pos.x, pos.y);
    drawCtx.stroke();
});

drawCanvas.addEventListener('mouseup', function() {
    if (drawing) {
        saveState();
    }
    drawing = false;
});

drawCanvas.addEventListener('mouseleave', function() {
    drawing = false;
});

function saveState() {
    history.push(drawCanvas.toDataURL());
    if (history.length > 30) history.shift();
}

function undo() {
    if (history.length < 2) return;
    history.pop();
    const prev = history[history.length - 1];
    const tmpImg = new Image();
    tmpImg.onload = function() {
        drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
        drawCtx.drawImage(tmpImg, 0, 0);
    };
    tmpImg.src = prev;
}

function clearAll() {
    if (!confirm('Clear all markings?')) return;
    drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    history = [];
    saveState();
}

async function generate3D() {
    document.getElementById('loading').classList.add('active');

    // Composite both canvases into one
    const finalCanvas = document.createElement('canvas');
    finalCanvas.width = bgCanvas.width;
    finalCanvas.height = bgCanvas.height;
    const fctx = finalCanvas.getContext('2d');
    fctx.drawImage(bgCanvas, 0, 0);
    fctx.drawImage(drawCanvas, 0, 0);

    const dataUrl = finalCanvas.toDataURL('image/png');

    try {
        const response = await fetch('/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataUrl })
        });
        const result = await response.json();
        if (result.success) {
            window.location.href = '/render3d';
        } else {
            alert('Error: ' + result.error);
            document.getElementById('loading').classList.remove('active');
        }
    } catch (err) {
        alert('Error: ' + err.message);
        document.getElementById('loading').classList.remove('active');
    }
}
</script>
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
.viewer-3d { width: 100%; height: 78vh; background: #000; border-radius: 12px; border: 1px solid #2a3050; overflow: hidden; position: relative; }
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
<div class="stat">Ext Walls: <b>{{ n_ext }}</b></div>
<div class="stat">Int Walls: <b>{{ n_int }}</b></div>
</div>
<div class="viewer-3d" id="viewer"></div>
<div class="actions">
<button onclick="screenshot()" class="btn btn-green">Download PNG</button>
<button onclick="topView()" class="btn btn-orange">Top View</button>
<button onclick="frontView()" class="btn btn-blue">Synchrony View</button>
<button onclick="resetView()" class="btn btn-blue">Reset</button>
<a href="/editor" class="btn btn-gray">Edit Markings</a>
<a href="/" class="btn btn-gray">New Plan</a>
</div>
<div class="footer">Made by Paolo V. and Emmanuel R.</div>

<script>
const data = {{ detection_json | safe }};

let scene, camera, renderer, controls, initPos, initTarget;

function init() {
    const c = document.getElementById('viewer');
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    camera = new THREE.PerspectiveCamera(28, c.clientWidth / c.clientHeight, 0.1, 20000);
    renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setSize(c.clientWidth, c.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    c.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    setupLighting();
    buildScene();
    window.addEventListener('resize', onResize);
    animate();
}

function setupLighting() {
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
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
    key.shadow.bias = -0.0008;
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.35);
    fill.position.set(-500, 800, -300);
    scene.add(fill);
}

function buildScene() {
    const cx = data.image_width / 2;
    const cy = data.image_height / 2;
    const sizeX = data.image_width;
    const sizeZ = data.image_height;
    const WALL_HEIGHT = 75;

    // Tile floor
    const tileCanvas = document.createElement('canvas');
    tileCanvas.width = 512;
    tileCanvas.height = 512;
    const tctx = tileCanvas.getContext('2d');
    tctx.fillStyle = '#a8a8ac';
    tctx.fillRect(0, 0, 512, 512);
    tctx.strokeStyle = '#ffffff';
    tctx.lineWidth = 2;
    for (let i = 0; i <= 512; i += 64) {
        tctx.beginPath(); tctx.moveTo(i, 0); tctx.lineTo(i, 512); tctx.stroke();
        tctx.beginPath(); tctx.moveTo(0, i); tctx.lineTo(512, i); tctx.stroke();
    }
    const tex = new THREE.CanvasTexture(tileCanvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(sizeX / 80, sizeZ / 80);

    const floorGeo = new THREE.PlaneGeometry(sizeX, sizeZ);
    const floor = new THREE.Mesh(floorGeo, new THREE.MeshStandardMaterial({ map: tex, roughness: 0.85 }));
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = 0.5;
    floor.receiveShadow = true;
    scene.add(floor);

    // Exterior walls
    const extMat = new THREE.MeshStandardMaterial({ color: 0x6a6e76, roughness: 0.9 });
    (data.exterior_walls || []).forEach(w => {
        const pts = w.points;
        for (let i = 0; i < pts.length - 1; i++) {
            buildWall(
                [pts[i][0] - cx, pts[i][1] - cy],
                [pts[i + 1][0] - cx, pts[i + 1][1] - cy],
                WALL_HEIGHT, 12, extMat
            );
        }
        if (pts.length >= 3) {
            buildWall(
                [pts[pts.length - 1][0] - cx, pts[pts.length - 1][1] - cy],
                [pts[0][0] - cx, pts[0][1] - cy],
                WALL_HEIGHT, 12, extMat
            );
        }
    });

    // Interior walls
    const intMat = new THREE.MeshStandardMaterial({ color: 0x848890, roughness: 0.88 });
    (data.interior_walls || []).forEach(w => {
        const pts = w.points;
        for (let i = 0; i < pts.length - 1; i++) {
            buildWall(
                [pts[i][0] - cx, pts[i][1] - cy],
                [pts[i + 1][0] - cx, pts[i + 1][1] - cy],
                WALL_HEIGHT * 0.92, 7, intMat
            );
        }
    });

    // Ducts
    const ductMat = new THREE.MeshStandardMaterial({ color: 0xf8f8f8, roughness: 0.35, metalness: 0.55 });
    (data.ducts || []).forEach(d => {
        const px = (d.x + d.w / 2) - cx;
        const pz = (d.y + d.h / 2) - cy;
        const length = Math.max(d.w, d.h);
        const width = Math.max(Math.min(d.w, d.h), 14);
        const angle = d.w >= d.h ? 0 : Math.PI / 2;
        const geo = new THREE.BoxGeometry(length, 14, width);
        const m = new THREE.Mesh(geo, ductMat);
        m.position.set(px, WALL_HEIGHT - 12, pz);
        m.rotation.y = angle;
        m.castShadow = true;
        scene.add(m);
    });

    // VAVs
    const vavMat = new THREE.MeshStandardMaterial({ color: 0x1e40af, roughness: 0.45, metalness: 0.35, emissive: 0x1e3a8a, emissiveIntensity: 0.18 });
    (data.vavs || []).forEach(v => {
        const geo = new THREE.BoxGeometry(28, 24, 28);
        const m = new THREE.Mesh(geo, vavMat);
        m.position.set(v.cx - cx, WALL_HEIGHT - 22, v.cy - cy);
        m.castShadow = true;
        scene.add(m);
    });

    // AHU
    (data.ahus || []).forEach(a => {
        const w = Math.max(a.w, 50);
        const d = Math.max(a.h, 40);
        const mat = new THREE.MeshStandardMaterial({ color: 0x16a34a, roughness: 0.5, metalness: 0.35, emissive: 0x14532d, emissiveIntensity: 0.18 });
        const geo = new THREE.BoxGeometry(w, 50, d);
        const m = new THREE.Mesh(geo, mat);
        m.position.set(a.cx - cx, WALL_HEIGHT - 30, a.cy - cy);
        m.castShadow = true;
        scene.add(m);
    });

    const maxSize = Math.max(sizeX, sizeZ);
    const dist = maxSize * 1.3;
    camera.position.set(0, dist * 0.45, dist * 0.95);
    initPos = camera.position.clone();
    initTarget = new THREE.Vector3(0, WALL_HEIGHT * 0.4, 0);
    controls.target.copy(initTarget);
    controls.update();
}

function buildWall(p1, p2, height, thickness, mat) {
    const dx = p2[0] - p1[0], dz = p2[1] - p1[1];
    const length = Math.sqrt(dx * dx + dz * dz);
    if (length < 5) return;
    const angle = Math.atan2(dz, dx);
    const geo = new THREE.BoxGeometry(length, height, thickness);
    const m = new THREE.Mesh(geo, mat);
    m.position.set((p1[0] + p2[0]) / 2, height / 2, (p1[1] + p2[1]) / 2);
    m.rotation.y = -angle;
    m.castShadow = true;
    m.receiveShadow = true;
    scene.add(m);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function onResize() {
    const c = document.getElementById('viewer');
    camera.aspect = c.clientWidth / c.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(c.clientWidth, c.clientHeight);
}

function resetView() {
    camera.position.copy(initPos);
    controls.target.copy(initTarget);
    controls.update();
}

function topView() {
    const max = Math.max(data.image_width, data.image_height);
    camera.position.set(0, max * 1.4, 0.1);
    controls.target.set(0, 0, 0);
    controls.update();
}

function frontView() {
    const max = Math.max(data.image_width, data.image_height);
    const dist = max * 1.5;
    camera.position.set(0, dist * 0.35, dist);
    controls.target.set(0, 30, 0);
    controls.update();
}

function screenshot() {
    renderer.render(scene, camera);
    const url = renderer.domElement.toDataURL('image/png');
    const link = document.createElement('a');
    link.download = 'bas_3d.png';
    link.href = url;
    link.click();
}

init();
</script>
</body>
</html>'''


@app.route("/")
def home():
    return HOME_PAGE


@app.route("/upload", methods=["POST"])
def upload():
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

    # Resize to manageable size for web canvas
    max_dim = 1400
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    cv2.imwrite(UPLOAD_IMAGE_PATH, img)

    return render_template_string(EDITOR_PAGE, image_b64=image_to_base64(UPLOAD_IMAGE_PATH))


@app.route("/editor")
def editor():
    if not os.path.exists(UPLOAD_IMAGE_PATH):
        return "<h2 style='color:white;background:#0d0f14;padding:30px;'>No plan uploaded yet. <a href='/' style='color:#2d89ef'>Upload one</a></h2>"
    return render_template_string(EDITOR_PAGE, image_b64=image_to_base64(UPLOAD_IMAGE_PATH))


@app.route("/process", methods=["POST"])
def process():
    try:
        data = request.get_json()
        img_b64 = data["image"].split(",")[1]
        img_bytes = base64.b64decode(img_b64)

        with open(MARKED_IMAGE_PATH, "wb") as f:
            f.write(img_bytes)

        detection = detect_all_components(MARKED_IMAGE_PATH)

        with open(os.path.join(OUTPUT_FOLDER, "detection.json"), "w") as f:
            json.dump(detection, f)

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/render3d")
def render3d():
    try:
        with open(os.path.join(OUTPUT_FOLDER, "detection.json"), "r") as f:
            detection = json.load(f)
    except FileNotFoundError:
        return "<h2 style='color:white;background:#0d0f14;padding:30px;'>No detection yet. <a href='/' style='color:#2d89ef'>Start over</a></h2>"

    return render_template_string(
        RESULT_PAGE,
        detection_json=json.dumps(detection),
        n_vavs=len(detection.get("vavs", [])),
        n_ahus=len(detection.get("ahus", [])),
        n_ducts=len(detection.get("ducts", [])),
        n_ext=len(detection.get("exterior_walls", [])),
        n_int=len(detection.get("interior_walls", []))
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

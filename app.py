from flask import Flask, request, render_template_string, jsonify
import os
import base64
import json
import cv2
import fitz

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
UPLOAD_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, "mechanical_upload.png")
UPLOAD_PDF_PATH = os.path.join(UPLOAD_FOLDER, "mechanical_upload.pdf")

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


HOME_PAGE = '''<!DOCTYPE html>
<html>
<head>
<title>BAS Generator v19 - Synchrony Style</title>
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
.badge { display: inline-block; background: linear-gradient(135deg, #ff9800, #ff5722); color: white; padding: 4px 12px; font-size: 11px; border-radius: 8px; margin-left: 6px; }
</style>
</head>
<body>
<div class="card">
<div class="logo">&#127970;</div>
<h1>BAS Generator v19 <span class="badge">SYNCHRONY</span></h1>
<p class="sub">Professional 3D HVAC graphics - Tracer Synchrony style</p>
<div style="text-align: left; margin-bottom: 24px;">
<div class="feature"><div class="color-dot" style="background:#9333ea"></div> Click corners for walls (always straight)</div>
<div class="feature"><div class="color-dot" style="background:#1e40af"></div> Click to place VAVs</div>
<div class="feature"><div class="color-dot" style="background:#16a34a"></div> Click to place AHU</div>
<div class="feature"><div class="color-dot" style="background:#000"></div> Two clicks for duct lines</div>
<div class="feature"><div class="color-dot" style="background:#fff;border:1px solid #888"></div> Synchrony-style render with soft shadows</div>
</div>
<form action="/upload" method="post" enctype="multipart/form-data">
<div class="zone">
<input type="file" name="file" accept="image/png,image/jpeg,application/pdf" required>
<span class="lbl">Upload original mechanical plan</span>
</div>
<button class="btn" type="submit">Open Pro CAD Editor</button>
</form>
<div class="footer">Made by Paolo V. and Emmanuel R.</div>
</div>
</body>
</html>'''


EDITOR_PAGE = '''<!DOCTYPE html>
<html>
<head>
<title>Pro CAD Editor</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d0f14; color: white; font-family: 'Segoe UI', Arial, sans-serif; padding: 8px; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
.topbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
h1 { font-size: 16px; background: linear-gradient(135deg, #2d89ef, #b388ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.toolbar { background: #181b24; border: 1px solid #252a38; border-radius: 10px; padding: 8px; display: flex; gap: 6px; flex-wrap: wrap; align-items: center; margin-bottom: 8px; }
.tool-btn { padding: 8px 12px; border: 2px solid transparent; background: #1e2233; color: white; border-radius: 8px; cursor: pointer; font-size: 12px; font-weight: 600; display: flex; align-items: center; gap: 6px; white-space: nowrap; }
.tool-btn:hover { background: #252a38; }
.tool-btn.active { border-color: #fff; background: #2d3348; }
.color-swatch { width: 14px; height: 14px; border-radius: 3px; border: 1px solid rgba(255,255,255,0.3); }
.divider { width: 1px; background: #333; height: 24px; margin: 0 3px; }
.canvas-wrap { flex: 1; position: relative; background: #1a1a1a; border-radius: 10px; border: 1px solid #2a3050; overflow: hidden; }
#canvasContainer { width: 100%; height: 100%; position: relative; overflow: auto; }
canvas { display: block; }
.action-btn { padding: 8px 16px; border: none; border-radius: 8px; font-size: 13px; font-weight: 700; cursor: pointer; }
.btn-green { background: #16a34a; color: white; }
.btn-blue { background: #1a6fd4; color: white; }
.btn-red { background: #dc2626; color: white; }
.btn-gray { background: #333; color: white; }
.spinner { display: inline-block; width: 20px; height: 20px; border: 3px solid #fff; border-top-color: transparent; border-radius: 50%; animation: spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.loading-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.85); display: none; align-items: center; justify-content: center; z-index: 100; flex-direction: column; gap: 16px; }
.loading-overlay.active { display: flex; }
.status { padding: 4px 12px; background: #1e2233; border-radius: 6px; font-size: 11px; color: #aab0c4; min-width: 200px; text-align: center; }
.cursor-cross { cursor: crosshair; }
.cursor-move { cursor: move; }
.cursor-default { cursor: default; }
</style>
</head>
<body>
<div class="topbar">
<h1>Pro CAD Editor &mdash; Click points to draw</h1>
<div style="display: flex; gap: 6px;">
<button onclick="undo()" class="action-btn btn-gray">&#8617; Undo</button>
<button onclick="clearAll()" class="action-btn btn-red">Clear</button>
<button onclick="generate3D()" class="action-btn btn-green">Generate 3D &rarr;</button>
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
<div class="color-swatch" style="background:#000;border:1px solid #fff"></div> Duct
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

<span class="status" id="statusBar">Click to start a wall. Double-click to finish.</span>
</div>

<div class="canvas-wrap">
<div id="canvasContainer">
<canvas id="bgCanvas" style="position:absolute;top:0;left:0;"></canvas>
<canvas id="drawCanvas" class="cursor-cross" style="position:absolute;top:0;left:0;"></canvas>
</div>
</div>

<div class="loading-overlay" id="loading">
<div class="spinner"></div>
<div style="color:white;font-size:14px;">Building Synchrony-style 3D...</div>
</div>

<script>
const imgB64 = '{{ image_b64 }}';

let bgCanvas = document.getElementById('bgCanvas');
let drawCanvas = document.getElementById('drawCanvas');
let bgCtx = bgCanvas.getContext('2d');
let drawCtx = drawCanvas.getContext('2d');

let currentTool = 'extwall';
let elements = [];
let history = [];
let currentPolyline = null;
let hoverPoint = null;
let selectedElement = null;
let dragOffset = null;

const COLORS = {
    extwall: '#9333ea',
    intwall: '#ea580c',
    duct: '#000000',
    vav: '#1e40af',
    ahu: '#16a34a',
    diffuser: '#ffffff'
};

const STATUS_TEXTS = {
    extwall: 'Click corners of building PERIMETER. Double-click to close polygon.',
    intwall: 'Click corners of an INTERIOR WALL. Double-click to finish.',
    duct: 'Click TWO points for a duct line.',
    vav: 'Click to place a VAV.',
    ahu: 'Click to place the AHU.',
    diffuser: 'Click to place a diffuser.',
    move: 'Click and drag any element to move it.',
    delete: 'Click any element to delete it.'
};

const img = new Image();
img.onload = function() {
    bgCanvas.width = img.width;
    bgCanvas.height = img.height;
    drawCanvas.width = img.width;
    drawCanvas.height = img.height;
    bgCtx.drawImage(img, 0, 0);
    saveState();
    redraw();
};
img.src = 'data:image/png;base64,' + imgB64;

function selectTool(btn) {
    document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentTool = btn.dataset.tool;
    document.getElementById('statusBar').textContent = STATUS_TEXTS[currentTool] || '';

    if (currentPolyline) {
        if (currentPolyline.points.length >= 2) {
            elements.push(currentPolyline);
        }
        currentPolyline = null;
        saveState();
    }

    drawCanvas.className = '';
    if (currentTool === 'move') drawCanvas.classList.add('cursor-move');
    else if (currentTool === 'delete') drawCanvas.classList.add('cursor-default');
    else drawCanvas.classList.add('cursor-cross');

    redraw();
}

function getMousePos(e) {
    const rect = drawCanvas.getBoundingClientRect();
    const scaleX = drawCanvas.width / rect.width;
    const scaleY = drawCanvas.height / rect.height;
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
    };
}

drawCanvas.addEventListener('click', function(e) {
    const pos = getMousePos(e);

    if (currentTool === 'delete') {
        const idx = findElementAt(pos);
        if (idx !== -1) {
            elements.splice(idx, 1);
            saveState();
            redraw();
        }
        return;
    }

    if (currentTool === 'move') return;

    if (currentTool === 'vav' || currentTool === 'ahu' || currentTool === 'diffuser') {
        elements.push({ type: currentTool, x: pos.x, y: pos.y });
        saveState();
        redraw();
        return;
    }

    if (currentTool === 'extwall' || currentTool === 'intwall') {
        if (!currentPolyline) {
            currentPolyline = { type: currentTool, points: [{ x: pos.x, y: pos.y }] };
        } else {
            currentPolyline.points.push({ x: pos.x, y: pos.y });
        }
        redraw();
        return;
    }

    if (currentTool === 'duct') {
        if (!currentPolyline) {
            currentPolyline = { type: 'duct', points: [{ x: pos.x, y: pos.y }] };
        } else {
            currentPolyline.points.push({ x: pos.x, y: pos.y });
            elements.push(currentPolyline);
            currentPolyline = null;
            saveState();
        }
        redraw();
        return;
    }
});

drawCanvas.addEventListener('dblclick', function(e) {
    if (currentPolyline && currentPolyline.points.length >= 2) {
        if (currentPolyline.type === 'extwall' && currentPolyline.points.length >= 3) {
            currentPolyline.closed = true;
        }
        elements.push(currentPolyline);
        currentPolyline = null;
        saveState();
        redraw();
    }
});

drawCanvas.addEventListener('mousemove', function(e) {
    const pos = getMousePos(e);
    hoverPoint = pos;

    if (currentTool === 'move' && selectedElement && dragOffset) {
        moveElement(selectedElement, pos.x - dragOffset.x, pos.y - dragOffset.y);
        const center = getElementCenter(selectedElement);
        dragOffset = { x: pos.x - center.x, y: pos.y - center.y };
        redraw();
        return;
    }

    if (currentPolyline) redraw();
});

drawCanvas.addEventListener('mousedown', function(e) {
    if (currentTool !== 'move') return;
    const pos = getMousePos(e);
    const idx = findElementAt(pos);
    if (idx !== -1) {
        selectedElement = elements[idx];
        const center = getElementCenter(selectedElement);
        dragOffset = { x: pos.x - center.x, y: pos.y - center.y };
    }
});

drawCanvas.addEventListener('mouseup', function() {
    if (selectedElement) {
        saveState();
        selectedElement = null;
        dragOffset = null;
    }
});

document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && currentPolyline) {
        currentPolyline = null;
        redraw();
    }
});

function findElementAt(pos) {
    for (let i = elements.length - 1; i >= 0; i--) {
        const el = elements[i];
        if (el.type === 'vav' || el.type === 'ahu' || el.type === 'diffuser') {
            const dx = pos.x - el.x;
            const dy = pos.y - el.y;
            if (Math.sqrt(dx * dx + dy * dy) < 25) return i;
        } else {
            for (const p of el.points) {
                const dx = pos.x - p.x;
                const dy = pos.y - p.y;
                if (Math.sqrt(dx * dx + dy * dy) < 15) return i;
            }
        }
    }
    return -1;
}

function getElementCenter(el) {
    if (el.type === 'vav' || el.type === 'ahu' || el.type === 'diffuser') {
        return { x: el.x, y: el.y };
    }
    let sx = 0, sy = 0;
    for (const p of el.points) { sx += p.x; sy += p.y; }
    return { x: sx / el.points.length, y: sy / el.points.length };
}

function moveElement(el, dx, dy) {
    if (el.type === 'vav' || el.type === 'ahu' || el.type === 'diffuser') {
        el.x += dx;
        el.y += dy;
    } else {
        for (const p of el.points) { p.x += dx; p.y += dy; }
    }
}

function redraw() {
    drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    for (const el of elements) drawElement(el);

    if (currentPolyline) {
        drawElement(currentPolyline, true);
        if (hoverPoint && currentPolyline.points.length > 0) {
            const last = currentPolyline.points[currentPolyline.points.length - 1];
            drawCtx.strokeStyle = COLORS[currentPolyline.type];
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

function drawElement(el, isInProgress = false) {
    const color = COLORS[el.type] || '#fff';

    if (el.type === 'vav') {
        drawCtx.fillStyle = color;
        drawCtx.strokeStyle = '#fff';
        drawCtx.lineWidth = 2;
        drawCtx.beginPath();
        drawCtx.arc(el.x, el.y, 14, 0, Math.PI * 2);
        drawCtx.fill();
        drawCtx.stroke();
        return;
    }

    if (el.type === 'ahu') {
        drawCtx.fillStyle = color;
        drawCtx.strokeStyle = '#fff';
        drawCtx.lineWidth = 2;
        drawCtx.fillRect(el.x - 24, el.y - 18, 48, 36);
        drawCtx.strokeRect(el.x - 24, el.y - 18, 48, 36);
        return;
    }

    if (el.type === 'diffuser') {
        drawCtx.fillStyle = color;
        drawCtx.strokeStyle = '#666';
        drawCtx.lineWidth = 1.5;
        drawCtx.beginPath();
        drawCtx.arc(el.x, el.y, 6, 0, Math.PI * 2);
        drawCtx.fill();
        drawCtx.stroke();
        return;
    }

    if (!el.points || el.points.length === 0) return;

    drawCtx.strokeStyle = color;
    drawCtx.lineWidth = el.type === 'duct' ? 4 : 5;
    drawCtx.lineCap = 'round';
    drawCtx.lineJoin = 'round';
    drawCtx.beginPath();
    drawCtx.moveTo(el.points[0].x, el.points[0].y);
    for (let i = 1; i < el.points.length; i++) {
        drawCtx.lineTo(el.points[i].x, el.points[i].y);
    }
    if (el.closed) drawCtx.closePath();
    drawCtx.stroke();

    drawCtx.fillStyle = color;
    for (const p of el.points) {
        drawCtx.beginPath();
        drawCtx.arc(p.x, p.y, 5, 0, Math.PI * 2);
        drawCtx.fill();
    }

    if (isInProgress && el.points.length > 0) {
        const first = el.points[0];
        drawCtx.strokeStyle = '#fff';
        drawCtx.lineWidth = 2;
        drawCtx.beginPath();
        drawCtx.arc(first.x, first.y, 8, 0, Math.PI * 2);
        drawCtx.stroke();
    }
}

function saveState() {
    history.push(JSON.stringify(elements));
    if (history.length > 40) history.shift();
}

function undo() {
    if (history.length < 2) return;
    history.pop();
    elements = JSON.parse(history[history.length - 1]);
    currentPolyline = null;
    redraw();
}

function clearAll() {
    if (!confirm('Clear everything?')) return;
    elements = [];
    currentPolyline = null;
    saveState();
    redraw();
}

async function generate3D() {
    if (currentPolyline && currentPolyline.points.length >= 2) {
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
<title>Synchrony 3D BAS</title>
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
.viewer-3d { width: 100%; height: 78vh; background: #1f1f23; border-radius: 12px; border: 1px solid #2a3050; overflow: hidden; position: relative; }
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
<h1>Synchrony-Style 3D BAS Graphic</h1>
<p class="sub">Drag = Rotate | Scroll = Zoom | Right-click = Pan</p>
<div class="stats">
<div class="stat">VAVs: <b>{{ n_vavs }}</b></div>
<div class="stat">AHUs: <b>{{ n_ahus }}</b></div>
<div class="stat">Ducts: <b>{{ n_ducts }}</b></div>
<div class="stat">Diffusers: <b>{{ n_diffs }}</b></div>
<div class="stat">Ext Walls: <b>{{ n_ext }}</b></div>
<div class="stat">Int Walls: <b>{{ n_int }}</b></div>
</div>
<div class="viewer-3d" id="viewer"></div>
<div class="actions">
<button onclick="screenshot()" class="btn btn-green">Download PNG</button>
<button onclick="topView()" class="btn btn-orange">Top View</button>
<button onclick="synchView()" class="btn btn-blue">Synchrony View</button>
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
    scene.background = new THREE.Color(0x1f1f23);
    camera = new THREE.PerspectiveCamera(25, c.clientWidth / c.clientHeight, 0.1, 30000);
    renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true, alpha: false });
    renderer.setSize(c.clientWidth, c.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
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
    // Soft, even ambient like a studio
    scene.add(new THREE.AmbientLight(0xffffff, 0.9));

    // Hemisphere - natural sky/ground
    const hemi = new THREE.HemisphereLight(0xffffff, 0xe0e0e0, 0.6);
    hemi.position.set(0, 1500, 0);
    scene.add(hemi);

    // Main key light - very soft shadows
    const key = new THREE.DirectionalLight(0xffffff, 0.55);
    key.position.set(800, 2500, 1000);
    key.castShadow = true;
    key.shadow.mapSize.width = 4096;
    key.shadow.mapSize.height = 4096;
    key.shadow.camera.left = -3000;
    key.shadow.camera.right = 3000;
    key.shadow.camera.top = 3000;
    key.shadow.camera.bottom = -3000;
    key.shadow.camera.near = 100;
    key.shadow.camera.far = 6000;
    key.shadow.bias = -0.0003;
    key.shadow.radius = 12;
    key.shadow.blurSamples = 25;
    scene.add(key);

    // Fill lights - very soft
    const fill1 = new THREE.DirectionalLight(0xffffff, 0.25);
    fill1.position.set(-1000, 1500, -500);
    scene.add(fill1);

    const fill2 = new THREE.DirectionalLight(0xffffff, 0.2);
    fill2.position.set(500, 1000, -800);
    scene.add(fill2);

    // Front rim light for depth
    const rim = new THREE.DirectionalLight(0xffffff, 0.15);
    rim.position.set(0, 300, 2000);
    scene.add(rim);
}

function buildScene() {
    const cx = data.image_width / 2;
    const cy = data.image_height / 2;
    const sizeX = data.image_width;
    const sizeZ = data.image_height;
    const WALL_HEIGHT = 70;

    const elements = data.elements || [];
    const extWallElement = elements.find(e => e.type === 'extwall' && e.points && e.points.length >= 3);

    // === FLOOR with VERY SUBTLE grid (Synchrony style) ===
    const floorCanvas = document.createElement('canvas');
    floorCanvas.width = 2048;
    floorCanvas.height = 2048;
    const fctx = floorCanvas.getContext('2d');

    // Soft gradient base - very light grey
    const gradient = fctx.createRadialGradient(1024, 1024, 100, 1024, 1024, 1400);
    gradient.addColorStop(0, '#f4f4f6');
    gradient.addColorStop(1, '#e6e6e9');
    fctx.fillStyle = gradient;
    fctx.fillRect(0, 0, 2048, 2048);

    // Very subtle grid lines
    fctx.strokeStyle = '#d8d8dc';
    fctx.lineWidth = 1;
    const tileSize = 100;
    for (let i = 0; i <= 2048; i += tileSize) {
        fctx.beginPath();
        fctx.moveTo(i, 0); fctx.lineTo(i, 2048);
        fctx.stroke();
        fctx.beginPath();
        fctx.moveTo(0, i); fctx.lineTo(2048, i);
        fctx.stroke();
    }

    // Even more subtle minor lines
    fctx.strokeStyle = '#e8e8ec';
    fctx.lineWidth = 0.5;
    const minorSize = 25;
    for (let i = 0; i <= 2048; i += minorSize) {
        if (i % tileSize === 0) continue;
        fctx.beginPath();
        fctx.moveTo(i, 0); fctx.lineTo(i, 2048);
        fctx.stroke();
        fctx.beginPath();
        fctx.moveTo(0, i); fctx.lineTo(2048, i);
        fctx.stroke();
    }

    const floorTex = new THREE.CanvasTexture(floorCanvas);
    floorTex.wrapS = THREE.RepeatWrapping;
    floorTex.wrapT = THREE.RepeatWrapping;
    floorTex.repeat.set(sizeX / 200, sizeZ / 200);
    floorTex.anisotropy = 16;
    floorTex.encoding = THREE.sRGBEncoding;

    const floorMat = new THREE.MeshStandardMaterial({
        map: floorTex,
        roughness: 0.95,
        metalness: 0.0
    });

    // Build floor from exterior wall polygon
    if (extWallElement) {
        const shape = new THREE.Shape();
        const pts = extWallElement.points;
        shape.moveTo(pts[0].x - cx, pts[0].y - cy);
        for (let i = 1; i < pts.length; i++) {
            shape.lineTo(pts[i].x - cx, pts[i].y - cy);
        }
        shape.lineTo(pts[0].x - cx, pts[0].y - cy);

        const floorGeo = new THREE.ShapeGeometry(shape);
        const floor = new THREE.Mesh(floorGeo, floorMat);
        floor.rotation.x = -Math.PI / 2;
        floor.position.y = 0.5;
        floor.receiveShadow = true;
        scene.add(floor);
    } else {
        const floorGeo = new THREE.PlaneGeometry(sizeX, sizeZ);
        const floor = new THREE.Mesh(floorGeo, floorMat);
        floor.rotation.x = -Math.PI / 2;
        floor.position.y = 0.5;
        floor.receiveShadow = true;
        scene.add(floor);
    }

    // === WALLS - clean white with soft edges ===
    const extMat = new THREE.MeshStandardMaterial({
        color: 0xfafafa,
        roughness: 0.9,
        metalness: 0.0
    });
    const intMat = new THREE.MeshStandardMaterial({
        color: 0xf5f5f7,
        roughness: 0.9,
        metalness: 0.0
    });

    // Top cap material (darker edge like in Synchrony)
    const wallTopMat = new THREE.MeshStandardMaterial({
        color: 0x9a9a9e,
        roughness: 0.85,
        metalness: 0.05
    });

    elements.forEach(el => {
        if (el.type === 'extwall' && el.points && el.points.length >= 2) {
            const pts = el.points;
            for (let i = 0; i < pts.length - 1; i++) {
                buildWallSynchrony(
                    [pts[i].x - cx, pts[i].y - cy],
                    [pts[i+1].x - cx, pts[i+1].y - cy],
                    WALL_HEIGHT, 12, extMat, wallTopMat
                );
            }
            if (pts.length >= 3) {
                buildWallSynchrony(
                    [pts[pts.length-1].x - cx, pts[pts.length-1].y - cy],
                    [pts[0].x - cx, pts[0].y - cy],
                    WALL_HEIGHT, 12, extMat, wallTopMat
                );
            }
        }
        if (el.type === 'intwall' && el.points && el.points.length >= 2) {
            const pts = el.points;
            for (let i = 0; i < pts.length - 1; i++) {
                buildWallSynchrony(
                    [pts[i].x - cx, pts[i].y - cy],
                    [pts[i+1].x - cx, pts[i+1].y - cy],
                    WALL_HEIGHT * 0.95, 7, intMat, wallTopMat
                );
            }
        }
    });

    // === DUCTS - clean white sheet metal ===
    const ductMat = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        roughness: 0.4,
        metalness: 0.2
    });
    const ductEdgeMat = new THREE.MeshStandardMaterial({
        color: 0xc0c0c4,
        roughness: 0.5,
        metalness: 0.3
    });

    elements.forEach(el => {
        if (el.type === 'duct' && el.points && el.points.length >= 2) {
            for (let i = 0; i < el.points.length - 1; i++) {
                buildDuctSynchrony(
                    [el.points[i].x - cx, el.points[i].y - cy],
                    [el.points[i+1].x - cx, el.points[i+1].y - cy],
                    WALL_HEIGHT - 10, ductMat, ductEdgeMat
                );
            }
        }
    });

    // === VAVs - blue cubes with white edges (Synchrony style) ===
    const vavMat = new THREE.MeshStandardMaterial({
        color: 0x1e3a8a,
        roughness: 0.5,
        metalness: 0.3,
        emissive: 0x1e40af,
        emissiveIntensity: 0.08
    });

    elements.forEach(el => {
        if (el.type === 'vav') {
            const grp = new THREE.Group();

            // Main cube
            const geo = new THREE.BoxGeometry(28, 26, 28);
            const m = new THREE.Mesh(geo, vavMat);
            m.castShadow = true;
            m.receiveShadow = true;
            grp.add(m);

            // White top accent
            const topGeo = new THREE.BoxGeometry(28.5, 3, 28.5);
            const topMat = new THREE.MeshStandardMaterial({
                color: 0xffffff, roughness: 0.6
            });
            const topMesh = new THREE.Mesh(topGeo, topMat);
            topMesh.position.y = 14.5;
            grp.add(topMesh);

            grp.position.set(el.x - cx, WALL_HEIGHT - 20, el.y - cy);
            scene.add(grp);
        }
    });

    // === AHU - green with details ===
    elements.forEach(el => {
        if (el.type === 'ahu') {
            const grp = new THREE.Group();

            const mat = new THREE.MeshStandardMaterial({
                color: 0x16a34a,
                roughness: 0.5,
                metalness: 0.3,
                emissive: 0x15803d,
                emissiveIntensity: 0.08
            });
            const geo = new THREE.BoxGeometry(60, 50, 50);
            const m = new THREE.Mesh(geo, mat);
            m.castShadow = true;
            m.receiveShadow = true;
            grp.add(m);

            // White top accent
            const topGeo = new THREE.BoxGeometry(61, 3, 51);
            const topMat = new THREE.MeshStandardMaterial({
                color: 0xffffff, roughness: 0.6
            });
            const topMesh = new THREE.Mesh(topGeo, topMat);
            topMesh.position.y = 26.5;
            grp.add(topMesh);

            grp.position.set(el.x - cx, WALL_HEIGHT - 25, el.y - cy);
            scene.add(grp);
        }
    });

    // === DIFFUSERS - small white squares ===
    const diffMat = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        roughness: 0.5,
        metalness: 0.15
    });
    const diffEdgeMat = new THREE.MeshStandardMaterial({
        color: 0xa0a0a4,
        roughness: 0.7
    });

    elements.forEach(el => {
        if (el.type === 'diffuser') {
            const grp = new THREE.Group();

            const geo = new THREE.BoxGeometry(12, 2.5, 12);
            const m = new THREE.Mesh(geo, diffMat);
            m.castShadow = true;
            grp.add(m);

            // Frame around
            const edgeGeo = new THREE.BoxGeometry(13, 1.5, 13);
            const edgeMesh = new THREE.Mesh(edgeGeo, diffEdgeMat);
            edgeMesh.position.y = -0.5;
            grp.add(edgeMesh);

            grp.position.set(el.x - cx, WALL_HEIGHT - 5, el.y - cy);
            scene.add(grp);
        }
    });

    // === Camera - Synchrony-style angle ===
    const maxSize = Math.max(sizeX, sizeZ);
    const dist = maxSize * 1.1;
    camera.position.set(0, dist * 0.75, dist * 0.65);
    initPos = camera.position.clone();
    initTarget = new THREE.Vector3(0, WALL_HEIGHT * 0.25, 0);
    controls.target.copy(initTarget);
    controls.update();
}

function buildWallSynchrony(p1, p2, height, thickness, sideMat, topMat) {
    const dx = p2[0] - p1[0], dz = p2[1] - p1[1];
    const length = Math.sqrt(dx * dx + dz * dz);
    if (length < 5) return;
    const angle = Math.atan2(dz, dx);
    const grp = new THREE.Group();

    // Main wall body
    const geo = new THREE.BoxGeometry(length, height, thickness);
    const m = new THREE.Mesh(geo, sideMat);
    m.position.y = height / 2;
    m.castShadow = true;
    m.receiveShadow = true;
    grp.add(m);

    // Darker top cap (Synchrony style)
    const topGeo = new THREE.BoxGeometry(length + 0.5, 2, thickness + 0.5);
    const topMesh = new THREE.Mesh(topGeo, topMat);
    topMesh.position.y = height + 1;
    topMesh.castShadow = true;
    grp.add(topMesh);

    grp.position.set((p1[0] + p2[0]) / 2, 0, (p1[1] + p2[1]) / 2);
    grp.rotation.y = -angle;
    scene.add(grp);
}

function buildDuctSynchrony(p1, p2, yPos, mainMat, edgeMat) {
    const dx = p2[0] - p1[0], dz = p2[1] - p1[1];
    const length = Math.sqrt(dx * dx + dz * dz);
    if (length < 5) return;
    const angle = Math.atan2(dz, dx);

    const grp = new THREE.Group();

    // Main duct body
    const geo = new THREE.BoxGeometry(length, 8, 14);
    const m = new THREE.Mesh(geo, mainMat);
    m.castShadow = true;
    m.receiveShadow = true;
    grp.add(m);

    // Side edge for definition
    const edgeGeo1 = new THREE.BoxGeometry(length, 1, 1);
    const edge1 = new THREE.Mesh(edgeGeo1, edgeMat);
    edge1.position.set(0, 4, 7);
    grp.add(edge1);

    const edge2 = new THREE.Mesh(edgeGeo1, edgeMat);
    edge2.position.set(0, 4, -7);
    grp.add(edge2);

    grp.position.set((p1[0] + p2[0]) / 2, yPos, (p1[1] + p2[1]) / 2);
    grp.rotation.y = -angle;
    scene.add(grp);
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
    camera.position.set(0, max * 1.5, 0.1);
    controls.target.set(0, 0, 0);
    controls.update();
}

function synchView() {
    const max = Math.max(data.image_width, data.image_height);
    camera.position.set(0, max * 0.75, max * 0.65);
    controls.target.set(0, 20, 0);
    controls.update();
}

function screenshot() {
    renderer.render(scene, camera);
    const url = renderer.domElement.toDataURL('image/png');
    const link = document.createElement('a');
    link.download = 'bas_synchrony_3d.png';
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
        return "<h2 style='color:white;background:#0d0f14;padding:30px;'>No plan uploaded. <a href='/' style='color:#2d89ef'>Upload one</a></h2>"
    return render_template_string(EDITOR_PAGE, image_b64=image_to_base64(UPLOAD_IMAGE_PATH))


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


@app.route("/render3d")
def render3d():
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
    n_ext = sum(1 for e in elements if e.get("type") == "extwall")
    n_int = sum(1 for e in elements if e.get("type") == "intwall")

    return render_template_string(
        RESULT_PAGE,
        detection_json=json.dumps(detection),
        n_vavs=n_vavs, n_ahus=n_ahus, n_ducts=n_ducts,
        n_diffs=n_diffs, n_ext=n_ext, n_int=n_int
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
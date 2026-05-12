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
<title>BAS Generator v16 - CAD Editor</title>
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
<div class="logo">&#128221;</div>
<h1>BAS Generator v16 <span class="badge">CAD MODE</span></h1>
<p class="sub">Click-to-draw polyline walls + click-to-place equipment</p>
<div style="text-align: left; margin-bottom: 24px;">
<div class="feature"><div class="color-dot" style="background:#9333ea"></div> POLYLINE walls (click corners, double-click to finish)</div>
<div class="feature"><div class="color-dot" style="background:#1e40af"></div> Click to place VAVs</div>
<div class="feature"><div class="color-dot" style="background:#16a34a"></div> Click to place AHU</div>
<div class="feature"><div class="color-dot" style="background:#dc2626"></div> Click 2 points for duct lines</div>
<div class="feature"><div class="color-dot" style="background:#888"></div> Edit/move anything after drawing</div>
</div>
<form action="/upload" method="post" enctype="multipart/form-data">
<div class="zone">
<input type="file" name="file" accept="image/png,image/jpeg,application/pdf" required>
<span class="lbl">Upload original mechanical plan</span>
</div>
<button class="btn" type="submit">Open CAD Editor</button>
</form>
<div class="footer">Made by Paolo V.</div>
</div>
</body>
</html>'''


EDITOR_PAGE = '''<!DOCTYPE html>
<html>
<head>
<title>CAD Editor</title>
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
.hint { font-size: 11px; color: #888; padding: 4px 8px; background: rgba(255,255,255,0.05); border-radius: 6px; }
.status { padding: 4px 12px; background: #1e2233; border-radius: 6px; font-size: 11px; color: #aab0c4; min-width: 200px; text-align: center; }
.cursor-cross { cursor: crosshair; }
.cursor-move { cursor: move; }
.cursor-default { cursor: default; }
</style>
</head>
<body>
<div class="topbar">
<h1>CAD Editor &mdash; Click points to draw</h1>
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
<div class="color-swatch" style="background:#dc2626"></div> Duct Line
</button>

<div class="divider"></div>

<button class="tool-btn" data-tool="vav" onclick="selectTool(this)">
<div class="color-swatch" style="background:#1e40af"></div> VAV
</button>
<button class="tool-btn" data-tool="ahu" onclick="selectTool(this)">
<div class="color-swatch" style="background:#16a34a"></div> AHU
</button>
<button class="tool-btn" data-tool="diffuser" onclick="selectTool(this)">
<div class="color-swatch" style="background:#fff"></div> Diffuser
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
<div style="color:white;font-size:14px;">Building 3D model...</div>
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
    duct: '#dc2626',
    vav: '#1e40af',
    ahu: '#16a34a',
    diffuser: '#ffffff'
};

const STATUS_TEXTS = {
    extwall: 'Click corners of the BUILDING PERIMETER. Double-click to close polygon.',
    intwall: 'Click corners of an INTERIOR WALL. Double-click to finish.',
    duct: 'Click TWO points for a duct line. Click again for next duct.',
    vav: 'Click to place a VAV (blue cube).',
    ahu: 'Click to place the AHU (green box).',
    diffuser: 'Click to place a diffuser (white square).',
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

    // Reset polyline if switching tools
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

    if (currentTool === 'move') {
        return;
    }

    // Single-point tools
    if (currentTool === 'vav' || currentTool === 'ahu' || currentTool === 'diffuser') {
        elements.push({ type: currentTool, x: pos.x, y: pos.y });
        saveState();
        redraw();
        return;
    }

    // Polyline tools (extwall, intwall)
    if (currentTool === 'extwall' || currentTool === 'intwall') {
        if (!currentPolyline) {
            currentPolyline = { type: currentTool, points: [{ x: pos.x, y: pos.y }] };
        } else {
            currentPolyline.points.push({ x: pos.x, y: pos.y });
        }
        redraw();
        return;
    }

    // Duct line - 2 clicks make a line
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
        // For exterior walls, close the polygon
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
        redraw();
        return;
    }

    if (currentPolyline) {
        redraw();
    }
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

// Cancel current polyline with Escape
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
            // Check polyline
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

    // Draw all completed elements
    for (const el of elements) {
        drawElement(el);
    }

    // Draw current polyline being created
    if (currentPolyline) {
        drawElement(currentPolyline, true);
        // Draw rubber-band line to mouse
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
        drawCtx.lineWidth = 1;
        drawCtx.fillRect(el.x - 8, el.y - 8, 16, 16);
        drawCtx.strokeRect(el.x - 8, el.y - 8, 16, 16);
        return;
    }

    // Polyline
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
    if (el.closed) {
        drawCtx.closePath();
    }
    drawCtx.stroke();

    // Draw vertices
    drawCtx.fillStyle = color;
    for (const p of el.points) {
        drawCtx.beginPath();
        drawCtx.arc(p.x, p.y, 5, 0, Math.PI * 2);
        drawCtx.fill();
    }

    // First point gets ring if in progress
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
    // Add current polyline if it has at least 2 points
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
<div class="stat">Diffusers: <b>{{ n_diffs }}</b></div>
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
    scene.add(new THREE.DirectionalLight(0xffffff, 0.35));
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

    // Use exterior wall as floor shape if available
    const extWalls = (data.elements || []).filter(e => e.type === 'extwall');
    if (extWalls.length > 0 && extWalls[0].points && extWalls[0].points.length >= 3) {
        const shape = new THREE.Shape();
        const pts = extWalls[0].points;
        shape.moveTo(pts[0].x - cx, pts[0].y - cy);
        for (let i = 1; i < pts.length; i++) {
            shape.lineTo(pts[i].x - cx, pts[i].y - cy);
        }
        shape.lineTo(pts[0].x - cx, pts[0].y - cy);
        const floorGeo = new THREE.ShapeGeometry(shape);
        const floor = new THREE.Mesh(floorGeo, new THREE.MeshStandardMaterial({ map: tex, roughness: 0.85 }));
        floor.rotation.x = -Math.PI / 2;
        floor.position.y = 0.5;
        floor.receiveShadow = true;
        scene.add(floor);
    } else {
        const floorGeo = new THREE.PlaneGeometry(sizeX, sizeZ);
        const floor = new THREE.Mesh(floorGeo, new THREE.MeshStandardMaterial({ map: tex, roughness: 0.85 }));
        floor.rotation.x = -Math.PI / 2;
        floor.position.y = 0.5;
        floor.receiveShadow = true;
        scene.add(floor);
    }

    // Build walls from polylines
    const extMat = new THREE.MeshStandardMaterial({ color: 0x6a6e76, roughness: 0.9 });
    const intMat = new THREE.MeshStandardMaterial({ color: 0x848890, roughness: 0.88 });

    (data.elements || []).forEach(el => {
        if (el.type === 'extwall' && el.points && el.points.length >= 2) {
            const pts = el.points;
            for (let i = 0; i < pts.length - 1; i++) {
                buildWall(
                    [pts[i].x - cx, pts[i].y - cy],
                    [pts[i+1].x - cx, pts[i+1].y - cy],
                    WALL_HEIGHT, 12, extMat
                );
            }
            // Close polygon
            if (pts.length >= 3) {
                buildWall(
                    [pts[pts.length-1].x - cx, pts[pts.length-1].y - cy],
                    [pts[0].x - cx, pts[0].y - cy],
                    WALL_HEIGHT, 12, extMat
                );
            }
        }
        if (el.type === 'intwall' && el.points && el.points.length >= 2) {
            const pts = el.points;
            for (let i = 0; i < pts.length - 1; i++) {
                buildWall(
                    [pts[i].x - cx, pts[i].y - cy],
                    [pts[i+1].x - cx, pts[i+1].y - cy],
                    WALL_HEIGHT * 0.92, 7, intMat
                );
            }
        }
    });

    // Ducts
    const ductMat = new THREE.MeshStandardMaterial({ color: 0xf8f8f8, roughness: 0.35, metalness: 0.55 });
    (data.elements || []).forEach(el => {
        if (el.type === 'duct' && el.points && el.points.length >= 2) {
            for (let i = 0; i < el.points.length - 1; i++) {
                buildDuct(
                    [el.points[i].x - cx, el.points[i].y - cy],
                    [el.points[i+1].x - cx, el.points[i+1].y - cy],
                    WALL_HEIGHT - 12, ductMat
                );
            }
        }
    });

    // VAVs
    const vavMat = new THREE.MeshStandardMaterial({ color: 0x1e40af, roughness: 0.45, metalness: 0.35, emissive: 0x1e3a8a, emissiveIntensity: 0.18 });
    (data.elements || []).forEach(el => {
        if (el.type === 'vav') {
            const geo = new THREE.BoxGeometry(28, 24, 28);
            const m = new THREE.Mesh(geo, vavMat);
            m.position.set(el.x - cx, WALL_HEIGHT - 22, el.y - cy);
            m.castShadow = true;
            scene.add(m);
        }
    });

    // AHUs
    (data.elements || []).forEach(el => {
        if (el.type === 'ahu') {
            const mat = new THREE.MeshStandardMaterial({ color: 0x16a34a, roughness: 0.5, metalness: 0.35, emissive: 0x14532d, emissiveIntensity: 0.18 });
            const geo = new THREE.BoxGeometry(60, 50, 45);
            const m = new THREE.Mesh(geo, mat);
            m.position.set(el.x - cx, WALL_HEIGHT - 30, el.y - cy);
            m.castShadow = true;
            scene.add(m);
        }
    });

    // Diffusers
    const diffMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.4, metalness: 0.2 });
    (data.elements || []).forEach(el => {
        if (el.type === 'diffuser') {
            const geo = new THREE.BoxGeometry(14, 3, 14);
            const m = new THREE.Mesh(geo, diffMat);
            m.position.set(el.x - cx, WALL_HEIGHT - 5, el.y - cy);
            m.castShadow = true;
            scene.add(m);
        }
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

function buildDuct(p1, p2, yPos, mat) {
    const dx = p2[0] - p1[0], dz = p2[1] - p1[1];
    const length = Math.sqrt(dx * dx + dz * dz);
    if (length < 5) return;
    const angle = Math.atan2(dz, dx);
    const geo = new THREE.BoxGeometry(length, 14, 18);
    const m = new THREE.Mesh(geo, mat);
    m.position.set((p1[0] + p2[0]) / 2, yPos, (p1[1] + p2[1]) / 2);
    m.rotation.y = -angle;
    m.castShadow = true;
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

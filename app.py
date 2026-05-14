from flask import Flask, request, render_template_string, jsonify, redirect, make_response
import os
import base64
import json
import cv2
import fitz

app = Flask(__name__)

APP_PASSWORD = os.environ.get("APP_PASSWORD", "1234")
SECRET_KEY = os.environ.get("SECRET_KEY", "change_this_secret")
app.secret_key = SECRET_KEY

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


LOGIN_PAGE = '''<!DOCTYPE html>
<html>
<head>
<title>BAS Generator Login</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background:#0d0f14; color:white; font-family:'Segoe UI', Arial, sans-serif; min-height:100vh; display:flex; align-items:center; justify-content:center; }
.card { background:#181b24; border:1px solid #2a3050; border-radius:24px; padding:45px; width:420px; text-align:center; box-shadow:0 0 60px rgba(0,0,0,0.65); }
.logo { font-size:48px; margin-bottom:12px; }
h1 { font-size:26px; margin-bottom:6px; background:linear-gradient(135deg,#2d89ef,#b388ff); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.sub { color:#8b93ad; font-size:14px; margin-bottom:24px; }
input { width:100%; padding:15px; border-radius:12px; border:1px solid #2a3050; background:#10131a; color:white; font-size:16px; outline:none; }
input:focus { border-color:#2d89ef; }
button { width:100%; padding:15px; border:none; border-radius:12px; margin-top:18px; background:linear-gradient(135deg,#1a6fd4,#2d89ef); color:white; font-size:16px; font-weight:700; cursor:pointer; }
.error { margin-top:14px; color:#ff6b6b; font-size:13px; }
.footer { color:#3a4060; font-size:11px; margin-top:24px; }
</style>
</head>
<body>
<div class="card">
<div class="logo">&#128274;</div>
<h1>BAS Generator v21</h1>
<p class="sub">Private Access</p>
<form method="POST" action="/login">
<input type="password" name="password" placeholder="Enter password" required autofocus>
<button type="submit">Login</button>
</form>
{% if error %}
<div class="error">Invalid password. Try again.</div>
{% endif %}
<div class="footer">Made by Paolo Vasquez</div>
</div>
</body>
</html>'''


HOME_PAGE = '''<!DOCTYPE html>
<html>
<head>
<title>BAS Generator v21 - Synchrony Pro</title>
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
<h1>BAS Generator v21 <span class="badge">SYNCHRONY PRO</span></h1>
<p class="sub">Professional architectural isometric BAS graphics</p>
<div style="text-align: left; margin-bottom: 24px;">
<div class="feature"><div class="color-dot" style="background:#9333ea"></div> Click corners for walls (always straight)</div>
<div class="feature"><div class="color-dot" style="background:#1e40af"></div> Click to place VAVs (beveled blue cubes)</div>
<div class="feature"><div class="color-dot" style="background:#16a34a"></div> Click to place AHU (beveled green box)</div>
<div class="feature"><div class="color-dot" style="background:#fff;border:1px solid #888"></div> Two clicks for trunk-branch duct lines</div>
<div class="feature"><div class="color-dot" style="background:#444"></div> Synchrony-style 2D isometric output</div>
</div>
<form action="/upload" method="post" enctype="multipart/form-data">
<div class="zone">
<input type="file" name="file" accept="image/png,image/jpeg,application/pdf" required>
<span class="lbl">Upload original mechanical plan</span>
</div>
<button class="btn" type="submit">Open CAD Editor</button>
</form>
<div class="footer">Made by Paolo V. R.</div>
</div>
</body>
</html>'''


EDITOR_PAGE = '''<!DOCTYPE html>
<html>
<head>
<title>CAD Editor v21</title>
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
.btn-red { background: #dc2626; color: white; }
.btn-gray { background: #333; color: white; }
.spinner { display: inline-block; width: 20px; height: 20px; border: 3px solid #fff; border-top-color: transparent; border-radius: 50%; animation: spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.loading-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.85); display: none; align-items: center; justify-content: center; z-index: 100; flex-direction: column; gap: 16px; }
.loading-overlay.active { display: flex; }
.status { padding: 4px 12px; background: #1e2233; border-radius: 6px; font-size: 11px; color: #aab0c4; min-width: 200px; text-align: center; }
.cursor-cross { cursor: crosshair; }
.cursor-move { cursor: move; }
</style>
</head>
<body>
<div class="topbar">
<h1>CAD Editor v21 &mdash; Click to draw, double-click to finish</h1>
<div style="display: flex; gap: 6px;">
<button onclick="undo()" class="action-btn btn-gray">&#8617; Undo</button>
<button onclick="clearAll()" class="action-btn btn-red">Clear</button>
<button onclick="generate()" class="action-btn btn-green">Generate Graphic &rarr;</button>
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
<div style="color:white;font-size:14px;">Building Synchrony-style SVG...</div>
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
    duct: '#ffffff',
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
        drawCtx.fillRect(el.x - 6, el.y - 6, 12, 12);
        drawCtx.strokeRect(el.x - 6, el.y - 6, 12, 12);
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

async function generate() {
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
            window.location.href = '/result';
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
<title>BAS Graphic Result v21</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d0f14; color: white; font-family: 'Segoe UI', Arial, sans-serif; padding: 12px; }
h1 { text-align: center; font-size: 22px; margin-bottom: 4px; background: linear-gradient(135deg, #2d89ef, #b388ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.sub { text-align: center; color: #6878a8; font-size: 12px; margin-bottom: 10px; }
.stats { display: flex; justify-content: center; gap: 10px; margin: 8px 0 12px; flex-wrap: wrap; }
.stat { background: #1e2233; padding: 5px 12px; border-radius: 8px; font-size: 12px; color: #aab0c4; border: 1px solid #2a3050; }
.stat b { color: #fff; }
.viewer-svg { width: 100%; height: 78vh; background: #000; border-radius: 12px; border: 1px solid #2a3050; overflow: auto; display: flex; align-items: center; justify-content: center; padding: 20px; }
.viewer-svg svg { max-width: 100%; height: auto; }
.actions { text-align: center; margin-top: 12px; display: flex; justify-content: center; gap: 8px; flex-wrap: wrap; }
.btn { padding: 10px 18px; border: none; border-radius: 10px; font-size: 13px; font-weight: 700; cursor: pointer; text-decoration: none; display: inline-block; }
.btn-blue { background: #1a6fd4; color: white; }
.btn-green { background: #1a9e4a; color: white; }
.btn-gray { background: #252a38; color: #aab0c4; }
.footer { text-align: center; color: #3a4060; font-size: 11px; margin-top: 10px; }
</style>
</head>
<body>
<h1>Synchrony-Style BAS Graphic v21</h1>
<p class="sub">SVG Isometric Render - Ready for Tracer Synchrony / Niagara</p>
<div class="stats">
<div class="stat">VAVs: <b>{{ n_vavs }}</b></div>
<div class="stat">AHUs: <b>{{ n_ahus }}</b></div>
<div class="stat">Ducts: <b>{{ n_ducts }}</b></div>
<div class="stat">Diffusers: <b>{{ n_diffs }}</b></div>
<div class="stat">Ext Walls: <b>{{ n_ext }}</b></div>
<div class="stat">Int Walls: <b>{{ n_int }}</b></div>
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

// Isometric projection: 30 degrees on x and y axes
const ISO_ANGLE = Math.PI / 6;
const COS30 = Math.cos(ISO_ANGLE);
const SIN30 = Math.sin(ISO_ANGLE);

function isoProject(x, y, z) {
    const sx = (x - y) * COS30;
    const sy = (x + y) * SIN30 - z;
    return [sx, sy];
}

function generateSVG() {
    const elements = data.elements || [];
    const extWall = elements.find(e => e.type === 'extwall' && e.points && e.points.length >= 3);

    let minX = 0, maxX = data.image_width, minY = 0, maxY = data.image_height;
    if (extWall) {
        const xs = extWall.points.map(p => p.x);
        const ys = extWall.points.map(p => p.y);
        minX = Math.min(...xs);
        maxX = Math.max(...xs);
        minY = Math.min(...ys);
        maxY = Math.max(...ys);
    }
    const bcx = (minX + maxX) / 2;
    const bcy = (minY + maxY) / 2;

    const WALL_HEIGHT = 55;

    function toLocal(p) {
        return { x: p.x - bcx, y: p.y - bcy };
    }

    function proj(x, y, z = 0) {
        return isoProject(x, y, z);
    }

    // Calculate SVG canvas bounds
    let svgMinX = 0, svgMaxX = 0, svgMinY = 0, svgMaxY = 0;
    const corners = [
        toLocal({ x: minX, y: minY }),
        toLocal({ x: maxX, y: minY }),
        toLocal({ x: maxX, y: maxY }),
        toLocal({ x: minX, y: maxY })
    ];
    for (const c of corners) {
        for (const z of [0, WALL_HEIGHT + 20]) {
            const [sx, sy] = proj(c.x, c.y, z);
            svgMinX = Math.min(svgMinX, sx);
            svgMaxX = Math.max(svgMaxX, sx);
            svgMinY = Math.min(svgMinY, sy);
            svgMaxY = Math.max(svgMaxY, sy);
        }
    }

    const padding = 80;
    const svgW = svgMaxX - svgMinX + padding * 2;
    const svgH = svgMaxY - svgMinY + padding * 2;
    const offsetX = -svgMinX + padding;
    const offsetY = -svgMinY + padding;

    function projSVG(x, y, z = 0) {
        const [sx, sy] = proj(x, y, z);
        return [sx + offsetX, sy + offsetY];
    }

    let svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${svgW} ${svgH}" width="${svgW}" height="${svgH}">`;
    svg += `<rect width="${svgW}" height="${svgH}" fill="#0a0a0d"/>`;

    // === DEFS: gradients, patterns, filters ===
    svg += `<defs>`;

    // Subtle isometric floor tile - diamond shape with soft gray
    const tw = 24 * COS30 * 2;
    const th = 24 * SIN30 * 2;
    svg += `<pattern id="isoFloor" width="${tw}" height="${th}" patternUnits="userSpaceOnUse">`;
    svg += `<polygon points="${tw/2},0 ${tw},${th/2} ${tw/2},${th} 0,${th/2}" fill="#d8d8dc" stroke="#cdcdd1" stroke-width="0.5"/>`;
    svg += `</pattern>`;

    // Wall side gradient (lighter at top, darker at bottom)
    svg += `<linearGradient id="extWallSide" x1="0%" y1="0%" x2="0%" y2="100%">`;
    svg += `<stop offset="0%" stop-color="#c4c4ca"/>`;
    svg += `<stop offset="100%" stop-color="#8e8e94"/>`;
    svg += `</linearGradient>`;

    svg += `<linearGradient id="extWallTop" x1="0%" y1="0%" x2="100%" y2="100%">`;
    svg += `<stop offset="0%" stop-color="#e8e8ec"/>`;
    svg += `<stop offset="100%" stop-color="#b8b8bc"/>`;
    svg += `</linearGradient>`;

    svg += `<linearGradient id="extWallEnd" x1="0%" y1="0%" x2="100%" y2="0%">`;
    svg += `<stop offset="0%" stop-color="#bababf"/>`;
    svg += `<stop offset="100%" stop-color="#9b9ba0"/>`;
    svg += `</linearGradient>`;

    svg += `<linearGradient id="intWallSide" x1="0%" y1="0%" x2="0%" y2="100%">`;
    svg += `<stop offset="0%" stop-color="#d0d0d4"/>`;
    svg += `<stop offset="100%" stop-color="#a4a4a8"/>`;
    svg += `</linearGradient>`;

    svg += `<linearGradient id="intWallTop" x1="0%" y1="0%" x2="100%" y2="100%">`;
    svg += `<stop offset="0%" stop-color="#ebebef"/>`;
    svg += `<stop offset="100%" stop-color="#c4c4c8"/>`;
    svg += `</linearGradient>`;

    // Duct gradients
    svg += `<linearGradient id="ductTop" x1="0%" y1="0%" x2="0%" y2="100%">`;
    svg += `<stop offset="0%" stop-color="#ffffff"/>`;
    svg += `<stop offset="100%" stop-color="#ececf0"/>`;
    svg += `</linearGradient>`;

    svg += `<linearGradient id="ductFront" x1="0%" y1="0%" x2="0%" y2="100%">`;
    svg += `<stop offset="0%" stop-color="#dadade"/>`;
    svg += `<stop offset="100%" stop-color="#b8b8bc"/>`;
    svg += `</linearGradient>`;

    // VAV gradients
    svg += `<linearGradient id="vavTop" x1="0%" y1="0%" x2="100%" y2="100%">`;
    svg += `<stop offset="0%" stop-color="#3b6df0"/>`;
    svg += `<stop offset="100%" stop-color="#1e40af"/>`;
    svg += `</linearGradient>`;

    svg += `<linearGradient id="vavFront" x1="0%" y1="0%" x2="0%" y2="100%">`;
    svg += `<stop offset="0%" stop-color="#1e3a8a"/>`;
    svg += `<stop offset="100%" stop-color="#152a6e"/>`;
    svg += `</linearGradient>`;

    svg += `<linearGradient id="vavRight" x1="0%" y1="0%" x2="100%" y2="0%">`;
    svg += `<stop offset="0%" stop-color="#1e40af"/>`;
    svg += `<stop offset="100%" stop-color="#0c1f5c"/>`;
    svg += `</linearGradient>`;

    // AHU gradients
    svg += `<linearGradient id="ahuTop" x1="0%" y1="0%" x2="100%" y2="100%">`;
    svg += `<stop offset="0%" stop-color="#34d365"/>`;
    svg += `<stop offset="100%" stop-color="#16a34a"/>`;
    svg += `</linearGradient>`;

    svg += `<linearGradient id="ahuFront" x1="0%" y1="0%" x2="0%" y2="100%">`;
    svg += `<stop offset="0%" stop-color="#15803d"/>`;
    svg += `<stop offset="100%" stop-color="#0a5828"/>`;
    svg += `</linearGradient>`;

    svg += `<linearGradient id="ahuRight" x1="0%" y1="0%" x2="100%" y2="0%">`;
    svg += `<stop offset="0%" stop-color="#16a34a"/>`;
    svg += `<stop offset="100%" stop-color="#0c5a26"/>`;
    svg += `</linearGradient>`;

    // Soft shadow filter
    svg += `<filter id="softShadow" x="-30%" y="-30%" width="160%" height="160%">`;
    svg += `<feGaussianBlur in="SourceAlpha" stdDeviation="2"/>`;
    svg += `<feOffset dx="1.5" dy="3"/>`;
    svg += `<feComponentTransfer><feFuncA type="linear" slope="0.4"/></feComponentTransfer>`;
    svg += `<feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge>`;
    svg += `</filter>`;

    // Floor inner shadow - subtle ambient occlusion near walls
    svg += `<filter id="floorAO" x="-10%" y="-10%" width="120%" height="120%">`;
    svg += `<feGaussianBlur in="SourceAlpha" stdDeviation="4" result="blur"/>`;
    svg += `<feComposite in="blur" in2="SourceAlpha" operator="arithmetic" k2="-1" k3="1"/>`;
    svg += `</filter>`;

    svg += `</defs>`;

    // === FLOOR with isometric tile pattern ===
    if (extWall) {
        const pts = extWall.points.map(p => toLocal(p));
        let floorPath = '';
        for (let i = 0; i < pts.length; i++) {
            const [sx, sy] = projSVG(pts[i].x, pts[i].y, 0);
            floorPath += (i === 0 ? 'M' : 'L') + sx + ',' + sy + ' ';
        }
        floorPath += 'Z';
        svg += `<path d="${floorPath}" fill="url(#isoFloor)" stroke="#a0a0a4" stroke-width="0.4"/>`;
        // Subtle inner shadow near walls
        svg += `<path d="${floorPath}" fill="none" stroke="#888" stroke-width="1.2" opacity="0.18"/>`;
    }

    // === Helper: draw thick wall with gradient faces and rounded look ===
    function drawThickWall(p1, p2, height, thickness, sideGrad, topGrad, endGrad, stroke) {
        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const len = Math.sqrt(dx * dx + dy * dy);
        if (len < 1) return '';
        const nx = -dy / len * thickness / 2;
        const ny = dx / len * thickness / 2;

        const p1a = { x: p1.x + nx, y: p1.y + ny };
        const p1b = { x: p1.x - nx, y: p1.y - ny };
        const p2a = { x: p2.x + nx, y: p2.y + ny };
        const p2b = { x: p2.x - nx, y: p2.y - ny };

        const [b1ax, b1ay] = projSVG(p1a.x, p1a.y, 0);
        const [b2ax, b2ay] = projSVG(p2a.x, p2a.y, 0);
        const [b1bx, b1by] = projSVG(p1b.x, p1b.y, 0);
        const [b2bx, b2by] = projSVG(p2b.x, p2b.y, 0);

        const [t1ax, t1ay] = projSVG(p1a.x, p1a.y, height);
        const [t2ax, t2ay] = projSVG(p2a.x, p2a.y, height);
        const [t1bx, t1by] = projSVG(p1b.x, p1b.y, height);
        const [t2bx, t2by] = projSVG(p2b.x, p2b.y, height);

        let walls = '';

        // Visible front face
        const front = `M ${b1bx},${b1by} L ${b2bx},${b2by} L ${t2bx},${t2by} L ${t1bx},${t1by} Z`;
        walls += `<path d="${front}" fill="${sideGrad}" stroke="${stroke}" stroke-width="0.5" stroke-linejoin="round"/>`;

        // Top face
        const top = `M ${t1ax},${t1ay} L ${t2ax},${t2ay} L ${t2bx},${t2by} L ${t1bx},${t1by} Z`;
        walls += `<path d="${top}" fill="${topGrad}" stroke="${stroke}" stroke-width="0.5" stroke-linejoin="round"/>`;

        // End cap on starting side
        const endCap = `M ${b1ax},${b1ay} L ${b1bx},${b1by} L ${t1bx},${t1by} L ${t1ax},${t1ay} Z`;
        walls += `<path d="${endCap}" fill="${endGrad}" stroke="${stroke}" stroke-width="0.5" stroke-linejoin="round"/>`;

        return walls;
    }

    // === EXTERIOR WALLS ===
    if (extWall && extWall.points.length >= 2) {
        const pts = extWall.points.map(p => toLocal(p));
        for (let i = 0; i < pts.length - 1; i++) {
            svg += drawThickWall(pts[i], pts[i + 1], WALL_HEIGHT, 14,
                'url(#extWallSide)', 'url(#extWallTop)', 'url(#extWallEnd)', '#5a5d63');
        }
        if (pts.length >= 3) {
            svg += drawThickWall(pts[pts.length - 1], pts[0], WALL_HEIGHT, 14,
                'url(#extWallSide)', 'url(#extWallTop)', 'url(#extWallEnd)', '#5a5d63');
        }
    }

    // === INTERIOR WALLS - thinner ===
    elements.forEach(el => {
        if (el.type === 'intwall' && el.points && el.points.length >= 2) {
            const pts = el.points.map(p => toLocal(p));
            for (let i = 0; i < pts.length - 1; i++) {
                svg += drawThickWall(pts[i], pts[i + 1], WALL_HEIGHT * 0.92, 8,
                    'url(#intWallSide)', 'url(#intWallTop)', 'url(#extWallEnd)', '#6a6d73');
            }
        }
    });

    // === DUCTS with depth and shadow ===
    function drawDuctSegment(p1, p2, zLevel) {
        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const len = Math.sqrt(dx * dx + dy * dy);
        if (len < 1) return '';

        const thickness = 11;
        const nx = -dy / len * thickness / 2;
        const ny = dx / len * thickness / 2;

        const p1a = { x: p1.x + nx, y: p1.y + ny };
        const p1b = { x: p1.x - nx, y: p1.y - ny };
        const p2a = { x: p2.x + nx, y: p2.y + ny };
        const p2b = { x: p2.x - nx, y: p2.y - ny };

        const ductH = 7;

        const [t1ax, t1ay] = projSVG(p1a.x, p1a.y, zLevel + ductH);
        const [t2ax, t2ay] = projSVG(p2a.x, p2a.y, zLevel + ductH);
        const [t1bx, t1by] = projSVG(p1b.x, p1b.y, zLevel + ductH);
        const [t2bx, t2by] = projSVG(p2b.x, p2b.y, zLevel + ductH);
        const [b1bx, b1by] = projSVG(p1b.x, p1b.y, zLevel);
        const [b2bx, b2by] = projSVG(p2b.x, p2b.y, zLevel);
        const [b1ax, b1ay] = projSVG(p1a.x, p1a.y, zLevel);

        let d = '';

        // Top face
        d += `<path d="M ${t1ax},${t1ay} L ${t2ax},${t2ay} L ${t2bx},${t2by} L ${t1bx},${t1by} Z" fill="url(#ductTop)" stroke="#aaaaae" stroke-width="0.4"/>`;
        // Front face
        d += `<path d="M ${b1bx},${b1by} L ${b2bx},${b2by} L ${t2bx},${t2by} L ${t1bx},${t1by} Z" fill="url(#ductFront)" stroke="#aaaaae" stroke-width="0.4"/>`;
        // End cap
        d += `<path d="M ${b1ax},${b1ay} L ${b1bx},${b1by} L ${t1bx},${t1by} L ${t1ax},${t1ay} Z" fill="#c8c8cc" stroke="#aaaaae" stroke-width="0.4"/>`;

        return d;
    }

    // Collect ducts and draw with shadows underneath
    const ductElements = elements.filter(e => e.type === 'duct' && e.points && e.points.length >= 2);

    // First pass: subtle shadow under ducts (on floor)
    ductElements.forEach(el => {
        const pts = el.points.map(p => toLocal(p));
        for (let i = 0; i < pts.length - 1; i++) {
            const dx = pts[i+1].x - pts[i].x;
            const dy = pts[i+1].y - pts[i].y;
            const len = Math.sqrt(dx * dx + dy * dy);
            if (len < 1) continue;
            const nx = -dy / len * 7;
            const ny = dx / len * 7;

            const sa = projSVG(pts[i].x + nx, pts[i].y + ny, 0);
            const sb = projSVG(pts[i].x - nx, pts[i].y - ny, 0);
            const sc = projSVG(pts[i+1].x - nx, pts[i+1].y - ny, 0);
            const sd = projSVG(pts[i+1].x + nx, pts[i+1].y + ny, 0);

            svg += `<path d="M ${sa[0]},${sa[1]} L ${sb[0]},${sb[1]} L ${sc[0]},${sc[1]} L ${sd[0]},${sd[1]} Z" fill="#000" opacity="0.12"/>`;
        }
    });

    // Second pass: actual ducts
    ductElements.forEach(el => {
        const pts = el.points.map(p => toLocal(p));
        for (let i = 0; i < pts.length - 1; i++) {
            svg += drawDuctSegment(pts[i], pts[i + 1], WALL_HEIGHT - 6);
        }
    });

    // === DIFFUSERS - clean white ceiling registers ===
    elements.forEach(el => {
        if (el.type === 'diffuser') {
            const p = toLocal({ x: el.x, y: el.y });
            const size = 7;
            const z = WALL_HEIGHT - 1;
            const corners = [
                { x: p.x - size, y: p.y - size },
                { x: p.x + size, y: p.y - size },
                { x: p.x + size, y: p.y + size },
                { x: p.x - size, y: p.y + size }
            ];
            const proj4 = corners.map(c => projSVG(c.x, c.y, z));
            const path = `M ${proj4[0][0]},${proj4[0][1]} L ${proj4[1][0]},${proj4[1][1]} L ${proj4[2][0]},${proj4[2][1]} L ${proj4[3][0]},${proj4[3][1]} Z`;
            svg += `<path d="${path}" fill="#ffffff" stroke="#888" stroke-width="0.5"/>`;
            // Inner detail
            const innerSize = 4;
            const innerCorners = [
                { x: p.x - innerSize, y: p.y - innerSize },
                { x: p.x + innerSize, y: p.y - innerSize },
                { x: p.x + innerSize, y: p.y + innerSize },
                { x: p.x - innerSize, y: p.y + innerSize }
            ];
            const proj4i = innerCorners.map(c => projSVG(c.x, c.y, z + 0.1));
            const innerPath = `M ${proj4i[0][0]},${proj4i[0][1]} L ${proj4i[1][0]},${proj4i[1][1]} L ${proj4i[2][0]},${proj4i[2][1]} L ${proj4i[3][0]},${proj4i[3][1]} Z`;
            svg += `<path d="${innerPath}" fill="none" stroke="#aaa" stroke-width="0.4"/>`;
        }
    });

    // === Helper: draw isometric cube with gradients ===
    function drawIsoCube(centerX, centerY, halfSize, height, baseZ, topGrad, frontGrad, rightGrad, stroke) {
        const hs = halfSize;
        const c = [
            { x: centerX - hs, y: centerY - hs },
            { x: centerX + hs, y: centerY - hs },
            { x: centerX + hs, y: centerY + hs },
            { x: centerX - hs, y: centerY + hs }
        ];
        const b = c.map(p => projSVG(p.x, p.y, baseZ));
        const t = c.map(p => projSVG(p.x, p.y, baseZ + height));

        let cube = '';

        // Shadow on floor under cube
        const shadow = c.map(p => projSVG(p.x + 4, p.y + 4, 0.5));
        cube += `<path d="M ${shadow[0][0]},${shadow[0][1]} L ${shadow[1][0]},${shadow[1][1]} L ${shadow[2][0]},${shadow[2][1]} L ${shadow[3][0]},${shadow[3][1]} Z" fill="#000" opacity="0.2"/>`;

        // Top face
        cube += `<path d="M ${t[0][0]},${t[0][1]} L ${t[1][0]},${t[1][1]} L ${t[2][0]},${t[2][1]} L ${t[3][0]},${t[3][1]} Z" fill="${topGrad}" stroke="${stroke}" stroke-width="0.5" stroke-linejoin="round"/>`;

        // Front face (corners 3-2, bottom -> top)
        cube += `<path d="M ${b[3][0]},${b[3][1]} L ${b[2][0]},${b[2][1]} L ${t[2][0]},${t[2][1]} L ${t[3][0]},${t[3][1]} Z" fill="${frontGrad}" stroke="${stroke}" stroke-width="0.5" stroke-linejoin="round"/>`;

        // Right face (corners 1-2, bottom -> top)
        cube += `<path d="M ${b[1][0]},${b[1][1]} L ${b[2][0]},${b[2][1]} L ${t[2][0]},${t[2][1]} L ${t[1][0]},${t[1][1]} Z" fill="${rightGrad}" stroke="${stroke}" stroke-width="0.5" stroke-linejoin="round"/>`;

        // Top highlight edge (gives beveled look)
        cube += `<line x1="${t[0][0]}" y1="${t[0][1]}" x2="${t[1][0]}" y2="${t[1][1]}" stroke="#ffffff" stroke-width="0.8" opacity="0.4"/>`;
        cube += `<line x1="${t[0][0]}" y1="${t[0][1]}" x2="${t[3][0]}" y2="${t[3][1]}" stroke="#ffffff" stroke-width="0.8" opacity="0.4"/>`;

        return cube;
    }

    // === VAVs - blue beveled cubes ===
    elements.forEach(el => {
        if (el.type === 'vav') {
            const p = toLocal({ x: el.x, y: el.y });
            const baseZ = WALL_HEIGHT - 22;
            svg += drawIsoCube(p.x, p.y, 11, 22, baseZ,
                'url(#vavTop)', 'url(#vavFront)', 'url(#vavRight)', '#0c1c5c');
        }
    });

    // === AHU - green beveled cube (bigger) ===
    elements.forEach(el => {
        if (el.type === 'ahu') {
            const p = toLocal({ x: el.x, y: el.y });
            const baseZ = WALL_HEIGHT - 28;
            svg += drawIsoCube(p.x, p.y, 20, 28, baseZ,
                'url(#ahuTop)', 'url(#ahuFront)', 'url(#ahuRight)', '#0a4220');
        }
    });

    svg += '</svg>';
    return svg;
}

const svgContent = generateSVG();
document.getElementById('svgViewer').innerHTML = svgContent;

function downloadSVG() {
    const blob = new Blob([svgContent], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.download = 'bas_graphic.svg';
    link.href = url;
    link.click();
    URL.revokeObjectURL(url);
}

function downloadPNG() {
    const svgBlob = new Blob([svgContent], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(svgBlob);
    const img = new Image();
    img.onload = function() {
        const canvas = document.createElement('canvas');
        canvas.width = img.width * 2;
        canvas.height = img.height * 2;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#0a0a0d';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(function(blob) {
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
</body>
</html>'''


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        password = request.form.get("password", "")
        if password == APP_PASSWORD:
            response = make_response(redirect("/"))
            response.set_cookie(
                "bas_auth",
                APP_PASSWORD,
                max_age=60 * 60 * 24 * 7,
                httponly=True,
                secure=True,
                samesite="Lax"
            )
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
    allowed_endpoints = ["login", "static"]
    if request.endpoint in allowed_endpoints:
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
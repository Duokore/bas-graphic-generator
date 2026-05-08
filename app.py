from flask import Flask, request, render_template_string, jsonify
import os
import base64
import json
import cv2
import fitz
import anthropic

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
UPLOAD_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, "mechanical_upload.png")
UPLOAD_PDF_PATH = os.path.join(UPLOAD_FOLDER, "mechanical_upload.pdf")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def pdf_to_png(pdf_path, out_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
    pix.save(out_path)
    doc.close()


def analyze_plan_with_claude(image_path):
    img_b64 = image_to_base64(image_path)

    prompt_text = (
        "You are an expert HVAC engineer analyzing a mechanical floor plan for 3D rendering. "
        "Look at this plan VERY carefully and extract precise data in JSON format. "
        "All coordinates MUST be normalized 0-1000 (where 0,0 is top-left of the plan). "
        "Return ONLY valid JSON with these fields:\n"
        "- building_outline: polygon points of the OUTER walls of the building\n"
        "- rooms: array of {bbox: [x,y,w,h]} for each room/space\n"
        "- vavs: array of {pos: [x,y]} for each VAV box (small rectangular HVAC boxes)\n"
        "- ahus: array of {pos: [x,y], size: [w,h]} for Air Handling Units (larger equipment)\n"
        "- ducts: array of {path: [[x,y],...], type: 'supply' or 'return'} - main duct routes\n"
        "- diffusers: array of {pos: [x,y]} - ceiling grills/registers\n"
        "Be VERY accurate with positions - this is for a 3D model. Detect EVERY VAV, EVERY duct branch, EVERY diffuser.\n"
        "The duct paths should be the actual centerline of ducts as they branch from AHU to VAVs to diffusers.\n"
        "Example output:\n"
        "{\"building_outline\":[[100,100],[900,100],[900,900],[100,900]],"
        "\"rooms\":[{\"bbox\":[120,120,200,150]}],"
        "\"vavs\":[{\"pos\":[200,180]}],"
        "\"ahus\":[{\"pos\":[500,500],\"size\":[80,60]}],"
        "\"ducts\":[{\"path\":[[500,500],[400,500],[200,200]],\"type\":\"supply\"}],"
        "\"diffusers\":[{\"pos\":[200,200]}]}"
    )

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
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


HOME_PAGE = '''<!DOCTYPE html>
<html>
<head>
<title>BAS Generator v11 - Pro 3D</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d0f14; color: white; font-family: 'Segoe UI', Arial, sans-serif; min-height: 100vh; display: flex; align-items: center; justify-content: center; }
.card { background: #181b24; border: 1px solid #2a2f3e; border-radius: 24px; padding: 50px; text-align: center; max-width: 720px; width: 90%; box-shadow: 0 0 60px rgba(0,0,0,0.6); }
.logo { font-size: 56px; margin-bottom: 16px; }
h1 { font-size: 32px; margin-bottom: 8px; background: linear-gradient(135deg, #2d89ef, #b388ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.sub { color: #7a8099; font-size: 14px; margin-bottom: 36px; }
.zone { border: 2px dashed #2d3348; border-radius: 16px; padding: 36px; margin-bottom: 24px; background: #13151d; transition: border-color 0.2s; }
.zone:hover { border-color: #2d89ef; }
input[type=file] { background: transparent; color: #aab0c4; border: none; font-size: 14px; width: 100%; cursor: pointer; }
.lbl { display: block; font-size: 13px; color: #5a6280; margin-top: 10px; }
.btn { background: linear-gradient(135deg, #1a6fd4, #2d89ef); color: white; border: none; border-radius: 14px; padding: 18px 40px; font-size: 18px; font-weight: 700; cursor: pointer; width: 100%; box-shadow: 0 4px 20px rgba(45,137,239,0.3); }
.btn:hover { transform: translateY(-2px); }
.badge { display: inline-block; background: #1e2233; border: 1px solid #2a3050; border-radius: 8px; padding: 6px 14px; font-size: 12px; color: #6878a8; margin: 4px; }
.ai { background: linear-gradient(135deg, #7c4dff, #b388ff); color: white; border: none; }
.pro { background: linear-gradient(135deg, #ff9800, #ff5722); color: white; border: none; }
.note { font-size: 12px; color: #5a6280; margin-top: 14px; font-style: italic; }
.footer { color: #3a4060; font-size: 12px; margin-top: 28px; }
</style>
</head>
<body>
<div class="card">
<div class="logo">&#127970;</div>
<h1>BAS Graphic Generator</h1>
<p class="sub">Professional 3D HVAC Visualization - Powered by Claude AI</p>
<div style="margin-bottom:24px;">
<span class="badge ai">Claude AI</span>
<span class="badge pro">Real 3D Engine</span>
<span class="badge">Lighting &amp; Shadows</span>
<span class="badge">Interactive View</span>
</div>
<form action="/generate" method="post" enctype="multipart/form-data">
<div class="zone">
<input type="file" name="file" accept="image/png,image/jpeg,application/pdf" required>
<span class="lbl">Upload mechanical plan - PNG, JPG or PDF</span>
</div>
<button class="btn" type="submit">Generate Professional 3D Graphic</button>
<p class="note">AI analysis takes 30-50 seconds for complex plans</p>
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
body { background: #0d0f14; color: white; font-family: 'Segoe UI', Arial, sans-serif; padding: 20px; }
h1 { text-align: center; font-size: 26px; margin-bottom: 6px; background: linear-gradient(135deg, #2d89ef, #b388ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.sub { text-align: center; color: #6878a8; font-size: 13px; margin-bottom: 20px; }
.stats { display: flex; justify-content: center; gap: 16px; margin: 14px 0 20px; flex-wrap: wrap; }
.stat { background: #1e2233; padding: 8px 18px; border-radius: 8px; font-size: 13px; color: #aab0c4; border: 1px solid #2a3050; }
.stat b { color: #fff; }
.viewer-3d { width: 100%; height: 75vh; background: #f0f0f0; border-radius: 16px; border: 1px solid #2a3050; overflow: hidden; position: relative; }
.controls-help { position: absolute; bottom: 16px; left: 16px; background: rgba(0,0,0,0.6); color: white; padding: 10px 16px; border-radius: 8px; font-size: 12px; z-index: 10; }
.actions { text-align: center; margin-top: 24px; display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; }
.btn { padding: 13px 28px; border: none; border-radius: 12px; font-size: 15px; font-weight: 700; cursor: pointer; text-decoration: none; display: inline-block; }
.btn-blue { background: #1a6fd4; color: white; }
.btn-green { background: #1a9e4a; color: white; }
.btn-gray { background: #252a38; color: #aab0c4; }
.footer { text-align: center; color: #3a4060; font-size: 12px; margin-top: 24px; }
.loading { position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%); color: #888; font-size: 14px; }
</style>
</head>
<body>
<h1>Professional 3D BAS Graphic</h1>
<p class="sub">Drag to rotate - Scroll to zoom - Right-click to pan</p>
<div class="stats">
<div class="stat">Rooms: <b>{{ n_rooms }}</b></div>
<div class="stat">VAVs: <b>{{ n_vavs }}</b></div>
<div class="stat">AHUs: <b>{{ n_ahus }}</b></div>
<div class="stat">Ducts: <b>{{ n_ducts }}</b></div>
<div class="stat">Diffusers: <b>{{ n_diffs }}</b></div>
</div>
<div class="viewer-3d" id="viewer">
<div class="controls-help">
<b>Controls:</b> Left click + drag = Rotate | Scroll = Zoom | Right click + drag = Pan
</div>
</div>
<div class="actions">
<button onclick="screenshot()" class="btn btn-green">Download PNG Screenshot</button>
<button onclick="resetView()" class="btn btn-blue">Reset View</button>
<a href="/" class="btn btn-gray">Generate Another</a>
</div>
<div class="footer">Made by Paolo V. and Emmanuel R.</div>

<script>
const planData = {{ plan_data_json | safe }};

let scene, camera, renderer, controls;
let initialCameraPos;

function init() {
    const container = document.getElementById('viewer');
    const width = container.clientWidth;
    const height = container.clientHeight;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);
    scene.fog = new THREE.Fog(0xf0f0f0, 800, 2500);

    camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 5000);
    camera.position.set(600, 500, 600);
    initialCameraPos = camera.position.clone();

    renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputEncoding = THREE.sRGBEncoding;
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0, 0);

    // Lighting setup - professional
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.55);
    scene.add(ambientLight);

    const sunLight = new THREE.DirectionalLight(0xffffff, 0.85);
    sunLight.position.set(400, 800, 300);
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.width = 2048;
    sunLight.shadow.mapSize.height = 2048;
    sunLight.shadow.camera.left = -800;
    sunLight.shadow.camera.right = 800;
    sunLight.shadow.camera.top = 800;
    sunLight.shadow.camera.bottom = -800;
    sunLight.shadow.camera.near = 0.5;
    sunLight.shadow.camera.far = 2000;
    sunLight.shadow.bias = -0.001;
    scene.add(sunLight);

    const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
    fillLight.position.set(-300, 400, -200);
    scene.add(fillLight);

    buildScene();

    window.addEventListener('resize', onResize);
    animate();
}

function buildScene() {
    // Center coordinates
    let cx = 500, cy = 500;
    if (planData.building_outline && planData.building_outline.length > 0) {
        const xs = planData.building_outline.map(p => p[0]);
        const ys = planData.building_outline.map(p => p[1]);
        cx = (Math.min(...xs) + Math.max(...xs)) / 2;
        cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    }

    const SCALE = 1.0;
    const WALL_HEIGHT = 60;
    const WALL_THICKNESS = 6;

    function toWorld(p) {
        return [(p[0] - cx) * SCALE, (p[1] - cy) * SCALE];
    }

    // === FLOOR ===
    if (planData.building_outline && planData.building_outline.length >= 3) {
        const floorShape = new THREE.Shape();
        const outline = planData.building_outline.map(toWorld);
        floorShape.moveTo(outline[0][0], outline[0][1]);
        for (let i = 1; i < outline.length; i++) {
            floorShape.lineTo(outline[i][0], outline[i][1]);
        }
        floorShape.lineTo(outline[0][0], outline[0][1]);

        const floorGeo = new THREE.ShapeGeometry(floorShape);
        const floorMat = new THREE.MeshStandardMaterial({
            color: 0xb8b8b8,
            roughness: 0.9,
            metalness: 0.0
        });
        const floor = new THREE.Mesh(floorGeo, floorMat);
        floor.rotation.x = -Math.PI / 2;
        floor.position.y = 0;
        floor.receiveShadow = true;
        scene.add(floor);

        // Tile pattern overlay using grid
        const gridSize = 30;
        const xs = outline.map(p => p[0]);
        const ys = outline.map(p => p[1]);
        const minX = Math.min(...xs), maxX = Math.max(...xs);
        const minZ = Math.min(...ys), maxZ = Math.max(...ys);

        const gridMaterial = new THREE.LineBasicMaterial({ color: 0x999999, opacity: 0.4, transparent: true });

        for (let x = minX; x <= maxX; x += gridSize) {
            const points = [
                new THREE.Vector3(x, 0.5, minZ),
                new THREE.Vector3(x, 0.5, maxZ)
            ];
            const geo = new THREE.BufferGeometry().setFromPoints(points);
            scene.add(new THREE.Line(geo, gridMaterial));
        }
        for (let z = minZ; z <= maxZ; z += gridSize) {
            const points = [
                new THREE.Vector3(minX, 0.5, z),
                new THREE.Vector3(maxX, 0.5, z)
            ];
            const geo = new THREE.BufferGeometry().setFromPoints(points);
            scene.add(new THREE.Line(geo, gridMaterial));
        }
    }

    // === EXTERIOR WALLS ===
    if (planData.building_outline && planData.building_outline.length >= 3) {
        const outline = planData.building_outline.map(toWorld);
        const wallMat = new THREE.MeshStandardMaterial({
            color: 0x6a6f78,
            roughness: 0.8,
            metalness: 0.05
        });
        for (let i = 0; i < outline.length; i++) {
            const p1 = outline[i];
            const p2 = outline[(i + 1) % outline.length];
            buildWall(p1, p2, WALL_HEIGHT, WALL_THICKNESS * 1.5, wallMat);
        }
    }

    // === INTERIOR WALLS (room dividers) ===
    const interiorWallMat = new THREE.MeshStandardMaterial({
        color: 0x7a7f88,
        roughness: 0.85,
        metalness: 0.05
    });

    (planData.rooms || []).forEach(room => {
        const [x, y, w, h] = room.bbox;
        const corners = [
            toWorld([x, y]),
            toWorld([x + w, y]),
            toWorld([x + w, y + h]),
            toWorld([x, y + h])
        ];
        for (let i = 0; i < 4; i++) {
            buildWall(corners[i], corners[(i + 1) % 4], WALL_HEIGHT * 0.95, WALL_THICKNESS, interiorWallMat);
        }
    });

    // === DUCTS (white tubular pipes near ceiling) ===
    const ductMat = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        roughness: 0.4,
        metalness: 0.3,
        emissive: 0xeeeeee,
        emissiveIntensity: 0.1
    });

    (planData.ducts || []).forEach(duct => {
        if (!duct.path || duct.path.length < 2) return;
        const path = duct.path.map(toWorld);
        for (let i = 0; i < path.length - 1; i++) {
            buildDuct(path[i], path[i + 1], WALL_HEIGHT - 8, ductMat);
        }
    });

    // === VAV BOXES (blue 3D cubes) ===
    const vavMat = new THREE.MeshStandardMaterial({
        color: 0x2244aa,
        roughness: 0.4,
        metalness: 0.2,
        emissive: 0x1133aa,
        emissiveIntensity: 0.15
    });
    const vavSize = 22;

    (planData.vavs || []).forEach(vav => {
        const [px, py] = toWorld(vav.pos);
        const geo = new THREE.BoxGeometry(vavSize, vavSize, vavSize);
        const mesh = new THREE.Mesh(geo, vavMat);
        mesh.position.set(px, WALL_HEIGHT - 18, py);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        scene.add(mesh);
    });

    // === AHU (large green/gray equipment) ===
    const ahuMat = new THREE.MeshStandardMaterial({
        color: 0x2a8a3a,
        roughness: 0.6,
        metalness: 0.3,
        emissive: 0x1a5a2a,
        emissiveIntensity: 0.1
    });

    (planData.ahus || []).forEach((ahu, idx) => {
        const [px, py] = toWorld(ahu.pos);
        const w = (ahu.size && ahu.size[0]) ? ahu.size[0] : 50;
        const h = (ahu.size && ahu.size[1]) ? ahu.size[1] : 40;
        const ahuColor = idx === 0 ? ahuMat : new THREE.MeshStandardMaterial({
            color: 0x8a9aaa, roughness: 0.6, metalness: 0.3
        });
        const geo = new THREE.BoxGeometry(w, 35, h);
        const mesh = new THREE.Mesh(geo, ahuColor);
        mesh.position.set(px, WALL_HEIGHT - 22, py);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        scene.add(mesh);
    });

    // === DIFFUSERS (small white squares on ceiling) ===
    const diffuserMat = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        roughness: 0.5,
        metalness: 0.2
    });

    (planData.diffusers || []).forEach(diff => {
        const [px, py] = toWorld(diff.pos);
        const geo = new THREE.BoxGeometry(8, 2, 8);
        const mesh = new THREE.Mesh(geo, diffuserMat);
        mesh.position.set(px, WALL_HEIGHT - 4, py);
        mesh.castShadow = true;
        scene.add(mesh);
    });

    // Auto-center camera on building
    if (planData.building_outline && planData.building_outline.length > 0) {
        const outline = planData.building_outline.map(toWorld);
        const xs = outline.map(p => p[0]);
        const ys = outline.map(p => p[1]);
        const sizeX = Math.max(...xs) - Math.min(...xs);
        const sizeZ = Math.max(...ys) - Math.min(...ys);
        const maxSize = Math.max(sizeX, sizeZ);
        const distance = maxSize * 1.3;
        camera.position.set(distance * 0.7, distance * 0.6, distance * 0.7);
        initialCameraPos = camera.position.clone();
        controls.target.set(0, 0, 0);
        controls.update();
    }
}

function buildWall(p1, p2, height, thickness, material) {
    const dx = p2[0] - p1[0];
    const dz = p2[1] - p1[1];
    const length = Math.sqrt(dx * dx + dz * dz);
    if (length < 1) return;

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

function buildDuct(p1, p2, yPos, material) {
    const dx = p2[0] - p1[0];
    const dz = p2[1] - p1[1];
    const length = Math.sqrt(dx * dx + dz * dz);
    if (length < 1) return;

    const angle = Math.atan2(dz, dx);
    const cx = (p1[0] + p2[0]) / 2;
    const cz = (p1[1] + p2[1]) / 2;

    // Rectangular duct
    const geo = new THREE.BoxGeometry(length, 8, 12);
    const mesh = new THREE.Mesh(geo, material);
    mesh.position.set(cx, yPos, cz);
    mesh.rotation.y = -angle;
    mesh.castShadow = true;
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
    controls.target.set(0, 0, 0);
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
        plan_data = analyze_plan_with_claude(UPLOAD_IMAGE_PATH)
    except Exception as e:
        error_msg = str(e)
        return "<h2 style='color:white;background:#0d0f14;padding:30px;'>AI analysis failed: " + error_msg + "</h2>", 500

    return render_template_string(
        RESULT_PAGE,
        plan_data_json=json.dumps(plan_data),
        n_rooms=len(plan_data.get("rooms", [])),
        n_vavs=len(plan_data.get("vavs", [])),
        n_ahus=len(plan_data.get("ahus", [])),
        n_ducts=len(plan_data.get("ducts", [])),
        n_diffs=len(plan_data.get("diffusers", []))
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

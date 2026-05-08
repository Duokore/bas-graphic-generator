from flask import Flask, request, render_template_string
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
        "You are an expert HVAC/BAS controls graphic designer and mechanical plan analyst. "
        "Analyze this HVAC mechanical floor plan and return ONLY valid JSON. "
        "All coordinates must be normalized 0-1000, where 0,0 is top-left. "
        "Detect the building perimeter, rooms, VAVs, AHUs, ducts, and diffusers. "
        "The output will be rendered as a professional 3D BAS graphic, so be accurate. "
        "Return this JSON schema only:\n"
        "{"
        "\"building_outline\":[[x,y],[x,y]],"
        "\"rooms\":[{\"bbox\":[x,y,w,h]}],"
        "\"vavs\":[{\"pos\":[x,y]}],"
        "\"ahus\":[{\"pos\":[x,y],\"size\":[w,h]}],"
        "\"ducts\":[{\"path\":[[x,y],[x,y]]}],"
        "\"diffusers\":[{\"pos\":[x,y]}]"
        "}"
    )

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_b64,
                        },
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
    )

    response_text = message.content[0].text.strip()

    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
        response_text = response_text.strip()

    return json.loads(response_text)


HOME_PAGE = """<!DOCTYPE html>
<html>
<head>
<title>BAS Generator v14</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    background: #050608;
    color: white;
    font-family: 'Segoe UI', Arial, sans-serif;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
}
.card {
    background: #11141d;
    border: 1px solid #262b3a;
    border-radius: 26px;
    padding: 52px;
    text-align: center;
    max-width: 760px;
    width: 90%;
    box-shadow: 0 0 70px rgba(0,0,0,0.75);
}
.logo { font-size: 58px; margin-bottom: 18px; }
h1 {
    font-size: 34px;
    margin-bottom: 8px;
    color: #7aa7ff;
}
.sub { color: #8b93ad; font-size: 14px; margin-bottom: 34px; }
.zone {
    border: 2px dashed #2b3145;
    border-radius: 18px;
    padding: 38px;
    margin-bottom: 24px;
    background: #0d1018;
}
.zone:hover { border-color: #2d89ef; }
input[type=file] {
    background: transparent;
    color: #cbd5e1;
    border: none;
    font-size: 14px;
    width: 100%;
}
.btn {
    background: linear-gradient(135deg, #1d65d8, #2d89ef);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 18px 40px;
    font-size: 18px;
    font-weight: 700;
    cursor: pointer;
    width: 100%;
}
.badge {
    display: inline-block;
    background: #171c2d;
    border: 1px solid #2c3655;
    border-radius: 9px;
    padding: 7px 15px;
    font-size: 12px;
    color: #aab7e8;
    margin: 4px;
}
.footer { color: #4a5068; font-size: 12px; margin-top: 28px; }
</style>
</head>
<body>
<div class="card">
    <div class="logo">🏢</div>
    <h1>BAS Graphic Generator v14</h1>
    <p class="sub">Professional 3D HVAC / BAS Visualization</p>

    <div style="margin-bottom:24px;">
        <span class="badge">Claude AI</span>
        <span class="badge">Three.js 3D</span>
        <span class="badge">BAS Style Renderer</span>
        <span class="badge">Isometric View</span>
    </div>

    <form action="/generate" method="post" enctype="multipart/form-data">
        <div class="zone">
            <input type="file" name="file" accept="image/png,image/jpeg,application/pdf" required>
            <div style="font-size:13px;color:#67708d;margin-top:12px;">
                Upload mechanical plan - PNG, JPG or PDF
            </div>
        </div>

        <button class="btn" type="submit">Generate Professional 3D Graphic</button>
    </form>

    <div class="footer">Made by Paolo V. and Emmanuel R.</div>
</div>
</body>
</html>"""


RESULT_PAGE = """<!DOCTYPE html>
<html>
<head>
<title>Professional 3D BAS Graphic v14</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    background:#050608;
    color:white;
    font-family:'Segoe UI', Arial, sans-serif;
    overflow:hidden;
}
.header {
    height:92px;
    background:#07090f;
    border-bottom:1px solid #151923;
    text-align:center;
    padding-top:12px;
}
h1 {
    font-size:24px;
    color:#76a9ff;
    letter-spacing:.4px;
}
.sub {
    color:#7d88aa;
    font-size:12px;
    margin-top:4px;
}
.stats {
    display:flex;
    justify-content:center;
    gap:12px;
    margin-top:10px;
    flex-wrap:wrap;
}
.stat {
    background:#161b2b;
    border:1px solid #26304a;
    padding:6px 14px;
    border-radius:8px;
    font-size:12px;
    color:#cbd5ff;
}
.stat b { color:#fff; }
.viewer-3d {
    width:100vw;
    height:calc(100vh - 92px);
    background:#050608;
    position:relative;
}
.controls-help {
    position:absolute;
    bottom:12px;
    left:12px;
    background:rgba(0,0,0,.72);
    padding:9px 14px;
    border-radius:8px;
    font-size:11px;
    color:#e5e7eb;
    z-index:10;
}
.actions {
    position:absolute;
    bottom:12px;
    right:12px;
    display:flex;
    gap:8px;
    z-index:10;
}
.btn {
    padding:10px 16px;
    border:none;
    border-radius:9px;
    font-size:12px;
    font-weight:700;
    cursor:pointer;
}
.btn-green { background:#16a34a; color:white; }
.btn-blue { background:#2563eb; color:white; }
.btn-gray { background:#242936; color:#d1d5db; }
</style>
</head>

<body>
<div class="header">
    <h1>Professional 3D BAS Graphic v14</h1>
    <p class="sub">Flat Isometric · Black Background · BAS Controls Style</p>
    <div class="stats">
        <div class="stat">Rooms: <b>{{ n_rooms }}</b></div>
        <div class="stat">VAVs: <b>{{ n_vavs }}</b></div>
        <div class="stat">AHUs: <b>{{ n_ahus }}</b></div>
        <div class="stat">Ducts: <b>{{ n_ducts }}</b></div>
        <div class="stat">Diffusers: <b>{{ n_diffs }}</b></div>
    </div>
</div>

<div class="viewer-3d" id="viewer">
    <div class="controls-help">
        Left drag = Rotate | Scroll = Zoom | Right drag = Pan
    </div>

    <div class="actions">
        <button onclick="screenshot()" class="btn btn-green">Download PNG</button>
        <button onclick="flatView()" class="btn btn-blue">Flat View</button>
        <button onclick="resetView()" class="btn btn-gray">Reset</button>
        <a href="/" style="text-decoration:none;"><button class="btn btn-gray">New Plan</button></a>
    </div>
</div>

<script>
const planData = {{ plan_data_json | safe }};

let scene, camera, renderer, controls;
let initialCameraPos, initialTarget;
let buildingCenter = { x: 500, z: 500, sizeX: 1000, sizeZ: 1000 };

function init() {
    const container = document.getElementById('viewer');
    const width = container.clientWidth;
    const height = container.clientHeight;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050608);

    camera = new THREE.PerspectiveCamera(22, width / height, 0.1, 9000);

    renderer = new THREE.WebGLRenderer({
        antialias:true,
        preserveDrawingBuffer:true,
        alpha:false
    });

    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputEncoding = THREE.sRGBEncoding;

    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 220;
    controls.maxDistance = 3500;
    controls.enablePan = true;

    setupLighting();
    buildScene();

    window.addEventListener('resize', onResize);
    animate();
}

function setupLighting() {
    scene.add(new THREE.AmbientLight(0xffffff, 0.82));

    const topLight = new THREE.DirectionalLight(0xffffff, 0.7);
    topLight.position.set(0, 950, 240);
    topLight.castShadow = true;
    topLight.shadow.mapSize.width = 4096;
    topLight.shadow.mapSize.height = 4096;
    topLight.shadow.camera.left = -1800;
    topLight.shadow.camera.right = 1800;
    topLight.shadow.camera.top = 1800;
    topLight.shadow.camera.bottom = -1800;
    scene.add(topLight);

    const softFill = new THREE.HemisphereLight(0xe2eaff, 0x202020, 0.65);
    scene.add(softFill);
}

function buildScene() {
    const SCALE = 1.0;
    const WALL_HEIGHT = 86;
    const EXT_WALL = 14;
    const INT_WALL = 8;

    let cx = 500, cy = 500;
    let sizeX = 1000, sizeZ = 1000;

    if (planData.building_outline && planData.building_outline.length > 2) {
        const xs = planData.building_outline.map(p => p[0]);
        const ys = planData.building_outline.map(p => p[1]);
        cx = (Math.min(...xs) + Math.max(...xs)) / 2;
        cy = (Math.min(...ys) + Math.max(...ys)) / 2;
        sizeX = Math.max(...xs) - Math.min(...xs);
        sizeZ = Math.max(...ys) - Math.min(...ys);
    }

    buildingCenter = { x:cx, z:cy, sizeX:sizeX, sizeZ:sizeZ };

    function toWorld(p) {
        return [(p[0] - cx) * SCALE, (p[1] - cy) * SCALE];
    }

    const floorMat = new THREE.MeshStandardMaterial({
        color:0x6f7379,
        roughness:0.92,
        metalness:0.02
    });

    const wallOuterMat = new THREE.MeshStandardMaterial({
        color:0x3e4249,
        roughness:0.85,
        metalness:0.05
    });

    const wallInnerMat = new THREE.MeshStandardMaterial({
        color:0x6f747c,
        roughness:0.86,
        metalness:0.04
    });

    createFloor(toWorld, floorMat);
    createTileGrid(toWorld);
    createExteriorWalls(toWorld, WALL_HEIGHT, EXT_WALL, wallOuterMat);
    createInteriorWalls(toWorld, WALL_HEIGHT, INT_WALL, wallInnerMat);
    createWindows(toWorld, WALL_HEIGHT);
    createDucts(toWorld, WALL_HEIGHT);
    createVAVs(toWorld, WALL_HEIGHT);
    createAHUs(toWorld, WALL_HEIGHT);
    createDiffusers(toWorld, WALL_HEIGHT);

    const maxSize = Math.max(sizeX, sizeZ);

    camera.position.set(maxSize * 0.42, maxSize * 0.50, maxSize * 0.88);
    initialCameraPos = camera.position.clone();
    initialTarget = new THREE.Vector3(0, 30, 0);

    controls.target.copy(initialTarget);
    controls.update();
}

function createFloor(toWorld, mat) {
    if (!planData.building_outline || planData.building_outline.length < 3) return;

    const outline = planData.building_outline.map(toWorld);
    const shape = new THREE.Shape();

    shape.moveTo(outline[0][0], outline[0][1]);

    for (let i = 1; i < outline.length; i++) {
        shape.lineTo(outline[i][0], outline[i][1]);
    }

    shape.lineTo(outline[0][0], outline[0][1]);

    const geo = new THREE.ShapeGeometry(shape);
    const floor = new THREE.Mesh(geo, mat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = 0;
    floor.receiveShadow = true;
    scene.add(floor);
}

function createTileGrid(toWorld) {
    if (!planData.building_outline || planData.building_outline.length < 3) return;

    const outline = planData.building_outline.map(toWorld);
    const xs = outline.map(p => p[0]);
    const zs = outline.map(p => p[1]);

    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minZ = Math.min(...zs);
    const maxZ = Math.max(...zs);

    const mat = new THREE.LineBasicMaterial({
        color:0xffffff,
        transparent:true,
        opacity:0.24
    });

    const step = 22;

    for (let x = minX; x <= maxX; x += step) {
        const geo = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(x, 1.2, minZ),
            new THREE.Vector3(x, 1.2, maxZ)
        ]);
        scene.add(new THREE.Line(geo, mat));
    }

    for (let z = minZ; z <= maxZ; z += step) {
        const geo = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(minX, 1.2, z),
            new THREE.Vector3(maxX, 1.2, z)
        ]);
        scene.add(new THREE.Line(geo, mat));
    }
}

function createExteriorWalls(toWorld, height, thickness, mat) {
    if (!planData.building_outline || planData.building_outline.length < 3) return;

    const outline = planData.building_outline.map(toWorld);

    for (let i = 0; i < outline.length; i++) {
        buildWall(outline[i], outline[(i + 1) % outline.length], height, thickness, mat);
    }
}

function createInteriorWalls(toWorld, height, thickness, mat) {
    (planData.rooms || []).forEach(room => {
        const [x,y,w,h] = room.bbox;

        const pts = [
            toWorld([x,y]),
            toWorld([x+w,y]),
            toWorld([x+w,y+h]),
            toWorld([x,y+h])
        ];

        for (let i = 0; i < 4; i++) {
            buildWall(pts[i], pts[(i+1)%4], height * 0.92, thickness, mat);
        }
    });
}

function createWindows(toWorld, wallHeight) {
    if (!planData.building_outline || planData.building_outline.length < 3) return;

    const glassMat = new THREE.MeshStandardMaterial({
        color:0x79aee8,
        transparent:true,
        opacity:0.55,
        roughness:0.15,
        metalness:0.1
    });

    const outline = planData.building_outline.map(toWorld);

    for (let i = 0; i < outline.length; i++) {
        const p1 = outline[i];
        const p2 = outline[(i+1)%outline.length];

        const dx = p2[0] - p1[0];
        const dz = p2[1] - p1[1];
        const len = Math.sqrt(dx*dx + dz*dz);

        if (len < 120) continue;

        const count = Math.floor(len / 115);

        for (let j = 1; j <= count; j++) {
            const t = j / (count + 1);
            const x = p1[0] + dx * t;
            const z = p1[1] + dz * t;
            const angle = Math.atan2(dz, dx);

            const geo = new THREE.BoxGeometry(34, 22, 2);
            const win = new THREE.Mesh(geo, glassMat);
            win.position.set(x, wallHeight * 0.58, z);
            win.rotation.y = -angle;
            scene.add(win);
        }
    }
}

function createDucts(toWorld, wallHeight) {
    const ductMat = new THREE.MeshStandardMaterial({
        color:0xffffff,
        roughness:0.2,
        metalness:0.42,
        emissive:0xffffff,
        emissiveIntensity:0.08
    });

    (planData.ducts || []).forEach(duct => {
        if (!duct.path || duct.path.length < 2) return;

        const path = duct.path.map(toWorld);

        for (let i = 0; i < path.length - 1; i++) {
            buildDuct(path[i], path[i+1], wallHeight + 8, ductMat);
        }
    });
}

function createVAVs(toWorld, wallHeight) {
    const mat = new THREE.MeshStandardMaterial({
        color:0x0b3b8f,
        roughness:0.34,
        metalness:0.28,
        emissive:0x0b4dcc,
        emissiveIntensity:0.18
    });

    (planData.vavs || []).forEach(vav => {
        const [x,z] = toWorld(vav.pos);
        const geo = new THREE.BoxGeometry(32, 25, 32);
        const box = new THREE.Mesh(geo, mat);
        box.position.set(x, wallHeight + 8, z);
        box.castShadow = true;
        box.receiveShadow = true;
        scene.add(box);
    });
}

function createAHUs(toWorld, wallHeight) {
    (planData.ahus || []).forEach((ahu, idx) => {
        const [x,z] = toWorld(ahu.pos);
        const sx = ahu.size ? Math.max(ahu.size[0], 62) : 72;
        const sz = ahu.size ? Math.max(ahu.size[1], 50) : 56;

        const mat = new THREE.MeshStandardMaterial({
            color: idx === 0 ? 0x139b3a : 0x8b949e,
            roughness:0.45,
            metalness:0.24,
            emissive: idx === 0 ? 0x0b6f2b : 0x000000,
            emissiveIntensity: idx === 0 ? 0.14 : 0
        });

        const geo = new THREE.BoxGeometry(sx, 43, sz);
        const unit = new THREE.Mesh(geo, mat);
        unit.position.set(x, wallHeight + 8, z);
        unit.castShadow = true;
        unit.receiveShadow = true;
        scene.add(unit);
    });
}

function createDiffusers(toWorld, wallHeight) {
    const mat = new THREE.MeshStandardMaterial({
        color:0xffffff,
        roughness:0.48,
        metalness:0.18
    });

    (planData.diffusers || []).forEach(diff => {
        const [x,z] = toWorld(diff.pos);
        const geo = new THREE.BoxGeometry(14, 3, 14);
        const d = new THREE.Mesh(geo, mat);
        d.position.set(x, wallHeight + 17, z);
        d.castShadow = true;
        scene.add(d);
    });
}

function buildWall(p1, p2, height, thickness, material) {
    const dx = p2[0] - p1[0];
    const dz = p2[1] - p1[1];
    const len = Math.sqrt(dx*dx + dz*dz);

    if (len < 2) return;

    const angle = Math.atan2(dz, dx);
    const cx = (p1[0] + p2[0]) / 2;
    const cz = (p1[1] + p2[1]) / 2;

    const geo = new THREE.BoxGeometry(len, height, thickness);
    const mesh = new THREE.Mesh(geo, material);

    mesh.position.set(cx, height/2, cz);
    mesh.rotation.y = -angle;
    mesh.castShadow = true;
    mesh.receiveShadow = true;

    scene.add(mesh);
}

function buildDuct(p1, p2, y, material) {
    const dx = p2[0] - p1[0];
    const dz = p2[1] - p1[1];
    const len = Math.sqrt(dx*dx + dz*dz);

    if (len < 2) return;

    const angle = Math.atan2(dz, dx);
    const cx = (p1[0] + p2[0]) / 2;
    const cz = (p1[1] + p2[1]) / 2;

    const geo = new THREE.BoxGeometry(len, 14, 19);
    const duct = new THREE.Mesh(geo, material);

    duct.position.set(cx, y, cz);
    duct.rotation.y = -angle;
    duct.castShadow = true;
    duct.receiveShadow = true;

    scene.add(duct);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function onResize() {
    const container = document.getElementById('viewer');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

function resetView() {
    camera.position.copy(initialCameraPos);
    controls.target.copy(initialTarget);
    controls.update();
}

function flatView() {
    const maxSize = Math.max(buildingCenter.sizeX, buildingCenter.sizeZ);
    camera.position.set(maxSize * 0.42, maxSize * 0.45, maxSize * 0.90);
    controls.target.set(0, 25, 0);
    controls.update();
}

function screenshot() {
    renderer.render(scene, camera);
    const dataURL = renderer.domElement.toDataURL('image/png');
    const link = document.createElement('a');
    link.download = 'bas_graphic_3d_v14.png';
    link.href = dataURL;
    link.click();
}

init();
</script>
</body>
</html>"""


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
        return (
            "<h2 style='color:white;background:#0d0f14;padding:30px;'>"
            "AI analysis failed: "
            + error_msg
            + "</h2>"
        ), 500

    return render_template_string(
        RESULT_PAGE,
        plan_data_json=json.dumps(plan_data),
        n_rooms=len(plan_data.get("rooms", [])),
        n_vavs=len(plan_data.get("vavs", [])),
        n_ahus=len(plan_data.get("ahus", [])),
        n_ducts=len(plan_data.get("ducts", [])),
        n_diffs=len(plan_data.get("diffusers", [])),
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
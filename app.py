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
# v30: Mask preview paths
MASK_BINARY_PATH = os.path.join(UPLOAD_FOLDER, "clean_mask.png")
MASK_PREVIEW_PATH = os.path.join(UPLOAD_FOLDER, "mask_preview.png")
FLOORPLAN_BASE_PATH = os.path.join(OUTPUT_FOLDER, "floorplan_shape_base.png")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def image_to_base64(path):
    """Read an image file and return base64 string (NO data URI prefix - templates add it)."""
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        data = f.read()
    if not data:
        return ""
    return base64.b64encode(data).decode("utf-8")


def _claude_image_block(path):
    """Build a Claude vision image block from a local PNG/JPG path."""
    if not os.path.exists(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    media_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    data = image_to_base64(path)
    if not data:
        return None
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": data
        }
    }


def run_claude_plan_review():
    """v34: Ask Claude to review current plan artifacts and recommend the next workflow step."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return (
            "ANTHROPIC_API_KEY is not set in this environment.\n\n"
            "Add it in Render Environment Variables, redeploy, then try Claude Review again."
        )

    try:
        from anthropic import Anthropic
    except Exception as e:
        return (
            "The anthropic package could not be imported.\n\n"
            f"Python error: {e}\n\n"
            "Make sure requirements.txt includes anthropic and redeploy Render."
        )

    content = [{
        "type": "text",
        "text": (
            "You are reviewing an AI-assisted BAS/HVAC graphics generation workflow.\n"
            "This is NOT BIM reconstruction and NOT a generic CAD app.\n"
            "Goal: Smart Trace + Smart Cleanup for BAS graphics inspired by Trane Tracer Synchrony.\n\n"
            "Review the attached images if present:\n"
            "1. Original isolated mechanical floorplan.\n"
            "2. Clean architectural mask preview.\n"
            "3. BAS-style floorplan base preview.\n\n"
            "Give short, practical recommendations for a controls technician workflow:\n"
            "- Is the exterior footprint usable or not?\n"
            "- Is the interior trace too noisy, too empty, or acceptable?\n"
            "- Which mode should be used next: Balanced, Mechanical Dense, Thin Scan, Floorplan Base, Trace Light, Trace Medium, Trace Detailed, Keep Exterior Only, Clear Interior Trace, or Room Rect?\n"
            "- What should the user clean manually first?\n"
            "- Do not invent exact geometry. Do not promise perfect room detection.\n"
            "- Keep the answer in Spanish, direct, and under 12 bullets."
        )
    }]

    image_paths = [
        ("Original isolated plan", UPLOAD_IMAGE_PATH),
        ("Clean mask preview", MASK_PREVIEW_PATH),
        ("Floorplan base preview", FLOORPLAN_BASE_PATH),
    ]
    attached = []
    for label, path in image_paths:
        block = _claude_image_block(path)
        if block:
            content.append({"type": "text", "text": f"Image: {label}"})
            content.append(block)
            attached.append(label)

    if not attached:
        return "No plan images were found yet. Upload a plan and run Mask Preview first."

    try:
        client = Anthropic(api_key=api_key)
        model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        message = client.messages.create(
            model=model,
            max_tokens=900,
            messages=[{"role": "user", "content": content}]
        )
        parts = []
        for block in message.content:
            if getattr(block, "type", "") == "text":
                parts.append(block.text)
        return "\n\n".join(parts).strip() or "Claude returned no text response."
    except Exception as e:
        return (
            "Claude Review failed while calling the API.\n\n"
            f"Error: {e}\n\n"
            "Check that ANTHROPIC_API_KEY is valid, Render has network access, and the model name is supported. "
            "You can override the model with ANTHROPIC_MODEL in Render."
        )


def pdf_to_png(pdf_path, out_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
    pix.save(out_path)
    doc.close()


# ============================================================
# SMART FLOORPLAN ISOLATION (v29 NEW)
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
# v30: CLEAN ARCHITECTURAL MASK GENERATOR
# ============================================================

def _estimate_line_thickness(binary_img, sample_pixels=200):
    """v30.2: Estimate the median line thickness in the binary image.
    Used to adapt erosion kernel size to the plan's drawing style.
    Returns thickness in pixels (median of distance transform peaks).
    """
    if np.count_nonzero(binary_img) == 0:
        return 3  # safe default
    # Distance transform: each white pixel gets distance to nearest black pixel
    dist = cv2.distanceTransform(binary_img, cv2.DIST_L2, 3)
    # Get distance values where there IS a line
    line_distances = dist[binary_img > 0]
    if len(line_distances) == 0:
        return 3
    # Half-thickness of a line = the max distance you can go before hitting black
    # so total thickness ~= 2 * median of local maxes
    # Simpler: just take 90th percentile of distance values * 2
    p90 = np.percentile(line_distances, 90)
    estimated_thickness = max(2, int(round(p90 * 2)))
    return min(estimated_thickness, 12)  # cap at 12 for sanity


def _is_blob_like(cw, ch, area):
    """v30.2: Detect if a connected component is a 'filled blob' vs a 'line/wall'.
    Walls are hollow elongated strokes (low fill ratio, high aspect).
    Blobs are filled regions (high fill ratio).
    Returns True if it looks like a filled blob we should reject.
    """
    bbox_area = cw * ch
    if bbox_area <= 0:
        return False
    fill_ratio = area / bbox_area
    # A wall outline (rectangle perimeter) has fill_ratio < 0.4
    # A filled blob/region has fill_ratio > 0.6
    # AND the blob is roughly square-ish (aspect < 4)
    aspect = max(cw, ch) / max(min(cw, ch), 1)
    return fill_ratio > 0.55 and aspect < 4 and area > 800


def _repair_small_gaps(mask, max_gap=10):
    """v30.2: Connect endpoint pairs that are very close (within max_gap pixels).
    Uses morphological closing with a small SMART kernel - line-shaped, not block.
    """
    # Use a small line-shaped close in both directions
    # Horizontal close (3xK)
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (max_gap, 1))
    closed_h = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kh, iterations=1)
    # Vertical close (Kx3)
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max_gap))
    closed_v = cv2.morphologyEx(closed_h, cv2.MORPH_CLOSE, kv, iterations=1)
    return closed_v


def _remove_solid_filled_regions(mask, min_area=400):
    """v30.2: Remove SOLID filled regions inside the mask (elevator shafts, hatch fills).
    
    A wall is a HOLLOW outline (thin stroke).
    A filled region is SOLID (mostly pixels-on).
    
    Strategy: For each connected component, check the inner area.
    If most of the bbox is filled = it's a blob, erase that region.
    
    To do this we:
    1. Find each filled region (solid blob) using morphological opening
    2. Subtract those regions from the mask
    """
    # Heavy opening with a small square kernel: this OBLITERATES line strokes
    # but PRESERVES solid filled regions
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    solid_only = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # Now solid_only contains only the "filled" parts (no thin walls)
    # Filter small false positives
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(solid_only, connectivity=8)
    blobs_to_remove = np.zeros_like(mask)
    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]
        # Only treat as blob if it has substantial solid area
        if area >= min_area:
            blobs_to_remove[labels == i] = 255
    
    # Dilate the blob mask slightly so we erase the blob AND its near-stroke neighborhood
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blobs_to_remove = cv2.dilate(blobs_to_remove, kernel_dilate, iterations=2)
    
    # Subtract from the mask
    result = cv2.bitwise_and(mask, cv2.bitwise_not(blobs_to_remove))
    return result


def generate_clean_mask(img_bgr, debug=False, preset="balanced"):
    """v30.3: Generate a clean architectural mask with presets + recovery fallback.

    Philosophy: Smart Trace + Smart Cleanup. The mask should never fail into a
    blank preview. If the selected preset is too aggressive, recovery mode keeps
    more source geometry so the technician can clean it manually.

    Presets:
    - balanced: default v30.2-like behavior
    - mechanical_dense: more aggressive for duct/symbol-heavy drawings
    - thin_scan: conservative for faint/thin wall scans
    - recovery: last-resort low-confidence mask when normal filtering fails

    Returns: clean_mask (binary image) OR (clean_mask, stats) if debug=True
    """
    preset = preset or "balanced"
    presets = {
        "balanced": {
            "threshold": 200,
            "blob_min_area": 400,
            "remove_solid": True,
            "erode_adjust": 0,
            "short_adjust": 0,
            "area_scale": 1.0,
            "long_scale": 1.0,
            "final_min_area": 200,
            "gap_scale": 1.0,
        },
        "mechanical_dense": {
            "threshold": 190,
            "blob_min_area": 300,
            "remove_solid": True,
            "erode_adjust": 1,
            "short_adjust": 1,
            "area_scale": 1.2,
            "long_scale": 1.1,
            "final_min_area": 250,
            "gap_scale": 1.0,
        },
        "thin_scan": {
            "threshold": 225,
            "blob_min_area": 900,
            "remove_solid": False,
            "erode_adjust": -1,
            "short_adjust": -2,
            "area_scale": 0.45,
            "long_scale": 0.55,
            "final_min_area": 80,
            "gap_scale": 0.8,
        },
        "recovery": {
            "threshold": 230,
            "blob_min_area": 1200,
            "remove_solid": False,
            "erode_adjust": -2,
            "short_adjust": -3,
            "area_scale": 0.25,
            "long_scale": 0.35,
            "final_min_area": 50,
            "gap_scale": 0.75,
        },
    }
    if preset not in presets:
        preset = "balanced"
    cfg = presets[preset]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Step 1: Binary inverted using preset-specific threshold.
    _, binary = cv2.threshold(gray, cfg["threshold"], 255, cv2.THRESH_BINARY_INV)
    raw_pixels = int(np.count_nonzero(binary))

    # Step 1b: Optional solid-region removal. Disabled for thin_scan/recovery
    # because this is the step most likely to erase faint architectural lines.
    if cfg["remove_solid"]:
        binary = _remove_solid_filled_regions(binary, min_area=cfg["blob_min_area"])
    after_blob_removal = int(np.count_nonzero(binary))

    # Step 2: Estimate line thickness in this specific plan.
    median_thickness = _estimate_line_thickness(binary)

    # Step 3: Base adaptive parameters by thickness.
    if median_thickness <= 3:
        erode_size = 2
        short_dim_min = 2
        area_min = 150
        long_dim_min = 50
    elif median_thickness >= 6:
        erode_size = 4
        short_dim_min = 5
        area_min = 350
        long_dim_min = 80
    else:
        erode_size = 3
        short_dim_min = 3
        area_min = 250
        long_dim_min = 70

    # Step 3b: Preset tuning. This is the v30.3 stabilizer: different plan
    # families get different tolerance without forcing one universal filter.
    erode_size = max(1, min(5, erode_size + cfg["erode_adjust"]))
    short_dim_min = max(1, short_dim_min + cfg["short_adjust"])
    area_min = max(40, int(area_min * cfg["area_scale"]))
    long_dim_min = max(20, int(long_dim_min * cfg["long_scale"]))

    # Step 4: Initial CC filter - kill obvious text and tiny symbols.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    architectural = np.zeros_like(binary)
    n_kept_step3 = 0
    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]
        aspect = max(cw, ch) / max(min(cw, ch), 1)
        keep_candidate = (
            (aspect > 4 and area > max(50, int(100 * cfg["area_scale"]))) or
            (area > max(250, int(800 * cfg["area_scale"])) and max(cw, ch) > long_dim_min)
        )
        if keep_candidate:
            architectural[labels == i] = 255
            n_kept_step3 += 1

    # Step 5: Erode/dilate to suppress thinner non-wall content.
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_size, erode_size))
    eroded = cv2.erode(architectural, kernel_erode, iterations=1)
    walls_only = cv2.dilate(eroded, kernel_erode, iterations=1)

    # Step 6: Final wall-like component filter.
    num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(walls_only, connectivity=8)
    clean_mask = np.zeros_like(walls_only)
    n_kept_step6 = 0
    n_blobs_rejected = 0
    for i in range(1, num_labels2):
        x, y, cw, ch, area = stats2[i]
        aspect = max(cw, ch) / max(min(cw, ch), 1)
        short_dim = min(cw, ch)

        # Reject medium filled blobs, but preserve very large building-scale masses.
        if cfg["remove_solid"] and _is_blob_like(cw, ch, area) and area < 15000:
            n_blobs_rejected += 1
            continue

        keep_wall = (
            short_dim >= short_dim_min and
            ((aspect > 3 and area > area_min and max(cw, ch) > long_dim_min) or area > 4000)
        )
        if keep_wall:
            clean_mask[labels2 == i] = 255
            n_kept_step6 += 1

    # Step 7: Smart line-shaped gap repair.
    gap_size = max(6, int(max(8, median_thickness * 2) * cfg["gap_scale"]))
    clean_mask = _repair_small_gaps(clean_mask, max_gap=gap_size)

    # Step 8: Final island cleanup.
    num_final, labels_final, stats_final, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
    final_mask = np.zeros_like(clean_mask)
    n_final = 0
    for i in range(1, num_final):
        x, y, cw, ch, area = stats_final[i]
        if area >= cfg["final_min_area"]:
            final_mask[labels_final == i] = 255
            n_final += 1

    final_pixels = int(np.count_nonzero(final_mask))
    final_density = final_pixels / max(h * w, 1)
    stats_dict = {
        "version": "v30.3",
        "preset": preset,
        "median_thickness": median_thickness,
        "erode_size": erode_size,
        "threshold": cfg["threshold"],
        "raw_pixels": raw_pixels,
        "after_blob_removal_pixels": after_blob_removal,
        "after_text_filter": n_kept_step3,
        "after_wall_filter": n_kept_step6,
        "blobs_rejected": n_blobs_rejected,
        "final_components": n_final,
        "final_pixels": final_pixels,
        "final_density": round(final_density, 5),
        "recovery_used": False,
        "recovery_from": "",
    }

    # v30.3 Recovery: never leave the user with a blank mask preview. If a
    # normal preset discards everything, rerun conservatively and keep more lines.
    mask_failed = (
        stats_dict["final_components"] == 0 or
        stats_dict["after_wall_filter"] == 0 or
        final_density < 0.0005
    )
    if mask_failed and preset != "recovery":
        recovery_mask, recovery_stats = generate_clean_mask(img_bgr, debug=True, preset="recovery")
        if recovery_stats.get("final_pixels", 0) > final_pixels:
            recovery_stats["recovery_used"] = True
            recovery_stats["recovery_from"] = preset
            if debug:
                return recovery_mask, recovery_stats
            return recovery_mask

    if debug:
        return final_mask, stats_dict
    return final_mask


def format_mask_stats(mask_stats):
    """Build compact v30.3 debug text for the preview page."""
    recovery = ""
    if mask_stats.get("recovery_used"):
        recovery = f", RECOVERY from {mask_stats.get('recovery_from', 'unknown')}"
    return (
        f"version={mask_stats.get('version', 'v30.3')}, "
        f"preset={mask_stats.get('preset', 'balanced')}{recovery}, "
        f"threshold={mask_stats.get('threshold', 'n/a')}, "
        f"thickness={mask_stats.get('median_thickness', 'n/a')}px, "
        f"erode={mask_stats.get('erode_size', 'n/a')}x{mask_stats.get('erode_size', 'n/a')}, "
        f"text-filter: {mask_stats.get('after_text_filter', 0)} kept, "
        f"wall-filter: {mask_stats.get('after_wall_filter', 0)} kept, "
        f"blobs rejected: {mask_stats.get('blobs_rejected', 0)}, "
        f"final components: {mask_stats.get('final_components', 0)}, "
        f"density={mask_stats.get('final_density', 0)}"
    )

def render_mask_preview(clean_mask, original_img):
    """v30: Render the clean mask as a presentable image.

    Returns a 3-channel BGR image:
    - Background: very dark gray (like blueprint background)
    - Walls: white/light gray (clear and visible)
    """
    h, w = clean_mask.shape[:2]
    preview = np.full((h, w, 3), 32, dtype=np.uint8)  # dark blueprint bg

    # Paint walls in white
    preview[clean_mask > 0] = [240, 240, 240]

    return preview


def _warp_mask_to_aerial(mask, canvas_size, offset_x, offset_y, skew=0.10, y_scale=0.58):
    """Project a top-down mask into a simple aerial/Synchrony-like view."""
    matrix = np.float32([[1, skew, offset_x], [0, y_scale, offset_y]])
    return cv2.warpAffine(mask, matrix, canvas_size, flags=cv2.INTER_NEAREST, borderValue=0)


def _shift_mask(mask, dx, dy):
    h, w = mask.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)


def _extract_visual_wall_lines(mask, major_only=False):
    """Keep architectural-looking strokes for the v31 floorplan render."""
    h, w = mask.shape[:2]
    max_dim = max(h, w)
    line_len = max(18, int(max_dim * (0.055 if major_only else 0.028)))

    line_width = 1 if major_only else 2
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, line_width))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (line_width, line_len))
    horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_h, iterations=1)
    vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_v, iterations=1)
    lines = cv2.bitwise_or(horizontal, vertical)

    # Restore a readable wall body, then drop tiny fragments.
    restore_size = 2 if major_only else 4
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (restore_size, restore_size)), iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    filtered = np.zeros_like(lines)
    min_area = max(120 if major_only else 45, int(h * w * (0.00022 if major_only else 0.00008)))
    min_aspect = 4.0 if major_only else 2.2
    min_long = line_len * (1.8 if major_only else 1.2)
    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]
        aspect = max(cw, ch) / max(min(cw, ch), 1)
        if area >= min_area and aspect >= min_aspect and max(cw, ch) >= min_long:
            filtered[labels == i] = 255
    return filtered


def _extract_footprint_from_original(original_img):
    """Extract a broad floorplan footprint from the original isolated plan."""
    if original_img is None:
        return None
    if len(original_img.shape) == 3:
        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = original_img
    h, w = gray.shape[:2]

    # Use a softer threshold than the clean mask so faint exterior lines survive.
    _, binary = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)

    # Remove tiny text/speckle but keep long building geometry.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filtered = np.zeros_like(binary)
    min_area = max(25, int(h * w * 0.00003))
    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]
        if area >= min_area:
            filtered[labels == i] = 255

    # Close gaps aggressively enough to build the whole building footprint.
    max_dim = max(h, w)
    close_x = max(25, int(max_dim * 0.045))
    close_y = max(17, int(max_dim * 0.030))
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (close_x, close_y))
    footprint = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, k1, iterations=2)
    footprint = cv2.dilate(footprint, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)), iterations=1)

    contours, _ = cv2.findContours(footprint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Keep major content components. This avoids title notes while allowing
    # separated wings/edges from messy scans.
    clean = np.zeros_like(footprint)
    min_contour_area = max(1000, int(h * w * 0.01))
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_contour_area:
            continue
        epsilon = max(2.0, 0.003 * cv2.arcLength(cnt, True))
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(clean, [approx], -1, 255, thickness=-1)

    if np.count_nonzero(clean) == 0:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean, [largest], -1, 255, thickness=-1)
    return clean


def _footprint_to_extwall(footprint, accuracy="normal"):
    """Convert a footprint mask into one editable exterior wall polygon."""
    contours, _ = cv2.findContours(footprint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 500:
        return None
    # v32.1: preserve more exterior corners. The old simplification made the
    # footprint look clean, but not accurate enough to the real plan shape.
    ratio = 0.0018 if accuracy == "high" else 0.006
    epsilon = max(1.8, ratio * cv2.arcLength(largest, True))
    approx = cv2.approxPolyDP(largest, epsilon, True)
    points = []
    for p in approx.reshape(-1, 2):
        points.append({"x": int(p[0]), "y": int(p[1])})
    if len(points) < 3:
        x, y, w, h = cv2.boundingRect(largest)
        points = [
            {"x": int(x), "y": int(y)},
            {"x": int(x + w), "y": int(y)},
            {"x": int(x + w), "y": int(y + h)},
            {"x": int(x), "y": int(y + h)}
        ]
    return {
        "type": "extwall",
        "points": points,
        "closed": True,
        "detected": True,
        "source": "v32_trace"
    }


def _extract_original_orthogonal_lines(original_img, major_only=False):
    """Find long horizontal/vertical wall-like strokes from the original plan."""
    if original_img is None:
        return None
    if len(original_img.shape) == 3:
        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = original_img
    h, w = gray.shape[:2]
    max_dim = max(h, w)

    # Softer than mask threshold: captures office partitions that the mask can lose.
    _, binary = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY_INV)

    # Kill tiny text dots but preserve linework.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filtered = np.zeros_like(binary)
    min_area = max(12, int(h * w * 0.000015))
    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]
        aspect = max(cw, ch) / max(min(cw, ch), 1)
        if area >= min_area and (aspect >= 1.8 or max(cw, ch) >= max_dim * 0.012):
            filtered[labels == i] = 255

    line_len = max(24, int(max_dim * (0.050 if major_only else 0.030)))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len))
    horizontal = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel_h, iterations=1)
    vertical = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel_v, iterations=1)
    lines = cv2.bitwise_or(horizontal, vertical)
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    return lines


def _dedupe_trace_lines(lines, tolerance=10):
    """Drop near-duplicate collinear trace lines."""
    kept = []
    for line in lines:
        x1, y1, x2, y2 = line
        horizontal = abs(y1 - y2) <= abs(x1 - x2)
        duplicate = False
        for other in kept:
            ox1, oy1, ox2, oy2 = other
            other_horizontal = abs(oy1 - oy2) <= abs(ox1 - ox2)
            if horizontal != other_horizontal:
                continue
            if horizontal:
                same_axis = abs(y1 - oy1) <= tolerance
                overlap = min(max(x1, x2), max(ox1, ox2)) - max(min(x1, x2), min(ox1, ox2))
                min_len = min(abs(x2 - x1), abs(ox2 - ox1))
            else:
                same_axis = abs(x1 - ox1) <= tolerance
                overlap = min(max(y1, y2), max(oy1, oy2)) - max(min(y1, y2), min(oy1, oy2))
                min_len = min(abs(y2 - y1), abs(oy2 - oy1))
            if same_axis and overlap > min_len * 0.55:
                duplicate = True
                break
        if not duplicate:
            kept.append(line)
    return kept


def detect_editable_wall_trace(clean_mask, original_img=None, trace_mode="light"):
    """v32: Create an editable wall trace layer from mask/original sources.

    This is not room understanding. It produces conservative vector geometry:
    one exterior shape plus long orthogonal interior wall candidates that the
    user can quickly move/delete/repair in the editor.
    """
    if clean_mask is None:
        return []
    if len(clean_mask.shape) == 3:
        clean_mask = cv2.cvtColor(clean_mask, cv2.COLOR_BGR2GRAY)
    _, wall_mask = cv2.threshold(clean_mask, 1, 255, cv2.THRESH_BINARY)
    h, w = wall_mask.shape[:2]
    trace_mode = trace_mode or "light"
    trace_cfg = {
        "light": {"max_walls": 18, "min_len_ratio": 0.105, "major_only": True, "use_original": True},
        "medium": {"max_walls": 38, "min_len_ratio": 0.070, "major_only": False, "use_original": True},
        "detailed": {"max_walls": 85, "min_len_ratio": 0.045, "major_only": False, "use_original": True},
    }.get(trace_mode, {"max_walls": 18, "min_len_ratio": 0.105, "major_only": True, "use_original": True})

    elements = []
    footprint = None
    if original_img is not None:
        footprint = _extract_footprint_from_original(original_img)
        if footprint is not None and footprint.shape[:2] != wall_mask.shape[:2]:
            footprint = cv2.resize(footprint, (w, h), interpolation=cv2.INTER_NEAREST)
    if footprint is None:
        close_size = max(17, int(max(h, w) * 0.020))
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
        footprint = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, k, iterations=2)

    extwall = _footprint_to_extwall(footprint, accuracy="high")
    if extwall:
        elements.append(extwall)

    # Major interior wall candidates only. Keep this conservative, with a
    # fallback to normal line extraction if the plan is too faint/fragmented.
    major_lines_mask = _extract_visual_wall_lines(wall_mask, major_only=trace_cfg["major_only"])
    if np.count_nonzero(major_lines_mask) < max(150, int(np.count_nonzero(wall_mask) * 0.015)):
        major_lines_mask = _extract_visual_wall_lines(wall_mask, major_only=False)
    if trace_cfg["use_original"] and original_img is not None:
        original_lines = _extract_original_orthogonal_lines(original_img, major_only=trace_cfg["major_only"])
        if original_lines is not None:
            if original_lines.shape[:2] != wall_mask.shape[:2]:
                original_lines = cv2.resize(original_lines, (w, h), interpolation=cv2.INTER_NEAREST)
            major_lines_mask = cv2.bitwise_or(major_lines_mask, original_lines)
    shell = np.zeros_like(wall_mask)
    if footprint is not None:
        shell_size = max(9, int(max(h, w) * 0.012))
        shell_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (shell_size, shell_size))
        shell = cv2.subtract(footprint, cv2.erode(footprint, shell_kernel, iterations=1))
        major_lines_mask = cv2.bitwise_and(
            major_lines_mask,
            cv2.bitwise_not(cv2.dilate(shell, shell_kernel, iterations=1))
        )

    raw_lines = detect_walls_hough(
        major_lines_mask,
        min_line_length=max(55, int(max(h, w) * trace_cfg["min_len_ratio"])),
        max_line_gap=max(12, int(max(h, w) * 0.018))
    )
    snapped = snap_to_orthogonal(raw_lines, angle_tolerance=7)
    merged = merge_collinear_lines(snapped, distance_threshold=max(10, int(max(h, w) * 0.012)))

    min_len = max(70, int(max(h, w) * trace_cfg["min_len_ratio"]))
    max_walls = trace_cfg["max_walls"]
    wall_lines = []
    for line in merged:
        x1, y1, x2, y2 = [int(v) for v in line]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_len:
            continue
        wall_lines.append((length, [x1, y1, x2, y2]))
    wall_lines.sort(reverse=True, key=lambda item: item[0])
    deduped = _dedupe_trace_lines([line for _length, line in wall_lines], tolerance=max(8, int(max(h, w) * 0.008)))

    for line in deduped[:max_walls]:
        x1, y1, x2, y2 = line
        elements.append({
            "type": "intwall",
            "points": [{"x": x1, "y": y1}, {"x": x2, "y": y2}],
            "detected": True,
            "source": "v32_trace"
        })

    # Door/opening candidates from gaps between collinear traced wall segments.
    # This is intentionally capped and conservative: doors are helpful hints,
    # not a claim that the app understands every opening.
    traced_lines = []
    for el in elements:
        if el.get("type") == "intwall" and el.get("points") and len(el["points"]) >= 2:
            p1, p2 = el["points"][0], el["points"][1]
            traced_lines.append([int(p1["x"]), int(p1["y"]), int(p2["x"]), int(p2["y"])])
    doors = detect_doors_in_walls(traced_lines, min_gap=max(18, int(max(h, w) * 0.015)), max_gap=max(55, int(max(h, w) * 0.055)))
    for door in doors[:20]:
        door["detected"] = True
        door["source"] = "v32_2_gap"
        elements.append(door)

    return elements


def render_floorplan_shape_base(clean_mask, original_img=None, source_mode="hybrid"):
    """v31: Render a first-pass BAS-style floorplan base from the clean mask.

    This is intentionally visual-first, not BIM reconstruction. It uses the
    approved mask as a trace reference, creates a broad footprint for the floor,
    draws a clipped grid, and lifts the wall pixels into a light aerial view.
    """
    if clean_mask is None:
        return None
    if len(clean_mask.shape) == 3:
        clean_mask = cv2.cvtColor(clean_mask, cv2.COLOR_BGR2GRAY)
    _, wall_mask = cv2.threshold(clean_mask, 1, 255, cv2.THRESH_BINARY)
    h, w = wall_mask.shape[:2]
    source_mode = source_mode or "hybrid"

    # Build a broad footprint from the wall trace. This keeps the same overall
    # plan shape while avoiding room/area inference.
    close_size = max(17, int(max(h, w) * 0.018))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
    footprint = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    footprint = cv2.dilate(footprint, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=1)

    contours, _ = cv2.findContours(footprint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    footprint_clean = np.zeros_like(footprint)
    min_area = max(500, int(h * w * 0.002))
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            epsilon = max(2.0, 0.0025 * cv2.arcLength(cnt, True))
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(footprint_clean, [approx], -1, 255, thickness=-1)
    if np.count_nonzero(footprint_clean) == 0:
        footprint_clean = footprint

    original_footprint = None
    if original_img is not None and source_mode in ("hybrid", "original"):
        original_footprint = _extract_footprint_from_original(original_img)
        if original_footprint is not None and original_footprint.shape[:2] != wall_mask.shape[:2]:
            original_footprint = cv2.resize(original_footprint, (w, h), interpolation=cv2.INTER_NEAREST)

    if source_mode == "original" and original_footprint is not None:
        footprint_clean = original_footprint
    elif source_mode == "hybrid" and original_footprint is not None:
        mask_area = np.count_nonzero(footprint_clean)
        original_area = np.count_nonzero(original_footprint)
        # If the mask footprint is missing big chunks, trust the original
        # footprint. Otherwise union them so weak exterior edges are restored.
        if original_area > 0 and mask_area < original_area * 0.72:
            footprint_clean = original_footprint
        else:
            footprint_clean = cv2.bitwise_or(footprint_clean, original_footprint)

    skew = 0.10
    y_scale = 0.58
    pad = 70
    out_w = int(w + h * skew + pad * 2)
    out_h = int(h * y_scale + pad * 2 + 70)
    canvas_size = (out_w, out_h)
    offset_x = pad
    offset_y = pad + 38

    # Exterior shell = only the outside band of the footprint. This avoids
    # turning noisy interior mask blobs into full-height walls.
    shell_kernel_size = max(9, int(max(h, w) * 0.010))
    shell_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (shell_kernel_size, shell_kernel_size))
    eroded_footprint = cv2.erode(footprint_clean, shell_kernel, iterations=1)
    exterior_shell = cv2.subtract(footprint_clean, eroded_footprint)

    floor_proj = _warp_mask_to_aerial(footprint_clean, canvas_size, offset_x, offset_y, skew, y_scale)
    shell_proj = _warp_mask_to_aerial(exterior_shell, canvas_size, offset_x, offset_y, skew, y_scale)
    visual_walls = _extract_visual_wall_lines(wall_mask, major_only=True)
    if np.count_nonzero(visual_walls) < max(100, int(np.count_nonzero(wall_mask) * 0.05)):
        visual_walls = _extract_visual_wall_lines(wall_mask, major_only=False)

    interior_walls = cv2.bitwise_and(visual_walls, cv2.bitwise_not(cv2.dilate(exterior_shell, shell_kernel, iterations=1)))
    interior_proj = _warp_mask_to_aerial(interior_walls, canvas_size, offset_x, offset_y, skew, y_scale)
    interior_proj = cv2.dilate(interior_proj, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    shell_proj = cv2.dilate(shell_proj, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    canvas = np.full((out_h, out_w, 3), 248, dtype=np.uint8)

    # Soft shadow under the building.
    shadow = _shift_mask(floor_proj, 18, 28)
    canvas[shadow > 0] = [205, 205, 205]

    # Floor fill.
    canvas[floor_proj > 0] = [218, 218, 220]

    # Clipped floor grid.
    grid = np.zeros_like(floor_proj)
    grid_step = 22
    for x in range(0, out_w, grid_step):
        cv2.line(grid, (x, 0), (x, out_h), 255, 1)
    for y in range(0, out_h, grid_step):
        cv2.line(grid, (0, y), (out_w, y), 255, 1)
    grid = cv2.bitwise_and(grid, floor_proj)
    canvas[grid > 0] = [194, 194, 198]

    # Footprint edge.
    edge_contours, _ = cv2.findContours(floor_proj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, edge_contours, -1, (118, 128, 145), 2, lineType=cv2.LINE_AA)

    # Exterior shell: taller extrusion, like the visible outside walls in BAS graphics.
    wall_height = 30
    for z in range(0, wall_height + 1, 3):
        shifted = _shift_mask(shell_proj, 0, -z)
        tone = 168 + int(45 * (z / max(wall_height, 1)))
        canvas[shifted > 0] = [tone, tone, min(245, tone + 8)]

    wall_top = _shift_mask(shell_proj, 0, -wall_height)
    wall_top = cv2.dilate(wall_top, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    canvas[wall_top > 0] = [246, 246, 250]
    wall_edges = cv2.Canny(wall_top, 50, 150)
    canvas[wall_edges > 0] = [130, 132, 140]

    # Interior walls: lower, cleaner raised strokes instead of chunky blocks.
    interior_height = 12
    for z in range(0, interior_height + 1, 4):
        shifted = _shift_mask(interior_proj, 0, -z)
        tone = 178 + int(34 * (z / max(interior_height, 1)))
        canvas[shifted > 0] = [tone, tone, min(242, tone + 6)]
    interior_top = _shift_mask(interior_proj, 0, -interior_height)
    canvas[interior_top > 0] = [239, 239, 242]
    interior_edges = cv2.Canny(interior_top, 50, 150)
    canvas[interior_edges > 0] = [150, 150, 158]

    return canvas


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


def detect_doors_in_walls(lines, min_gap=20, max_gap=70):
    """Detect doors as small gaps between collinear wall segments.
    Returns list of door dicts with start/end points."""
    doors = []
    if not lines:
        return doors

    # Separate horizontal and vertical
    horizontal = [l for l in lines if l[1] == l[3]]
    vertical = [l for l in lines if l[0] == l[2]]

    # For horizontal lines: find pairs with same Y and small X gap
    horizontal.sort(key=lambda l: (l[1], l[0]))
    for i in range(len(horizontal) - 1):
        l1 = horizontal[i]
        l2 = horizontal[i + 1]
        # Same Y level (within 8px)
        if abs(l1[1] - l2[1]) > 8:
            continue
        # Gap between end of l1 and start of l2
        gap = l2[0] - l1[2]
        if min_gap <= gap <= max_gap:
            # This is likely a door
            doors.append({
                "type": "door",
                "points": [
                    {"x": int(l1[2]), "y": int((l1[1] + l2[1]) // 2)},
                    {"x": int(l2[0]), "y": int((l1[1] + l2[1]) // 2)}
                ],
                "detected": True
            })

    # For vertical lines: find pairs with same X and small Y gap
    vertical.sort(key=lambda l: (l[0], l[1]))
    for i in range(len(vertical) - 1):
        l1 = vertical[i]
        l2 = vertical[i + 1]
        if abs(l1[0] - l2[0]) > 8:
            continue
        gap = l2[1] - l1[3]
        if min_gap <= gap <= max_gap:
            doors.append({
                "type": "door",
                "points": [
                    {"x": int((l1[0] + l2[0]) // 2), "y": int(l1[3])},
                    {"x": int((l1[0] + l2[0]) // 2), "y": int(l2[1])}
                ],
                "detected": True
            })

    return doors


def measure_line_thickness(binary_img, x1, y1, x2, y2, max_check=10):
    """Measure perpendicular thickness of a line in the binary image.
    Returns average thickness in pixels."""
    h, w = binary_img.shape[:2]
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length < 1:
        return 0
    # Perpendicular direction (unit vector)
    nx = -dy / length
    ny = dx / length

    # Sample 5 points along the line
    samples = []
    for t in [0.2, 0.35, 0.5, 0.65, 0.8]:
        px = int(x1 + dx * t)
        py = int(y1 + dy * t)
        # Measure how far in perpendicular direction we have white pixels
        thickness_pos = 0
        for k in range(max_check):
            tx = int(px + nx * k)
            ty = int(py + ny * k)
            if 0 <= tx < w and 0 <= ty < h and binary_img[ty, tx] > 0:
                thickness_pos = k + 1
            else:
                break
        thickness_neg = 0
        for k in range(max_check):
            tx = int(px - nx * k)
            ty = int(py - ny * k)
            if 0 <= tx < w and 0 <= ty < h and binary_img[ty, tx] > 0:
                thickness_neg = k + 1
            else:
                break
        samples.append(thickness_pos + thickness_neg)

    samples.sort()
    # Return median to reduce outlier impact
    return samples[len(samples) // 2]


def filter_lines_by_thickness(lines, binary_img, min_thickness=3):
    """Keep only lines whose underlying drawing is thick (walls), not thin (ducts/text)."""
    filtered = []
    for l in lines:
        x1, y1, x2, y2 = l
        thickness = measure_line_thickness(binary_img, x1, y1, x2, y2)
        if thickness >= min_thickness:
            filtered.append(l)
    return filtered


def filter_dense_zones(lines, img_shape, cell_size=60, max_lines_per_cell=6):
    """Detect dense areas (HVAC symbols) and remove lines passing through them."""
    h, w = img_shape[:2]
    if not lines:
        return lines

    cells_h = (h // cell_size) + 1
    cells_w = (w // cell_size) + 1
    grid = [[0 for _ in range(cells_w)] for _ in range(cells_h)]

    # Count line midpoints per cell
    midpoints = []
    for l in lines:
        mx = (l[0] + l[2]) // 2
        my = (l[1] + l[3]) // 2
        cx = mx // cell_size
        cy = my // cell_size
        if 0 <= cy < cells_h and 0 <= cx < cells_w:
            grid[cy][cx] += 1
        midpoints.append((cx, cy))

    # Mark dense cells
    dense_cells = set()
    for cy in range(cells_h):
        for cx in range(cells_w):
            if grid[cy][cx] > max_lines_per_cell:
                dense_cells.add((cx, cy))

    # Remove lines whose midpoint falls in dense cell
    filtered = []
    for i, l in enumerate(lines):
        if midpoints[i] not in dense_cells:
            filtered.append(l)
    return filtered


def filter_dimension_pairs(lines, max_parallel_dist=5):
    """Remove pairs of very close parallel lines (likely dimension lines/double lines)."""
    horizontal = [l for l in lines if l[1] == l[3]]
    vertical = [l for l in lines if l[0] == l[2]]
    to_remove = set()

    # Find horizontal pairs that are too close (likely dimension lines)
    for i in range(len(horizontal)):
        for j in range(i + 1, len(horizontal)):
            l1 = horizontal[i]
            l2 = horizontal[j]
            ydiff = abs(l1[1] - l2[1])
            if 1 <= ydiff <= max_parallel_dist:
                # Check x overlap
                ox1 = max(l1[0], l2[0])
                ox2 = min(l1[2], l2[2])
                if ox2 > ox1:
                    to_remove.add(tuple(l1))
                    to_remove.add(tuple(l2))

    for i in range(len(vertical)):
        for j in range(i + 1, len(vertical)):
            l1 = vertical[i]
            l2 = vertical[j]
            xdiff = abs(l1[0] - l2[0])
            if 1 <= xdiff <= max_parallel_dist:
                oy1 = max(l1[1], l2[1])
                oy2 = min(l1[3], l2[3])
                if oy2 > oy1:
                    to_remove.add(tuple(l1))
                    to_remove.add(tuple(l2))

    return [l for l in lines if tuple(l) not in to_remove]


def reconstruct_rooms_perimeter(lines, img_shape):
    """Try to reconstruct the actual building perimeter from longest exterior walls.
    Returns ordered list of points or None if can't reconstruct."""
    if not lines or len(lines) < 4:
        return None

    h, w = img_shape[:2]

    # Sort lines by length descending; longest are most likely exterior walls
    lines_sorted = sorted(lines, key=lambda l: -math.hypot(l[2] - l[0], l[3] - l[1]))

    # Take top 30% of longest lines as exterior candidates
    n_top = max(8, len(lines_sorted) // 3)
    top_lines = lines_sorted[:n_top]

    horizontal = [l for l in top_lines if l[1] == l[3]]
    vertical = [l for l in top_lines if l[0] == l[2]]

    if not horizontal or not vertical:
        return None

    # Find topmost and bottommost long horizontal lines
    horizontal.sort(key=lambda l: l[1])
    top_h = horizontal[0]
    bottom_h = horizontal[-1]

    vertical.sort(key=lambda l: l[0])
    left_v = vertical[0]
    right_v = vertical[-1]

    min_x = left_v[0]
    max_x = right_v[0]
    min_y = top_h[1]
    max_y = bottom_h[1]

    # Validate the rectangle has meaningful size
    if (max_x - min_x) < w * 0.3 or (max_y - min_y) < h * 0.3:
        return None

    return [
        {"x": int(min_x), "y": int(min_y)},
        {"x": int(max_x), "y": int(min_y)},
        {"x": int(max_x), "y": int(max_y)},
        {"x": int(min_x), "y": int(max_y)}
    ]


def chain_walls_by_endpoints(lines, endpoint_tolerance=20):
    """v29: Connect walls whose endpoints are very close.
    Merges chains of collinear walls that have small gaps."""
    if not lines:
        return []

    # Snap close endpoints to a shared point
    all_points = []
    for li, l in enumerate(lines):
        all_points.append({"x": l[0], "y": l[1], "line": li, "end": 0})
        all_points.append({"x": l[2], "y": l[3], "line": li, "end": 1})

    # Group close points
    used = [False] * len(all_points)
    for i in range(len(all_points)):
        if used[i]:
            continue
        cluster = [i]
        for j in range(i + 1, len(all_points)):
            if used[j]:
                continue
            dx = all_points[i]["x"] - all_points[j]["x"]
            dy = all_points[i]["y"] - all_points[j]["y"]
            if math.hypot(dx, dy) < endpoint_tolerance:
                cluster.append(j)
                used[j] = True
        # Average cluster position
        cx = sum(all_points[k]["x"] for k in cluster) // len(cluster)
        cy = sum(all_points[k]["y"] for k in cluster) // len(cluster)
        for k in cluster:
            all_points[k]["x"] = cx
            all_points[k]["y"] = cy
        used[i] = True

    # Rebuild lines from snapped endpoints
    chained = []
    for li, l in enumerate(lines):
        p0 = next(p for p in all_points if p["line"] == li and p["end"] == 0)
        p1 = next(p for p in all_points if p["line"] == li and p["end"] == 1)
        if abs(p0["x"] - p1["x"]) > 2 or abs(p0["y"] - p1["y"]) > 2:
            chained.append([p0["x"], p0["y"], p1["x"], p1["y"]])
    return chained


def detect_main_corridor(lines, img_shape, min_length_ratio=0.3):
    """v29: Identify the main horizontal/vertical corridor spine.
    v29: Less strict ratio (30%) and gap range (30-300px)."""
    h, w = img_shape[:2]
    if not lines:
        return None

    horizontal = sorted([l for l in lines if l[1] == l[3]], key=lambda l: l[2] - l[0], reverse=True)
    vertical = sorted([l for l in lines if l[0] == l[2]], key=lambda l: l[3] - l[1], reverse=True)

    best_corridor = None
    best_score = 0

    # Look for horizontal corridor - v29: take top 12 lines, allow shorter corridors
    min_len_h = w * min_length_ratio
    for i, l1 in enumerate(horizontal[:12]):
        len1 = l1[2] - l1[0]
        if len1 < min_len_h:
            continue
        for l2 in horizontal[i+1:12]:
            len2 = l2[2] - l2[0]
            if len2 < min_len_h:
                continue
            gap = abs(l1[1] - l2[1])
            # v29: wider corridor gap range (30-300 px)
            if 30 < gap < 300:
                ox1 = max(l1[0], l2[0])
                ox2 = min(l1[2], l2[2])
                overlap = ox2 - ox1
                # v29: 25% overlap instead of 30%
                if overlap > w * 0.25:
                    score = overlap - gap * 1.5
                    if score > best_score:
                        best_score = score
                        y_top = min(l1[1], l2[1])
                        y_bot = max(l1[1], l2[1])
                        best_corridor = {
                            "type": "corridor",
                            "orientation": "horizontal",
                            "x1": ox1, "x2": ox2,
                            "y1": y_top, "y2": y_bot
                        }

    # Look for vertical corridor
    min_len_v = h * min_length_ratio
    for i, l1 in enumerate(vertical[:12]):
        len1 = l1[3] - l1[1]
        if len1 < min_len_v:
            continue
        for l2 in vertical[i+1:12]:
            len2 = l2[3] - l2[1]
            if len2 < min_len_v:
                continue
            gap = abs(l1[0] - l2[0])
            if 30 < gap < 300:
                oy1 = max(l1[1], l2[1])
                oy2 = min(l1[3], l2[3])
                overlap = oy2 - oy1
                if overlap > h * 0.25:
                    score = overlap - gap * 1.5
                    if score > best_score:
                        best_score = score
                        x_left = min(l1[0], l2[0])
                        x_right = max(l1[0], l2[0])
                        best_corridor = {
                            "type": "corridor",
                            "orientation": "vertical",
                            "x1": x_left, "x2": x_right,
                            "y1": oy1, "y2": oy2
                        }

    return best_corridor


def segment_rooms_via_floodfill(walls, img_shape, min_room_area=2000):
    """v29: Use flood fill to find enclosed rooms.
    Returns list of room bounding boxes."""
    h, w = img_shape[:2]
    if not walls:
        return []

    # Render walls on a fresh mask
    mask = np.zeros((h, w), dtype=np.uint8)
    for l in walls:
        cv2.line(mask, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), 255, 3)

    # Find connected white components (potential rooms / open areas)
    # Invert so rooms are foreground
    inverted = cv2.bitwise_not(mask)

    # Build a border so flood-fill from outside doesn't escape
    bordered = cv2.copyMakeBorder(inverted, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bordered, connectivity=4)

    rooms = []
    # Identify background label (the largest/outermost component touching the border)
    bh, bw = bordered.shape
    bg_label = 0
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]
        if x <= 1 or y <= 1 or x + ww >= bw - 1 or y + hh >= bh - 1:
            if area > stats[bg_label][4] if bg_label > 0 else True:
                bg_label = i

    for i in range(1, num_labels):
        if i == bg_label:
            continue
        x, y, ww, hh, area = stats[i]
        if area < min_room_area:
            continue
        # Account for border offset
        rooms.append({
            "x": int(x - 2),
            "y": int(y - 2),
            "w": int(ww),
            "h": int(hh),
            "area": int(area)
        })

    return rooms


def reconstruct_perimeter_from_rooms(rooms):
    """v29: Build the building outer footprint from the union of detected rooms.
    Returns a simple bounding polygon (axis-aligned)."""
    if not rooms:
        return None

    min_x = min(r["x"] for r in rooms)
    max_x = max(r["x"] + r["w"] for r in rooms)
    min_y = min(r["y"] for r in rooms)
    max_y = max(r["y"] + r["h"] for r in rooms)

    return [
        {"x": int(min_x), "y": int(min_y)},
        {"x": int(max_x), "y": int(min_y)},
        {"x": int(max_x), "y": int(max_y)},
        {"x": int(min_x), "y": int(max_y)}
    ]


def detect_architecture(image_path):
    """v29: Architectural reconstruction - balanced filtering + corridor + rooms + chaining."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    # === STEP 0: Smart Floorplan Isolation ===
    isolated, crop_box = smart_isolate_floorplan(img)
    cv2.imwrite(ISOLATED_IMAGE_PATH, isolated)
    cv2.imwrite(image_path, isolated)

    img_h, img_w = isolated.shape[:2]
    gray = cv2.cvtColor(isolated, cv2.COLOR_BGR2GRAY)

    # === STEP 1: Binary for thickness ===
    _, binary_full = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # === STEP 2: Remove text ===
    binary_clean = remove_text_from_plan(gray)
    cv2.imwrite(CLEAN_IMAGE_PATH, binary_clean)

    # === STEP 3: Hough detection ===
    raw_lines = detect_walls_hough(binary_clean)
    n_raw = len(raw_lines)

    if n_raw == 0:
        return {
            "image_width": img_w, "image_height": img_h, "elements": [],
            "stats": {"lines_raw": 0, "after_filter": 0, "rooms": 0,
                      "corridor": "none", "chained": 0, "final_walls": 0,
                      "doors": 0, "crop": crop_box}
        }

    # === STEP 4: Snap orthogonal ===
    snapped = snap_to_orthogonal(raw_lines, angle_tolerance=6)

    # === STEP 5: First merge collinear ===
    merged = merge_collinear_lines(snapped, distance_threshold=18)

    # === STEP 6 (v29 REBALANCED): Length filter - MIN 60 (was 80) ===
    merged = [l for l in merged if math.hypot(l[2] - l[0], l[3] - l[1]) > 60]

    # === STEP 7 (v29 REBALANCED): Thickness filter - MIN 2 (was 3) ===
    thick_lines = filter_lines_by_thickness(merged, binary_full, min_thickness=2)

    # === STEP 8 (v29 REBALANCED): Density - MAX 8 per cell (was 5) ===
    after_density = filter_dense_zones(thick_lines, (img_h, img_w),
                                        cell_size=70, max_lines_per_cell=8)

    # === STEP 9: Remove dimension pairs ===
    after_dims = filter_dimension_pairs(after_density, max_parallel_dist=6)

    # === STEP 10: Aggressive merge again (close text gaps) ===
    after_merge2 = merge_collinear_lines(after_dims, distance_threshold=35)
    n_filtered = len(after_merge2)

    # === STEP 11 (v29 NEW): Chain walls by endpoints ===
    # v29: Slightly more permissive chaining (25px instead of 20)
    chained_walls = chain_walls_by_endpoints(after_merge2, endpoint_tolerance=25)
    n_chained = len(chained_walls)

    # === STEP 12 (v29 NEW): Detect main corridor ===
    corridor = detect_main_corridor(chained_walls, (img_h, img_w))
    corridor_label = corridor["orientation"] if corridor else "none"

    # === STEP 13 (v29 NEW): Room segmentation via flood fill ===
    # v29: Lower min area (1500 instead of 2500) to catch more small rooms
    rooms = segment_rooms_via_floodfill(chained_walls, (img_h, img_w), min_room_area=1500)
    n_rooms = len(rooms)

    # === STEP 14 (v29 NEW): Reconstruct perimeter from rooms ===
    ext_polygon = None
    if rooms:
        ext_polygon = reconstruct_perimeter_from_rooms(rooms)
    if ext_polygon is None:
        # Fallback to wall-based perimeter
        ext_polygon = reconstruct_rooms_perimeter(chained_walls, (img_h, img_w))
    if ext_polygon is None:
        # Final fallback: footprint bbox
        footprint = find_building_footprint(chained_walls, (img_h, img_w))
        ext_polygon = build_exterior_polygon(footprint) if footprint else None

    # === STEP 15: Classify exterior vs interior ===
    if ext_polygon:
        xs = [p["x"] for p in ext_polygon]
        ys = [p["y"] for p in ext_polygon]
        footprint = {"min_x": min(xs), "max_x": max(xs),
                     "min_y": min(ys), "max_y": max(ys)}
    else:
        footprint = None

    exterior_lines, interior_lines = classify_exterior_vs_interior(chained_walls, footprint)

    # === STEP 16: Build elements ===
    elements = []
    if ext_polygon:
        elements.append({
            "type": "extwall",
            "points": ext_polygon,
            "closed": True,
            "detected": True
        })

    for line in interior_lines:
        elements.append({
            "type": "intwall",
            "points": [
                {"x": int(line[0]), "y": int(line[1])},
                {"x": int(line[2]), "y": int(line[3])}
            ],
            "detected": True
        })

    # === STEP 17: Detect doors ===
    doors = detect_doors_in_walls(chained_walls)
    elements.extend(doors)

    final_walls_count = sum(1 for e in elements if e.get("type") in ("extwall", "intwall"))

    return {
        "image_width": img_w,
        "image_height": img_h,
        "elements": elements,
        "stats": {
            "lines_raw": n_raw,
            "after_filter": n_filtered,
            "chained": n_chained,
            "corridor": corridor_label,
            "rooms": n_rooms,
            "final_walls": final_walls_count,
            "doors": len(doors),
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
<h1>BAS Generator v29</h1>
<p class="sub">Private Access</p>
<form method="POST" action="/login">
<input type="password" name="password" placeholder="Enter password" required autofocus>
<button type="submit">Login</button>
</form>
{% if error %}<div class="error">Invalid password. Try again.</div>{% endif %}
<div class="footer">Made by Paolo V. R.</div>
</div></body></html>'''


HOME_PAGE = '''<!DOCTYPE html>
<html><head><title>BAS Generator v29</title>
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
<h1>BAS Generator v33 <span class="badge">ROOM TOOLBOX</span></h1>
<p class="sub">Smart trace + clean floorplan shape workflow</p>

<div class="tip">
<b>v33 NEW:</b> Room Rect toolbox, short-line cleanup, exterior-only cleanup, and faster manual floorplan building.
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
<button type="button" class="option-btn" id="maskBtn" onclick="setMode('mask')" style="border:2px solid #22d3ee;">
&#10024; Mask Preview <span style="background:#22d3ee;color:#000;padding:1px 5px;border-radius:4px;font-size:9px;font-weight:700;">NEW v33</span>
<small>Mask first, then Floorplan Base</small>
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
    document.getElementById('maskBtn').classList.toggle('active', mode==='mask');
    document.getElementById('modeInput').value = mode;
}
</script>
</body></html>'''


MASK_PREVIEW_PAGE = '''<!DOCTYPE html>
<html><head><title>Mask Preview v30.3</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{background:#0d0f14;color:white;font-family:'Segoe UI',Arial,sans-serif;min-height:100vh;padding:18px;}
.wrap{max-width:1700px;margin:0 auto;}
.head{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;padding:14px 22px;background:#1a1d24;border-radius:12px;border:1px solid #22d3ee44;}
.head h1{color:#22d3ee;font-size:22px;}
.head .sub{color:#a0aec0;font-size:13px;margin-top:4px;}
.head .badge{background:#22d3ee;color:#000;padding:3px 9px;border-radius:5px;font-size:11px;font-weight:700;margin-left:8px;}
.actions{display:flex;gap:8px;}
.btn{padding:10px 18px;border-radius:8px;border:none;font-size:14px;font-weight:600;cursor:pointer;}
.btn-green{background:#16a34a;color:white;}
.btn-orange{background:#ea580c;color:white;}
.btn-gray{background:#333;color:white;}
.btn-purple{background:#7c3aed;color:white;}
.btn:hover{transform:translateY(-1px);}
.compare{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px;}
.col{background:#1a1d24;border-radius:12px;padding:14px;border:1px solid #2a2f3a;}
.col h2{color:#a0aec0;font-size:14px;margin-bottom:10px;text-transform:uppercase;letter-spacing:1px;}
.col img{width:100%;border-radius:8px;border:1px solid #2a2f3a;display:block;}
.col.preview{border-color:#22d3ee66;}
.col.preview h2{color:#22d3ee;}
.info{background:#1a1d24;border:1px solid #facc1544;border-radius:12px;padding:14px 22px;color:#e5e5e5;font-size:14px;line-height:1.6;}
.info b{color:#facc15;}
.foot{text-align:center;color:#666;margin-top:18px;font-size:12px;}
</style>
</head><body>
<div class="wrap">

<div class="head">
<div>
<h1>&#10024; Architectural Mask Preview <span class="badge">v33</span></h1>
<div class="sub">Compare original vs. clean architectural mask before editing</div>
</div>
<div class="actions">
<form method="GET" action="/" style="display:inline;"><button type="submit" class="btn btn-gray">&larr; Back</button></form>
<form method="POST" action="/mask-retry" style="display:inline;"><input type="hidden" name="mask_preset" value="balanced"><button type="submit" class="btn btn-orange">&#8635; Balanced</button></form>
<form method="POST" action="/mask-retry" style="display:inline;"><input type="hidden" name="mask_preset" value="mechanical_dense"><button type="submit" class="btn btn-orange">Mechanical Dense</button></form>
<form method="POST" action="/mask-retry" style="display:inline;"><input type="hidden" name="mask_preset" value="thin_scan"><button type="submit" class="btn btn-orange">Thin Scan</button></form>
<form method="POST" action="/floorplan-base" style="display:inline;"><button type="submit" class="btn btn-gray">Floorplan Base</button></form>
<form method="POST" action="/trace-editor" style="display:inline;"><input type="hidden" name="trace_mode" value="light"><button type="submit" class="btn btn-gray">Trace Light</button></form>
<form method="POST" action="/trace-editor" style="display:inline;"><input type="hidden" name="trace_mode" value="medium"><button type="submit" class="btn btn-gray">Trace Medium</button></form>
<form method="POST" action="/trace-editor" style="display:inline;"><input type="hidden" name="trace_mode" value="detailed"><button type="submit" class="btn btn-gray">Trace Detailed</button></form>
<form method="POST" action="/claude-review" style="display:inline;"><button type="submit" class="btn btn-purple">Claude Review</button></form>
<form method="POST" action="/mask-approve" style="display:inline;"><button type="submit" class="btn btn-green">&#10003; Approve & Continue to Editor &rarr;</button></form>
</div>
</div>

<div class="info">
<b>How this works:</b> The app filtered out HVAC ducts, text, dimension lines, and symbols.
The clean mask on the right shows ONLY the architectural walls and building shape.
If it looks good, click <b>Approve</b> to enter the editor with this as a reference layer.
If too much was removed, use a preset below to re-process with a different filter style.
</div>

<div style="background:#1a1d24;border:1px solid #2a2f3a;border-radius:12px;padding:10px 22px;margin-top:10px;font-family:'Courier New',monospace;font-size:12px;color:#94a3b8;">
<b style="color:#22d3ee;">Debug stats:</b> {{ stats_text }}
</div>

<div class="compare">
<div class="col">
<h2>&#128247; Original Plan (Isolated)</h2>
<img src="data:image/png;base64,{{ original_b64 }}" alt="Original">
</div>
<div class="col preview">
<h2>&#10024; Clean Architectural Mask</h2>
<img src="data:image/png;base64,{{ preview_b64 }}" alt="Clean Mask">
</div>
</div>

<div class="foot">Made by Paolo V. R.</div>

</div>
</body></html>'''


CLAUDE_REVIEW_PAGE = '''<!DOCTYPE html>
<html><head><title>Claude Review v34</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{background:#0d0f14;color:white;font-family:'Segoe UI',Arial,sans-serif;min-height:100vh;padding:18px;}
.wrap{max-width:1100px;margin:0 auto;}
.head{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;padding:16px 22px;background:#1a1d24;border-radius:12px;border:1px solid #7c3aed66;}
h1{color:#c4b5fd;font-size:22px;}
.sub{color:#a0aec0;font-size:13px;margin-top:4px;}
.actions{display:flex;gap:8px;flex-wrap:wrap;}
.btn{padding:10px 18px;border-radius:8px;border:none;font-size:14px;font-weight:700;cursor:pointer;text-decoration:none;display:inline-block;}
.btn-gray{background:#333;color:white;}
.btn-blue{background:#2563eb;color:white;}
.btn-green{background:#16a34a;color:white;}
.panel{background:#151922;border:1px solid #2a2f3a;border-radius:12px;padding:18px 22px;line-height:1.55;}
.panel h2{font-size:13px;letter-spacing:1px;text-transform:uppercase;color:#c4b5fd;margin-bottom:10px;}
.review{white-space:pre-wrap;color:#e5e7eb;font-size:15px;}
.hint{background:#1f2937;border-left:4px solid #7c3aed;border-radius:6px;padding:10px 12px;margin-bottom:14px;color:#cbd5e1;font-size:13px;}
.foot{text-align:center;color:#666;margin-top:18px;font-size:12px;}
</style></head><body>
<div class="wrap">
<div class="head">
<div>
<h1>Claude Review <span style="font-size:12px;background:#7c3aed;color:white;padding:3px 8px;border-radius:5px;">v34 advisor</span></h1>
<div class="sub">AI reviewer for Smart Trace + Smart Cleanup decisions</div>
</div>
<div class="actions">
<form method="POST" action="/mask-retry" style="display:inline;"><input type="hidden" name="mask_preset" value="balanced"><button type="submit" class="btn btn-gray">Back to Mask</button></form>
<form method="POST" action="/floorplan-base" style="display:inline;"><button type="submit" class="btn btn-blue">Floorplan Base</button></form>
<form method="POST" action="/trace-editor" style="display:inline;"><input type="hidden" name="trace_mode" value="light"><button type="submit" class="btn btn-green">Trace Light</button></form>
</div>
</div>

<div class="hint">
This does not replace the trace engine. It reviews the current original/mask/base images and recommends the fastest BAS cleanup path.
</div>

<div class="panel">
<h2>Recommendation</h2>
<div class="review">{{ review_text }}</div>
</div>

<div class="foot">Made by Paolo V. R.</div>
</div>
</body></html>'''


FLOORPLAN_BASE_PAGE = '''<!DOCTYPE html>
<html><head><title>Floorplan Base v31</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{background:#f5f6f8;color:#111827;font-family:'Segoe UI',Arial,sans-serif;min-height:100vh;padding:18px;}
.wrap{max-width:1700px;margin:0 auto;}
.head{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;padding:14px 22px;background:white;border-radius:12px;border:1px solid #d8dee8;box-shadow:0 8px 24px rgba(15,23,42,0.08);}
.head h1{color:#1f4f82;font-size:22px;}
.head .sub{color:#64748b;font-size:13px;margin-top:4px;}
.head .badge{background:#1f4f82;color:white;padding:3px 9px;border-radius:5px;font-size:11px;font-weight:700;margin-left:8px;}
.actions{display:flex;gap:8px;flex-wrap:wrap;}
.btn{padding:10px 18px;border-radius:8px;border:none;font-size:14px;font-weight:700;cursor:pointer;text-decoration:none;display:inline-block;}
.btn-blue{background:#2563eb;color:white;}
.btn-green{background:#16a34a;color:white;}
.btn-gray{background:#e5e7eb;color:#111827;}
.btn-purple{background:#7c3aed;color:white;}
.panel{background:white;border:1px solid #d8dee8;border-radius:12px;padding:14px;box-shadow:0 8px 24px rgba(15,23,42,0.08);}
.panel h2{font-size:13px;letter-spacing:1px;text-transform:uppercase;color:#475569;margin-bottom:10px;}
.panel img{width:100%;display:block;border-radius:8px;border:1px solid #cbd5e1;background:white;}
.info{background:#fff7ed;border:1px solid #fed7aa;border-radius:12px;padding:12px 18px;color:#7c2d12;font-size:14px;margin-bottom:14px;}
.foot{text-align:center;color:#94a3b8;margin-top:18px;font-size:12px;}
</style></head><body>
<div class="wrap">
<div class="head">
<div>
<h1>Floorplan Shape Base <span class="badge">v33 preview</span></h1>
<div class="sub">First visual pass: same floorplan shape, cleaner BAS-style base</div>
</div>
<div class="actions">
<form method="GET" action="/" style="display:inline;"><button type="submit" class="btn btn-gray">&larr; New Upload</button></form>
<form method="POST" action="/floorplan-base" style="display:inline;"><input type="hidden" name="base_source" value="hybrid"><button type="submit" class="btn btn-gray">Hybrid</button></form>
<form method="POST" action="/floorplan-base" style="display:inline;"><input type="hidden" name="base_source" value="mask"><button type="submit" class="btn btn-gray">Mask Source</button></form>
<form method="POST" action="/floorplan-base" style="display:inline;"><input type="hidden" name="base_source" value="original"><button type="submit" class="btn btn-gray">Original Source</button></form>
<form method="POST" action="/mask-approve" style="display:inline;"><button type="submit" class="btn btn-blue">Back to Mask Editor Flow</button></form>
<form method="POST" action="/trace-editor" style="display:inline;"><input type="hidden" name="trace_mode" value="light"><button type="submit" class="btn btn-blue">Trace Light</button></form>
<form method="POST" action="/trace-editor" style="display:inline;"><input type="hidden" name="trace_mode" value="medium"><button type="submit" class="btn btn-blue">Trace Medium</button></form>
<form method="POST" action="/claude-review" style="display:inline;"><button type="submit" class="btn btn-purple">Claude Review</button></form>
<form method="POST" action="/floorplan-base-approve" style="display:inline;"><button type="submit" class="btn btn-green">Use This Base in Editor &rarr;</button></form>
</div>
</div>

<div class="info">
This preview is not trying to understand rooms or HVAC yet. It preserves the floorplan shape first. Source: <b>{{ source_mode }}</b>.
</div>

<div class="panel">
<h2>Generated BAS-Style Floorplan Base</h2>
<img src="data:image/png;base64,{{ base_b64 }}" alt="Floorplan Shape Base">
</div>

<div class="foot">Made by Paolo V. R.</div>
</div>
</body></html>'''


EDITOR_PAGE = '''<!DOCTYPE html>
<html><head><title>CAD Editor v29</title>
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
.btn-orange{background:#ea580c;color:white;}
.btn-blue{background:#2563eb;color:white;}
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
<h1>CAD Editor v29</h1>
<div style="display:flex;gap:6px;flex-wrap:wrap;">
<button onclick="undo()" class="action-btn btn-gray">&#8617; Undo</button>
<button onclick="alignSelected('h')" class="action-btn btn-blue" title="Align selected walls horizontal">&#8596; Align H</button>
<button onclick="alignSelected('v')" class="action-btn btn-blue" title="Align selected walls vertical">&#8597; Align V</button>
<button onclick="duplicateSelected()" class="action-btn btn-gray" title="Duplicate selected (Ctrl+D)">&#x2398; Duplicate</button>
<button onclick="clearDetectedOnly()" class="action-btn btn-orange">Clear Detected</button>
<button onclick="keepExteriorOnly()" class="action-btn btn-orange">Keep Exterior Only</button>
<button onclick="clearInteriorTrace()" class="action-btn btn-orange">Clear Interior Trace</button>
<button onclick="clearShortLines()" class="action-btn btn-orange">Clear Short Lines</button>
<button onclick="snapWalls()" class="action-btn btn-blue">Snap Walls</button>
<button onclick="clearAll()" class="action-btn btn-red">Clear All</button>
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
<button class="tool-btn" data-tool="room_rect" onclick="selectTool(this)">
<div class="color-swatch" style="background:#ea580c"></div> Room Rect
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
<button class="tool-btn" data-tool="door" onclick="selectTool(this)">
<div class="color-swatch" style="background:#facc15"></div> Door
</button>
<button class="tool-btn" data-tool="erase_rect" onclick="selectTool(this)">&#9633; Box Erase</button>

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
// v29: Multi-select + endpoint drag
let selectedSet = new Set();  // indices of multi-selected elements
let draggedEndpoint = null;   // { elIdx, pointIdx, originalPos }
let clipboard = [];           // for duplicate
// v29: Box-select + auto-extend
let boxSelectStart = null;
let boxSelectCurrent = null;
let shiftHeld = false;
let snapTarget = null;        // { x, y, type: 'endpoint'|'wall' } - shown as preview
// v29: Rectangle eraser state
let eraseRectStart = null;
let eraseRectCurrent = null;
// v33: Room rectangle state
let roomRectStart = null;
let roomRectCurrent = null;
// v29: Door state (2-click line)
let doorFirstPoint = null;

const COLORS = {
    extwall:'#9333ea', intwall:'#ea580c', duct:'#dcdce0',
    vav:'#1e40af', ahu:'#16a34a', diffuser:'#ffffff',
    door:'#facc15'
};

const STATUS_TEXTS = {
    extwall:'Click corners of building PERIMETER. Double-click to close.',
    intwall:'Click corners of an INTERIOR WALL. Double-click to finish.',
    room_rect:'Drag a rectangle to create a room/office as one editable wall group.',
    duct:'Click TWO points for a straight duct line.',
    vav:'Click to place a VAV.',
    ahu:'Click to place the AHU.',
    diffuser:'Click to place a diffuser.',
    move:'Drag endpoints (SHIFT=lock H/V). Drag empty area=box-select. Snap preview shown.',
    delete:'Click any element to delete it.',
    door:'Click TWO points for a door opening on a wall.',
    erase_rect:'Click & drag to draw a rectangle - everything inside gets deleted.'
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
    // v29: Reset new tool states
    eraseRectStart = null;
    eraseRectCurrent = null;
    roomRectStart = null;
    roomRectCurrent = null;
    doorFirstPoint = null;
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
    // v29: Door tool (2 clicks)
    if(currentTool === 'door'){
        if(!doorFirstPoint){
            doorFirstPoint = { x: pos.x, y: pos.y };
        } else {
            elements.push({
                type: 'door',
                points: [doorFirstPoint, { x: pos.x, y: pos.y }]
            });
            doorFirstPoint = null;
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

    // v29: Box-select drag
    if(boxSelectStart){
        boxSelectCurrent = pos;
        redraw();
        return;
    }

    // v29+v29: Drag endpoint (priority) with shift-lock and snap preview
    if(currentTool === 'move' && draggedEndpoint){
        const el = elements[draggedEndpoint.elIdx];
        if(!el || !el.points) return;
        const pt = el.points[draggedEndpoint.pointIdx];
        let nx = pos.x, ny = pos.y;

        // v29: SHIFT held = lock to perfect horizontal/vertical from "other" endpoint
        if(el.points.length >= 2){
            const otherIdx = draggedEndpoint.pointIdx === 0 ? 1 : draggedEndpoint.pointIdx - 1;
            const other = el.points[otherIdx];
            if(shiftHeld){
                // Hard lock: snap to H or V based on which axis is dominant
                const dx = Math.abs(nx - other.x);
                const dy = Math.abs(ny - other.y);
                if(dx > dy) ny = other.y; else nx = other.x;
            } else {
                // Soft auto-snap (v29 behavior)
                const dx = Math.abs(nx - other.x);
                const dy = Math.abs(ny - other.y);
                if(dx > dy * 2.5) ny = other.y;
                else if(dy > dx * 2.5) nx = other.x;
            }
        }
        pt.x = nx;
        pt.y = ny;

        // v29: Find snap preview target (don't apply yet, just show)
        snapTarget = null;
        for(let ei = 0; ei < elements.length; ei++){
            if(ei === draggedEndpoint.elIdx) continue;
            const other = elements[ei];
            if(!other.points) continue;
            // Snap to endpoints first
            for(const op of other.points){
                if(Math.hypot(pt.x - op.x, pt.y - op.y) < 18){
                    snapTarget = { x: op.x, y: op.y, type: 'endpoint' };
                    break;
                }
            }
            if(snapTarget) break;
            // Then snap to walls (auto-extend)
            for(let si = 0; si < other.points.length - 1; si++){
                const a = other.points[si];
                const b = other.points[si+1];
                const proj = nearestPointOnSegment(pt, a, b);
                if(Math.hypot(pt.x - proj.x, pt.y - proj.y) < 18){
                    snapTarget = { x: proj.x, y: proj.y, type: 'wall' };
                    break;
                }
            }
            if(snapTarget) break;
        }

        redraw();
        return;
    }

    if(currentTool === 'move' && selectedElement && dragOffset){
        moveElement(selectedElement, pos.x - dragOffset.x, pos.y - dragOffset.y);
        const c = getElementCenter(selectedElement);
        dragOffset = { x: pos.x - c.x, y: pos.y - c.y };
        redraw();
        return;
    }
    if(currentTool === 'erase_rect' && eraseRectStart){
        eraseRectCurrent = pos;
        redraw();
        return;
    }
    if(currentTool === 'room_rect' && roomRectStart){
        roomRectCurrent = pos;
        redraw();
        return;
    }
    if(currentPolyline || doorFirstPoint) redraw();
});

drawCanvas.addEventListener('mousedown', function(e){
    const pos = getMousePos(e);
    // v29: Rectangle eraser start
    if(currentTool === 'erase_rect'){
        eraseRectStart = pos;
        eraseRectCurrent = pos;
        return;
    }
    // v33: Room rectangle start
    if(currentTool === 'room_rect'){
        roomRectStart = pos;
        roomRectCurrent = pos;
        return;
    }
    if(currentTool !== 'move') return;

    // v29: First check if user clicked on a wall ENDPOINT (priority over center)
    const ep = findEndpointAt(pos, 12);
    if(ep){
        draggedEndpoint = ep;
        return;
    }

    // Otherwise look for whole-element click
    const idx = findElementAt(pos);
    if(idx !== -1){
        // v29: Shift+click for multi-select
        if(e.shiftKey){
            if(selectedSet.has(idx)) selectedSet.delete(idx);
            else selectedSet.add(idx);
            redraw();
            return;
        }
        // Normal: clear multi-select, drag single
        if(!selectedSet.has(idx)) selectedSet.clear();
        selectedElement = elements[idx];
        const c = getElementCenter(selectedElement);
        dragOffset = { x: pos.x - c.x, y: pos.y - c.y };
    } else {
        // v29: Empty-area click starts BOX SELECT (drag rectangle)
        if(!e.shiftKey) selectedSet.clear();
        boxSelectStart = pos;
        boxSelectCurrent = pos;
        redraw();
    }
});

// v29: Find a wall endpoint near a position
function findEndpointAt(pos, threshold){
    for(let i = elements.length - 1; i >= 0; i--){
        const el = elements[i];
        if(!el.points || el.type === 'branch') continue;
        // Don't drag endpoints of closed extwall polygons (they're computed)
        for(let j = 0; j < el.points.length; j++){
            const p = el.points[j];
            if(Math.hypot(pos.x - p.x, pos.y - p.y) < threshold){
                return { elIdx: i, pointIdx: j };
            }
        }
    }
    return null;
}

drawCanvas.addEventListener('mouseup', function(e){
    // v33: Room rectangle apply
    if(currentTool === 'room_rect' && roomRectStart && roomRectCurrent){
        const rx1 = Math.min(roomRectStart.x, roomRectCurrent.x);
        const ry1 = Math.min(roomRectStart.y, roomRectCurrent.y);
        const rx2 = Math.max(roomRectStart.x, roomRectCurrent.x);
        const ry2 = Math.max(roomRectStart.y, roomRectCurrent.y);
        const rw = rx2 - rx1;
        const rh = ry2 - ry1;
        roomRectStart = null;
        roomRectCurrent = null;
        if(rw > 12 && rh > 12){
            elements.push({
                type: 'intwall',
                points: [
                    { x: rx1, y: ry1 },
                    { x: rx2, y: ry1 },
                    { x: rx2, y: ry2 },
                    { x: rx1, y: ry2 },
                    { x: rx1, y: ry1 }
                ],
                detected: false,
                source: 'room_rect'
            });
            saveState();
        }
        redraw();
        return;
    }
    // v29: Rectangle eraser apply
    if(currentTool === 'erase_rect' && eraseRectStart && eraseRectCurrent){
        const rx1 = Math.min(eraseRectStart.x, eraseRectCurrent.x);
        const ry1 = Math.min(eraseRectStart.y, eraseRectCurrent.y);
        const rx2 = Math.max(eraseRectStart.x, eraseRectCurrent.x);
        const ry2 = Math.max(eraseRectStart.y, eraseRectCurrent.y);
        const before = elements.length;
        elements = elements.filter(el => !elementIntersectsRect(el, rx1, ry1, rx2, ry2));
        eraseRectStart = null;
        eraseRectCurrent = null;
        if(elements.length !== before) saveState();
        redraw();
        return;
    }
    // v29: Box-select commit
    if(boxSelectStart && boxSelectCurrent){
        const dx = Math.abs(boxSelectCurrent.x - boxSelectStart.x);
        const dy = Math.abs(boxSelectCurrent.y - boxSelectStart.y);
        if(dx > 5 || dy > 5){
            const rx1 = Math.min(boxSelectStart.x, boxSelectCurrent.x);
            const ry1 = Math.min(boxSelectStart.y, boxSelectCurrent.y);
            const rx2 = Math.max(boxSelectStart.x, boxSelectCurrent.x);
            const ry2 = Math.max(boxSelectStart.y, boxSelectCurrent.y);
            for(let i = 0; i < elements.length; i++){
                if(elementIntersectsRect(elements[i], rx1, ry1, rx2, ry2)){
                    selectedSet.add(i);
                }
            }
            document.getElementById('statusBar').textContent =
                `Selected ${selectedSet.size} elements. Align/Duplicate/Delete or Shift+drag for more.`;
        }
        boxSelectStart = null;
        boxSelectCurrent = null;
        redraw();
        return;
    }
    // v29+v29: Commit endpoint drag using snapTarget (auto-extend)
    if(draggedEndpoint){
        const el = elements[draggedEndpoint.elIdx];
        if(el && el.points){
            const pt = el.points[draggedEndpoint.pointIdx];
            // v29: Use snapTarget (endpoint OR wall intersection) for auto-extend
            if(snapTarget){
                pt.x = snapTarget.x;
                pt.y = snapTarget.y;
            } else {
                // Fallback: endpoint-only snap
                for(let ei = 0; ei < elements.length; ei++){
                    if(ei === draggedEndpoint.elIdx) continue;
                    const other = elements[ei];
                    if(!other.points) continue;
                    let snapped = false;
                    for(const op of other.points){
                        if(Math.hypot(pt.x - op.x, pt.y - op.y) < 15){
                            pt.x = op.x;
                            pt.y = op.y;
                            snapped = true;
                            break;
                        }
                    }
                    if(snapped) break;
                }
            }
        }
        draggedEndpoint = null;
        snapTarget = null;
        saveState();
        redraw();
        return;
    }
    if(selectedElement){ saveState(); selectedElement = null; dragOffset = null; }
});

function elementIntersectsRect(el, x1, y1, x2, y2){
    function pointIn(p){ return p.x >= x1 && p.x <= x2 && p.y >= y1 && p.y <= y2; }
    if(el.type === 'vav' || el.type === 'ahu' || el.type === 'diffuser'){
        return pointIn({ x: el.x, y: el.y });
    }
    if(el.points){
        for(const p of el.points){
            if(pointIn(p)) return true;
        }
    }
    return false;
}

document.addEventListener('keydown', function(e){
    // v29: Track Shift for axis lock during drag
    if(e.key === 'Shift') shiftHeld = true;
    if(e.key === 'Escape'){
        if(currentPolyline){ currentPolyline = null; redraw(); }
        if(doorFirstPoint){ doorFirstPoint = null; redraw(); }
        if(eraseRectStart){ eraseRectStart = null; eraseRectCurrent = null; redraw(); }
        if(roomRectStart){ roomRectStart = null; roomRectCurrent = null; redraw(); }
        // v29: Escape clears multi-selection
        if(selectedSet.size > 0){ selectedSet.clear(); redraw(); }
    }
    // v29: Ctrl+D duplicates selection
    if((e.ctrlKey || e.metaKey) && e.key === 'd'){
        e.preventDefault();
        duplicateSelected();
    }
    // v29: Delete key removes multi-selection
    if((e.key === 'Delete' || e.key === 'Backspace') && selectedSet.size > 0){
        e.preventDefault();
        const indices = Array.from(selectedSet).sort((a,b) => b - a);
        for(const idx of indices) elements.splice(idx, 1);
        selectedSet.clear();
        saveState();
        redraw();
    }
});

// v29: Release Shift state
document.addEventListener('keyup', function(e){
    if(e.key === 'Shift') shiftHeld = false;
});

// v29: Align selected walls to horizontal or vertical
function alignSelected(orientation){
    if(selectedSet.size === 0){
        alert('Shift+click walls to select them first, then click Align.');
        return;
    }
    let changes = 0;
    for(const idx of selectedSet){
        const el = elements[idx];
        if(!el || !el.points || el.points.length < 2) continue;
        if(el.type !== 'extwall' && el.type !== 'intwall' && el.type !== 'duct') continue;
        for(let i = 0; i < el.points.length - 1; i++){
            const p1 = el.points[i];
            const p2 = el.points[i+1];
            if(orientation === 'h'){
                const yAvg = (p1.y + p2.y) / 2;
                p1.y = yAvg; p2.y = yAvg;
            } else {
                const xAvg = (p1.x + p2.x) / 2;
                p1.x = xAvg; p2.x = xAvg;
            }
            changes++;
        }
    }
    if(changes > 0){
        saveState();
        redraw();
        document.getElementById('statusBar').textContent = `Aligned ${changes} segments to ${orientation.toUpperCase()}.`;
    }
}

// v29: Duplicate selected elements (offset 30px right+down)
function duplicateSelected(){
    if(selectedSet.size === 0){
        alert('Shift+click elements to select them first.');
        return;
    }
    const offsetX = 30, offsetY = 30;
    const newIndices = new Set();
    for(const idx of selectedSet){
        const el = elements[idx];
        if(!el) continue;
        const copy = JSON.parse(JSON.stringify(el));
        delete copy.detected;  // Cloned elements are user-made
        if(copy.type === 'vav' || copy.type === 'ahu' || copy.type === 'diffuser'){
            copy.x += offsetX;
            copy.y += offsetY;
        } else if(copy.points){
            for(const p of copy.points){
                p.x += offsetX;
                p.y += offsetY;
            }
        }
        elements.push(copy);
        newIndices.add(elements.length - 1);
    }
    selectedSet = newIndices;  // Select the new copies
    saveState();
    redraw();
    document.getElementById('statusBar').textContent = `Duplicated ${newIndices.size} elements.`;
}

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
    for(let i = 0; i < elements.length; i++){
        drawElement(elements[i], false, i);
    }
    // v29: Draw endpoint handles when in Move mode
    if(currentTool === 'move'){
        drawCtx.fillStyle = '#00ffaa';
        drawCtx.strokeStyle = '#fff';
        drawCtx.lineWidth = 1.5;
        for(const el of elements){
            if(!el.points || el.type === 'branch') continue;
            for(const p of el.points){
                drawCtx.beginPath();
                drawCtx.arc(p.x, p.y, 5, 0, Math.PI * 2);
                drawCtx.fill();
                drawCtx.stroke();
            }
        }
    }
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
    // v29: Door preview after first click
    if(doorFirstPoint && hoverPoint){
        drawCtx.fillStyle = '#facc15';
        drawCtx.beginPath();
        drawCtx.arc(doorFirstPoint.x, doorFirstPoint.y, 5, 0, Math.PI * 2);
        drawCtx.fill();
        drawCtx.strokeStyle = '#facc15';
        drawCtx.lineWidth = 4;
        drawCtx.setLineDash([6, 4]);
        drawCtx.beginPath();
        drawCtx.moveTo(doorFirstPoint.x, doorFirstPoint.y);
        drawCtx.lineTo(hoverPoint.x, hoverPoint.y);
        drawCtx.stroke();
        drawCtx.setLineDash([]);
    }
    // v29: Rectangle eraser preview
    if(eraseRectStart && eraseRectCurrent){
        const x = Math.min(eraseRectStart.x, eraseRectCurrent.x);
        const y = Math.min(eraseRectStart.y, eraseRectCurrent.y);
        const w = Math.abs(eraseRectCurrent.x - eraseRectStart.x);
        const h = Math.abs(eraseRectCurrent.y - eraseRectStart.y);
        drawCtx.fillStyle = 'rgba(220, 38, 38, 0.2)';
        drawCtx.fillRect(x, y, w, h);
        drawCtx.strokeStyle = '#dc2626';
        drawCtx.lineWidth = 2;
        drawCtx.setLineDash([8, 4]);
        drawCtx.strokeRect(x, y, w, h);
        drawCtx.setLineDash([]);
    }
    // v33: Room rectangle preview
    if(roomRectStart && roomRectCurrent){
        const x = Math.min(roomRectStart.x, roomRectCurrent.x);
        const y = Math.min(roomRectStart.y, roomRectCurrent.y);
        const w = Math.abs(roomRectCurrent.x - roomRectStart.x);
        const h = Math.abs(roomRectCurrent.y - roomRectStart.y);
        drawCtx.fillStyle = 'rgba(234, 88, 12, 0.10)';
        drawCtx.fillRect(x, y, w, h);
        drawCtx.strokeStyle = '#ea580c';
        drawCtx.lineWidth = 4;
        drawCtx.setLineDash([10, 5]);
        drawCtx.strokeRect(x, y, w, h);
        drawCtx.setLineDash([]);
    }
    // v29: Box-select preview (cyan, distinct from red eraser)
    if(boxSelectStart && boxSelectCurrent){
        const x = Math.min(boxSelectStart.x, boxSelectCurrent.x);
        const y = Math.min(boxSelectStart.y, boxSelectCurrent.y);
        const w = Math.abs(boxSelectCurrent.x - boxSelectStart.x);
        const h = Math.abs(boxSelectCurrent.y - boxSelectStart.y);
        drawCtx.fillStyle = 'rgba(0, 212, 255, 0.15)';
        drawCtx.fillRect(x, y, w, h);
        drawCtx.strokeStyle = '#00d4ff';
        drawCtx.lineWidth = 1.5;
        drawCtx.setLineDash([6, 4]);
        drawCtx.strokeRect(x, y, w, h);
        drawCtx.setLineDash([]);
    }
    // v29: Snap target preview (cyan circle on snap point + dashed line from current pos)
    if(snapTarget && draggedEndpoint){
        const el = elements[draggedEndpoint.elIdx];
        if(el && el.points){
            const pt = el.points[draggedEndpoint.pointIdx];
            // Dashed line from current endpoint to snap target
            drawCtx.strokeStyle = '#00d4ff';
            drawCtx.lineWidth = 2;
            drawCtx.setLineDash([5, 4]);
            drawCtx.beginPath();
            drawCtx.moveTo(pt.x, pt.y);
            drawCtx.lineTo(snapTarget.x, snapTarget.y);
            drawCtx.stroke();
            drawCtx.setLineDash([]);
            // Big cyan ring on snap target
            drawCtx.strokeStyle = '#00d4ff';
            drawCtx.lineWidth = 2.5;
            drawCtx.beginPath();
            drawCtx.arc(snapTarget.x, snapTarget.y, 12, 0, Math.PI * 2);
            drawCtx.stroke();
            // Inner filled dot
            drawCtx.fillStyle = snapTarget.type === 'endpoint' ? '#00ffaa' : '#facc15';
            drawCtx.beginPath();
            drawCtx.arc(snapTarget.x, snapTarget.y, 4, 0, Math.PI * 2);
            drawCtx.fill();
        }
    }
}

function drawElement(el, inProgress = false, elIdx = -1){
    const color = COLORS[el.type] || '#fff';
    const detectedAlpha = el.detected ? 0.7 : 1.0;
    const isSelected = elIdx >= 0 && selectedSet.has(elIdx);

    // v29: Draw selection halo behind the element
    if(isSelected){
        drawCtx.save();
        drawCtx.shadowColor = '#00d4ff';
        drawCtx.shadowBlur = 12;
        drawCtx.strokeStyle = '#00d4ff';
        drawCtx.lineWidth = 8;
        drawCtx.globalAlpha = 0.5;
        if(el.type === 'vav' || el.type === 'ahu' || el.type === 'diffuser'){
            drawCtx.beginPath();
            drawCtx.arc(el.x, el.y, 18, 0, Math.PI * 2);
            drawCtx.stroke();
        } else if(el.points && el.points.length > 0){
            drawCtx.beginPath();
            drawCtx.moveTo(el.points[0].x, el.points[0].y);
            for(let i = 1; i < el.points.length; i++){
                drawCtx.lineTo(el.points[i].x, el.points[i].y);
            }
            drawCtx.stroke();
        }
        drawCtx.restore();
    }

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
    // v29: Door rendering
    if(el.type === 'door'){
        if(!el.points || el.points.length < 2) return;
        drawCtx.strokeStyle = '#facc15';
        drawCtx.lineWidth = 6;
        drawCtx.lineCap = 'round';
        drawCtx.globalAlpha = el.detected ? 0.8 : 1.0;
        drawCtx.beginPath();
        drawCtx.moveTo(el.points[0].x, el.points[0].y);
        drawCtx.lineTo(el.points[1].x, el.points[1].y);
        drawCtx.stroke();
        drawCtx.globalAlpha = 1.0;
        // Yellow endpoint dots
        drawCtx.fillStyle = '#facc15';
        for(const p of el.points){
            drawCtx.beginPath();
            drawCtx.arc(p.x, p.y, 4, 0, Math.PI * 2);
            drawCtx.fill();
        }
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
    selectedSet.clear();  // v29
    redraw();
}

function clearAll(){
    if(!confirm('Clear everything?')) return;
    elements = [];
    currentPolyline = null;
    selectedSet.clear();  // v29
    saveState();
    redraw();
}

// v29: Clear only auto-detected elements (keep what user drew manually)
function clearDetectedOnly(){
    const before = elements.length;
    elements = elements.filter(el => !el.detected);
    const removed = before - elements.length;
    if(removed === 0){
        alert('No auto-detected elements to clear.');
        return;
    }
    saveState();
    redraw();
}

// v32.2: Keep only exterior shape when interior trace is too noisy.
function keepExteriorOnly(){
    const before = elements.length;
    elements = elements.filter(el => el.type === 'extwall');
    selectedSet.clear();
    const removed = before - elements.length;
    if(removed === 0){
        alert('Only exterior walls were present.');
        return;
    }
    saveState();
    redraw();
    document.getElementById('statusBar').textContent = `Kept exterior shape and removed ${removed} other elements.`;
}

// v32.2: Remove auto-traced interiors/doors, keep exterior and manual HVAC.
function clearInteriorTrace(){
    const before = elements.length;
    elements = elements.filter(el => {
        if(!el.detected) return true;
        if(el.type === 'extwall') return true;
        return !(el.type === 'intwall' || el.type === 'door');
    });
    selectedSet.clear();
    const removed = before - elements.length;
    if(removed === 0){
        alert('No detected interior trace to clear.');
        return;
    }
    saveState();
    redraw();
    document.getElementById('statusBar').textContent = `Cleared ${removed} detected interior trace elements.`;
}

// v33: Remove short detected wall fragments while keeping exterior and room rectangles.
function clearShortLines(){
    const MIN_LEN = Math.max(50, Math.min(drawCanvas.width, drawCanvas.height) * 0.08);
    const before = elements.length;
    elements = elements.filter(el => {
        if(el.type !== 'intwall' || !el.points || el.points.length < 2) return true;
        if(!el.detected) return true;
        let total = 0;
        for(let i = 0; i < el.points.length - 1; i++){
            const a = el.points[i], b = el.points[i+1];
            total += Math.hypot(b.x - a.x, b.y - a.y);
        }
        return total >= MIN_LEN;
    });
    selectedSet.clear();
    const removed = before - elements.length;
    if(removed === 0){
        alert('No short detected lines were found.');
        return;
    }
    saveState();
    redraw();
    document.getElementById('statusBar').textContent = `Removed ${removed} short detected wall fragments.`;
}

// v29: Snap walls - align almost-straight walls to perfect H/V and merge close ones
function snapWalls(){
    const SNAP_ANGLE_DEG = 8;
    const MERGE_DIST = 12;
    let changes = 0;

    // Step 1: Snap each wall segment to H or V if close
    elements.forEach(el => {
        if((el.type === 'extwall' || el.type === 'intwall') && el.points && el.points.length >= 2){
            for(let i = 0; i < el.points.length - 1; i++){
                const p1 = el.points[i];
                const p2 = el.points[i+1];
                const dx = p2.x - p1.x;
                const dy = p2.y - p1.y;
                if(Math.abs(dx) < 1 && Math.abs(dy) < 1) continue;
                let angle = Math.atan2(dy, dx) * 180 / Math.PI;
                if(angle < 0) angle += 180;
                if(angle < SNAP_ANGLE_DEG || angle > 180 - SNAP_ANGLE_DEG){
                    // Snap to horizontal
                    const yAvg = (p1.y + p2.y) / 2;
                    p1.y = yAvg; p2.y = yAvg;
                    changes++;
                } else if(Math.abs(angle - 90) < SNAP_ANGLE_DEG){
                    // Snap to vertical
                    const xAvg = (p1.x + p2.x) / 2;
                    p1.x = xAvg; p2.x = xAvg;
                    changes++;
                }
            }
        }
    });

    // Step 2: Snap close endpoints together
    const allPoints = [];
    elements.forEach((el, ei) => {
        if(el.points){
            el.points.forEach((p, pi) => {
                allPoints.push({ p, ei, pi });
            });
        }
    });
    for(let i = 0; i < allPoints.length; i++){
        for(let j = i+1; j < allPoints.length; j++){
            const a = allPoints[i].p;
            const b = allPoints[j].p;
            const d = Math.hypot(a.x - b.x, a.y - b.y);
            if(d > 0 && d < MERGE_DIST){
                const mx = (a.x + b.x) / 2;
                const my = (a.y + b.y) / 2;
                a.x = mx; a.y = my;
                b.x = mx; b.y = my;
                changes++;
            }
        }
    }

    if(changes === 0){
        alert('No walls needed snapping.');
        return;
    }
    saveState();
    redraw();
    document.getElementById('statusBar').textContent = `Snapped ${changes} segments/endpoints.`;
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
<html><head><title>BAS Graphic v29</title>
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
<h1>Synchrony BAS Graphic v29</h1>
<p class="sub">Top-down aerial projection - Ready for Tracer Synchrony / Niagara</p>
<div class="stats">
<div class="stat">VAVs: <b>{{ n_vavs }}</b></div>
<div class="stat">AHUs: <b>{{ n_ahus }}</b></div>
<div class="stat">Ducts: <b>{{ n_ducts }}</b></div>
<div class="stat">Diffusers: <b>{{ n_diffs }}</b></div>
<div class="stat">Walls: <b>{{ n_walls }}</b></div>
<div class="stat">Doors: <b>{{ n_doors }}</b></div>
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

// === TOP-DOWN AERIAL CABINET PROJECTION (v29) ===
// True cabinet projection from above: floor compressed Y but kept large,
// walls extruded vertically with subtle perspective.
// Much more "looking down at the building" feel.
const TILT_ANGLE = Math.PI / 4;  // 45 degrees - more aerial
const COS_T = Math.cos(TILT_ANGLE);
const SIN_T = Math.sin(TILT_ANGLE);

// Aerial cabinet projection:
// - X stays fully horizontal
// - Y is compressed (multiplied by ~0.7) to give that top-down look
// - Z (wall height) appears as small vertical lift
function cabinetProject(x, y, z){
    // Floor compression for aerial view (0.7 = strong top-down)
    const Y_COMPRESSION = 0.72;
    // Wall height visual scale (kept moderate so walls don't dominate)
    const Z_SCALE = 0.65;
    const sx = x;
    const sy = y * Y_COMPRESSION - z * Z_SCALE;
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
    const WALL_HEIGHT = 55;
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

    // === FLOOR PATTERN v29 - subtle grid with soft lighting ===
    svg += `<pattern id="floorGrid" width="48" height="34" patternUnits="userSpaceOnUse">`;
    svg += `<rect width="48" height="34" fill="#e2e2e6"/>`;
    svg += `<path d="M 0 0 L 48 0 M 0 0 L 0 34" stroke="#c8c8cc" stroke-width="0.5" opacity="0.6"/>`;
    svg += `</pattern>`;

    // Soft floor lighting gradient (lighter center, darker edges = ambient depth)
    svg += `<radialGradient id="floorLight" cx="50%" cy="35%" r="65%">`;
    svg += `<stop offset="0%" stop-color="#ffffff" stop-opacity="0.25"/>`;
    svg += `<stop offset="100%" stop-color="#000000" stop-opacity="0.12"/>`;
    svg += `</radialGradient>`;

    // === WALL GRADIENTS v29 - darker outer, AO shading ===
    svg += `<linearGradient id="extSide" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#b8b8be"/><stop offset="100%" stop-color="#7a7a80"/></linearGradient>`;
    svg += `<linearGradient id="extTop" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#dcdce0"/><stop offset="100%" stop-color="#a8a8ac"/></linearGradient>`;
    svg += `<linearGradient id="intSide" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#c8c8cc"/><stop offset="100%" stop-color="#9a9aa0"/></linearGradient>`;
    svg += `<linearGradient id="intTop" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#e2e2e6"/><stop offset="100%" stop-color="#b8b8bc"/></linearGradient>`;
    svg += `<linearGradient id="wallEnd" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#a8a8ad"/><stop offset="100%" stop-color="#86868c"/></linearGradient>`;

    // === DUCT GRADIENTS v29 - cleaner white, more volumetric ===
    svg += `<linearGradient id="ductTop" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#ffffff"/><stop offset="60%" stop-color="#f4f4f7"/><stop offset="100%" stop-color="#e0e0e4"/></linearGradient>`;
    svg += `<linearGradient id="ductSide" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#d8d8dc"/><stop offset="100%" stop-color="#a4a4a8"/></linearGradient>`;

    // Equipment gradients (unchanged - they already look good)
    svg += `<linearGradient id="vavTop" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#3b6df0"/><stop offset="100%" stop-color="#1e40af"/></linearGradient>`;
    svg += `<linearGradient id="vavFront" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#1e3a8a"/><stop offset="100%" stop-color="#152a6e"/></linearGradient>`;
    svg += `<linearGradient id="vavRight" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#1e40af"/><stop offset="100%" stop-color="#0c1f5c"/></linearGradient>`;
    svg += `<linearGradient id="ahuTop" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#34d365"/><stop offset="100%" stop-color="#16a34a"/></linearGradient>`;
    svg += `<linearGradient id="ahuFront" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#15803d"/><stop offset="100%" stop-color="#0a5828"/></linearGradient>`;
    svg += `<linearGradient id="ahuRight" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#16a34a"/><stop offset="100%" stop-color="#0c5a26"/></linearGradient>`;

    // Wall shadow filter for soft drop shadows on floor
    svg += `<filter id="wallShadow" x="-20%" y="-20%" width="140%" height="140%">`;
    svg += `<feGaussianBlur in="SourceAlpha" stdDeviation="3"/>`;
    svg += `<feOffset dx="2" dy="3"/>`;
    svg += `<feComponentTransfer><feFuncA type="linear" slope="0.35"/></feComponentTransfer>`;
    svg += `<feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge>`;
    svg += `</filter>`;
    svg += `</defs>`;

    if(extWall){
        const pts = extWall.points.map(p => toLocal(p));
        let path = '';
        for(let i = 0; i < pts.length; i++){
            const [sx, sy] = projSVG(pts[i].x, pts[i].y, 0);
            path += (i === 0 ? 'M' : 'L') + sx + ',' + sy + ' ';
        }
        path += 'Z';
        // Base floor grid
        svg += `<path d="${path}" fill="url(#floorGrid)" stroke="#888" stroke-width="0.5"/>`;
        // Soft lighting overlay
        svg += `<path d="${path}" fill="url(#floorLight)"/>`;
        // Ambient occlusion inner ring (subtle darker edge near walls)
        svg += `<path d="${path}" fill="none" stroke="#5a5a60" stroke-width="2.5" opacity="0.25"/>`;
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
        // v29 FIX: Walls start slightly BELOW floor (z = -2) to eliminate gap
        const BASE_Z = -2;
        const [b1ax, b1ay] = projSVG(p1a.x, p1a.y, BASE_Z);
        const [b1bx, b1by] = projSVG(p1b.x, p1b.y, BASE_Z);
        const [b2bx, b2by] = projSVG(p2b.x, p2b.y, BASE_Z);
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
            svg += drawThickWall(pts[i], pts[i+1], WALL_HEIGHT, 16, 'url(#extSide)', 'url(#extTop)', '#48484e');
        }
        if(pts.length >= 3){
            svg += drawThickWall(pts[pts.length-1], pts[0], WALL_HEIGHT, 16, 'url(#extSide)', 'url(#extTop)', '#48484e');
        }
    }

    elements.forEach(el => {
        if(el.type === 'intwall' && el.points && el.points.length >= 2){
            const pts = el.points.map(p => toLocal(p));
            for(let i = 0; i < pts.length - 1; i++){
                svg += drawThickWall(pts[i], pts[i+1], WALL_HEIGHT * 0.88, 9, 'url(#intSide)', 'url(#intTop)', '#58585e');
            }
        }
    });

    // v29: Render doors as short low walls (visible opening)
    elements.forEach(el => {
        if(el.type === 'door' && el.points && el.points.length === 2){
            const p1 = toLocal(el.points[0]);
            const p2 = toLocal(el.points[1]);
            // Draw a low door frame (half wall height = visible opening)
            const dx = p2.x - p1.x;
            const dy = p2.y - p1.y;
            const len = Math.sqrt(dx*dx + dy*dy);
            if(len < 1) return;
            const thickness = 9;
            const nx = -dy / len * thickness / 2;
            const ny = dx / len * thickness / 2;
            const p1a = { x: p1.x + nx, y: p1.y + ny };
            const p1b = { x: p1.x - nx, y: p1.y - ny };
            const p2b = { x: p2.x - nx, y: p2.y - ny };
            // Floor-level door threshold (visible as small rectangle on floor)
            const [b1a] = [projSVG(p1a.x, p1a.y, 0)];
            const [b1b] = [projSVG(p1b.x, p1b.y, 0)];
            const [b2a] = [projSVG(p2.x + nx, p2.y + ny, 0)];
            const [b2b] = [projSVG(p2b.x, p2b.y, 0)];
            // Door floor marker (slightly raised, distinct color)
            svg += `<path d="M ${b1a[0]},${b1a[1]} L ${b2a[0]},${b2a[1]} L ${b2b[0]},${b2b[1]} L ${b1b[0]},${b1b[1]} Z" fill="#e8e0c8" stroke="#a89860" stroke-width="0.6"/>`;
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
            // v29: more consistent thickness, slightly chunkier for visibility
            const ductW = 16, ductH = 11;
            const nx = -dy / len * ductW / 2;
            const ny = dx / len * ductW / 2;
            const p1a = { x: p1.x + nx, y: p1.y + ny };
            const p1b = { x: p1.x - nx, y: p1.y - ny };
            const p2a = { x: p2.x + nx, y: p2.y + ny };
            const p2b = { x: p2.x - nx, y: p2.y - ny };
            const zLevel = WALL_HEIGHT - 10;
            const [t1ax, t1ay] = projSVG(p1a.x, p1a.y, zLevel + ductH);
            const [t2ax, t2ay] = projSVG(p2a.x, p2a.y, zLevel + ductH);
            const [t1bx, t1by] = projSVG(p1b.x, p1b.y, zLevel + ductH);
            const [t2bx, t2by] = projSVG(p2b.x, p2b.y, zLevel + ductH);
            const [b1bx, b1by] = projSVG(p1b.x, p1b.y, zLevel);
            const [b2bx, b2by] = projSVG(p2b.x, p2b.y, zLevel);
            const [b1ax, b1ay] = projSVG(p1a.x, p1a.y, zLevel);
            const [b2ax, b2ay] = projSVG(p2a.x, p2a.y, zLevel);
            // Soft shadow underneath
            const [sh1ax, sh1ay] = projSVG(p1a.x + 2, p1a.y + 2, 0.5);
            const [sh2ax, sh2ay] = projSVG(p2a.x + 2, p2a.y + 2, 0.5);
            const [sh1bx, sh1by] = projSVG(p1b.x + 2, p1b.y + 2, 0.5);
            const [sh2bx, sh2by] = projSVG(p2b.x + 2, p2b.y + 2, 0.5);
            svg += `<path d="M ${sh1ax},${sh1ay} L ${sh2ax},${sh2ay} L ${sh2bx},${sh2by} L ${sh1bx},${sh1by} Z" fill="#000" opacity="0.18"/>`;
            // Top face - cleaner white with subtle gradient
            svg += `<path d="M ${t1ax},${t1ay} L ${t2ax},${t2ay} L ${t2bx},${t2by} L ${t1bx},${t1by} Z" fill="url(#ductTop)" stroke="#7a7a80" stroke-width="0.5"/>`;
            // Front face
            svg += `<path d="M ${b1bx},${b1by} L ${b2bx},${b2by} L ${t2bx},${t2by} L ${t1bx},${t1by} Z" fill="url(#ductSide)" stroke="#7a7a80" stroke-width="0.5"/>`;
            // End caps
            svg += `<path d="M ${b1ax},${b1ay} L ${b1bx},${b1by} L ${t1bx},${t1by} L ${t1ax},${t1ay} Z" fill="#a8a8ac" stroke="#7a7a80" stroke-width="0.5"/>`;
            svg += `<path d="M ${b2ax},${b2ay} L ${b2bx},${b2by} L ${t2bx},${t2by} L ${t2ax},${t2ay} Z" fill="#a8a8ac" stroke="#7a7a80" stroke-width="0.5"/>`;
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
            svg += drawCube(p.x, p.y, 9, 20, WALL_HEIGHT - 20, 'url(#vavTop)', 'url(#vavFront)', 'url(#vavRight)', '#0c1c5c');
        }
    });

    elements.forEach(el => {
        if(el.type === 'ahu'){
            const p = toLocal({ x: el.x, y: el.y });
            svg += drawCube(p.x, p.y, 22, 30, WALL_HEIGHT - 30, 'url(#ahuTop)', 'url(#ahuFront)', 'url(#ahuRight)', '#0a4220');
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
                detected_message = (
                    f"v29 Architectural Reconstruction: cropped {crop[2]}x{crop[3]}. "
                    f"{stats['lines_raw']} raw -> {stats.get('after_filter', 0)} filtered "
                    f"-> chained {stats.get('chained', 0)} | corridor: {stats.get('corridor', 'none')} "
                    f"| rooms: {stats.get('rooms', 0)} | walls: {stats.get('final_walls', n_walls)} "
                    f"+ {stats.get('doors', 0)} doors."
                )
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
    elif mode == "mask":
        # v30: Generate clean architectural mask and show preview
        try:
            # First apply smart isolation (reuse existing function)
            isolated, _crop = smart_isolate_floorplan(img)
            cv2.imwrite(UPLOAD_IMAGE_PATH, isolated)
            # Generate the clean architectural mask (v30.2: with debug stats)
            clean_mask, mask_stats = generate_clean_mask(isolated, debug=True, preset="balanced")
            preview = render_mask_preview(clean_mask, isolated)
            cv2.imwrite(MASK_BINARY_PATH, clean_mask)
            cv2.imwrite(MASK_PREVIEW_PATH, preview)
            # v30.2: Build stats text for display
            stats_text = format_mask_stats(mask_stats)
            return render_template_string(
                MASK_PREVIEW_PAGE,
                original_b64=image_to_base64(UPLOAD_IMAGE_PATH),
                preview_b64=image_to_base64(MASK_PREVIEW_PATH),
                stats_text=stats_text
            )
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            return f"<h2 style='color:white;background:#0d0f14;padding:30px;'>Mask generation error: {str(e)}<br><pre style='color:#aaa;font-size:11px;'>{tb}</pre><a href='/' style='color:#2d89ef'>Back</a></h2>"

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


@app.route("/claude-review", methods=["POST"])
def claude_review():
    """v34: Claude API advisor for choosing the next cleanup/trace step."""
    try:
        review_text = run_claude_plan_review()
        return render_template_string(
            CLAUDE_REVIEW_PAGE,
            review_text=review_text
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return f"<h2 style='color:white;background:#0d0f14;padding:30px;'>Claude Review error: {str(e)}<br><pre style='color:#aaa;font-size:11px;'>{tb}</pre><a href='/' style='color:#2d89ef'>Back</a></h2>"


@app.route("/floorplan-base", methods=["POST"])
def floorplan_base():
    """v31 preview: render a BAS-style base from the approved/current mask."""
    if not os.path.exists(MASK_BINARY_PATH):
        return "<h2 style='color:white;background:#0d0f14;padding:30px;'>No mask found. Run Mask Preview first. <a href='/' style='color:#2d89ef'>Back</a></h2>"
    try:
        source_mode = request.form.get("base_source", "hybrid")
        clean_mask = cv2.imread(MASK_BINARY_PATH, cv2.IMREAD_GRAYSCALE)
        if clean_mask is None:
            return "<h2 style='color:white;background:#0d0f14;padding:30px;'>Could not read clean mask. <a href='/' style='color:#2d89ef'>Back</a></h2>"
        original_img = cv2.imread(UPLOAD_IMAGE_PATH) if os.path.exists(UPLOAD_IMAGE_PATH) else None
        base = render_floorplan_shape_base(clean_mask, original_img=original_img, source_mode=source_mode)
        if base is None:
            return "<h2 style='color:white;background:#0d0f14;padding:30px;'>Could not render floorplan base. <a href='/' style='color:#2d89ef'>Back</a></h2>"
        cv2.imwrite(FLOORPLAN_BASE_PATH, base)
        return render_template_string(
            FLOORPLAN_BASE_PAGE,
            base_b64=image_to_base64(FLOORPLAN_BASE_PATH),
            source_mode=source_mode
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return f"<h2 style='color:white;background:#0d0f14;padding:30px;'>Floorplan base error: {str(e)}<br><pre style='color:#aaa;font-size:11px;'>{tb}</pre><a href='/' style='color:#2d89ef'>Back</a></h2>"


@app.route("/floorplan-base-approve", methods=["POST"])
def floorplan_base_approve():
    """Load the v31 floorplan base into the editor as the visual reference."""
    if not os.path.exists(FLOORPLAN_BASE_PATH):
        return "<h2 style='color:white;background:#0d0f14;padding:30px;'>No floorplan base generated. <a href='/' style='color:#2d89ef'>Back</a></h2>"
    return render_template_string(
        EDITOR_PAGE,
        image_b64=image_to_base64(FLOORPLAN_BASE_PATH),
        initial_elements=json.dumps([]),
        detected_message="v31 Floorplan Base: clean BAS-style shape loaded as reference. Next step is fast cleanup/editing tools."
    )


@app.route("/trace-editor", methods=["POST"])
def trace_editor():
    """v32: Open editor with an editable auto-traced wall layer."""
    if not os.path.exists(MASK_BINARY_PATH):
        return "<h2 style='color:white;background:#0d0f14;padding:30px;'>No mask found. Run Mask Preview first. <a href='/' style='color:#2d89ef'>Back</a></h2>"
    try:
        trace_mode = request.form.get("trace_mode", "light")
        clean_mask = cv2.imread(MASK_BINARY_PATH, cv2.IMREAD_GRAYSCALE)
        original_img = cv2.imread(UPLOAD_IMAGE_PATH) if os.path.exists(UPLOAD_IMAGE_PATH) else None
        if clean_mask is None:
            return "<h2 style='color:white;background:#0d0f14;padding:30px;'>Could not read clean mask. <a href='/' style='color:#2d89ef'>Back</a></h2>"
        elements = detect_editable_wall_trace(clean_mask, original_img=original_img, trace_mode=trace_mode)
        n_ext = sum(1 for e in elements if e.get("type") == "extwall")
        n_int = sum(1 for e in elements if e.get("type") == "intwall")
        n_doors = sum(1 for e in elements if e.get("type") == "door")
        bg_path = UPLOAD_IMAGE_PATH if os.path.exists(UPLOAD_IMAGE_PATH) else MASK_PREVIEW_PATH
        return render_template_string(
            EDITOR_PAGE,
            image_b64=image_to_base64(bg_path),
            initial_elements=json.dumps(elements),
            detected_message=(
                f"v32.2 Auto Trace {trace_mode}: {n_ext} exterior shape + {n_int} editable wall lines "
                f"+ {n_doors} door/opening hints. Use cleanup buttons when trace is noisy."
            )
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return f"<h2 style='color:white;background:#0d0f14;padding:30px;'>Auto trace error: {str(e)}<br><pre style='color:#aaa;font-size:11px;'>{tb}</pre><a href='/' style='color:#2d89ef'>Back</a></h2>"


@app.route("/mask-approve", methods=["POST"])
def mask_approve():
    """v30: User approved the mask preview. Enter editor with mask as visible reference."""
    if not os.path.exists(UPLOAD_IMAGE_PATH):
        return "<h2 style='color:white;background:#0d0f14;padding:30px;'>No plan uploaded. <a href='/' style='color:#2d89ef'>Back</a></h2>"
    # If mask preview exists, use it as the editor background (the clean mask)
    # Otherwise fallback to original isolated image
    bg_path = MASK_PREVIEW_PATH if os.path.exists(MASK_PREVIEW_PATH) else UPLOAD_IMAGE_PATH
    return render_template_string(
        EDITOR_PAGE,
        image_b64=image_to_base64(bg_path),
        initial_elements=json.dumps([]),
        detected_message="v30.3 Mask Mode: clean architectural mask loaded as reference. Start tracing walls on top!"
    )


@app.route("/mask-retry", methods=["POST"])
def mask_retry():
    """v30: Re-run the mask generation (could later expose tunable params)."""
    if not os.path.exists(UPLOAD_IMAGE_PATH):
        return "<h2 style='color:white;background:#0d0f14;padding:30px;'>No plan uploaded. <a href='/' style='color:#2d89ef'>Back</a></h2>"
    try:
        img = cv2.imread(UPLOAD_IMAGE_PATH)
        if img is None:
            return "<h2 style='color:white;background:#0d0f14;padding:30px;'>Could not re-read uploaded image. <a href='/' style='color:#2d89ef'>Back</a></h2>"
        preset = request.form.get("mask_preset", "balanced")
        clean_mask, mask_stats = generate_clean_mask(img, debug=True, preset=preset)
        preview = render_mask_preview(clean_mask, img)
        cv2.imwrite(MASK_BINARY_PATH, clean_mask)
        cv2.imwrite(MASK_PREVIEW_PATH, preview)
        stats_text = format_mask_stats(mask_stats)
        return render_template_string(
            MASK_PREVIEW_PAGE,
            original_b64=image_to_base64(UPLOAD_IMAGE_PATH),
            preview_b64=image_to_base64(MASK_PREVIEW_PATH),
            stats_text=stats_text
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return f"<h2 style='color:white;background:#0d0f14;padding:30px;'>Mask regeneration error: {str(e)}<br><pre style='color:#aaa;font-size:11px;'>{tb}</pre><a href='/' style='color:#2d89ef'>Back</a></h2>"


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
    n_doors = sum(1 for e in elements if e.get("type") == "door")

    return render_template_string(
        RESULT_PAGE,
        detection_json=json.dumps(detection),
        n_vavs=n_vavs, n_ahus=n_ahus, n_ducts=n_ducts,
        n_diffs=n_diffs, n_walls=n_walls, n_doors=n_doors
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

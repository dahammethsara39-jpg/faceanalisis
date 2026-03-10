"""
aura_engine.py
──────────────
Core scoring logic using MediaPipe + OpenCV only (no paid APIs).
Calculates 4 sub-scores and returns a final Aura score 0–100.

Sub-scores:
  1. Face Symmetry      (25%) – left/right landmark mirror distances
  2. Skin Glow          (25%) – brightness + smoothness in LAB color space
  3. Eye Intensity      (25%) – eye openness ratio + iris contrast
  4. Jawline Sharpness  (25%) – Canny edge strength along chin contour
"""

import cv2
import numpy as np
import mediapipe as mp
import math

# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh

# ── MediaPipe landmark indices (478-point model) ─────────────────────────────
# Left/Right symmetric pairs for symmetry scoring
SYMMETRY_PAIRS = [
    (234, 454),   # cheekbones
    (93,  323),   # lower cheek
    (132, 361),   # jaw sides
    (58,  288),   # mouth corners
    (33,  263),   # eye outer corners
    (133, 362),   # eye inner corners
    (70,  300),   # brow outer
    (107, 336),   # brow inner
    (116, 345),   # nose sides
]

# Eye landmarks
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Iris landmarks (MediaPipe face mesh with refine_landmarks=True)
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Jawline landmarks
JAW_LINE = [10, 338, 297, 332, 284, 251, 389, 356, 454,
            323, 361, 288, 397, 365, 379, 378, 400, 377,
            152, 148, 176, 149, 150, 136, 172, 58, 132,
            93, 234, 127, 162, 21, 54, 103, 67, 109, 10]


def _landmark_to_px(lm, w, h):
    """Convert normalized landmark to pixel coordinates."""
    return int(lm.x * w), int(lm.y * h)


def _euclidean(p1, p2):
    """Euclidean distance between two (x,y) points."""
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


# ── 1. FACE SYMMETRY ─────────────────────────────────────────────────────────
def score_symmetry(landmarks, w, h):
    """
    Compare distances from face midline to symmetric landmark pairs.
    Perfect symmetry = 100. Score drops as asymmetry increases.
    """
    # Estimate midline: midpoint between nose tip (1) and chin (152)
    nose  = _landmark_to_px(landmarks[1],   w, h)
    chin  = _landmark_to_px(landmarks[152], w, h)
    mid_x = (nose[0] + chin[0]) / 2

    diffs = []
    for li, ri in SYMMETRY_PAIRS:
        lp = _landmark_to_px(landmarks[li], w, h)
        rp = _landmark_to_px(landmarks[ri], w, h)
        left_dist  = abs(lp[0] - mid_x)
        right_dist = abs(rp[0] - mid_x)
        if left_dist + right_dist > 0:
            # Asymmetry ratio 0 (perfect) → 1 (total asymmetry)
            ratio = abs(left_dist - right_dist) / ((left_dist + right_dist) / 2)
            diffs.append(ratio)

    if not diffs:
        return 50.0

    avg_asymmetry = np.mean(diffs)          # 0 = perfect, higher = worse
    score = max(0, 100 - (avg_asymmetry * 250))  # scale to 0-100
    return round(float(np.clip(score, 20, 100)), 1)


# ── 2. SKIN GLOW ─────────────────────────────────────────────────────────────
def score_skin_glow(image, landmarks, w, h):
    """
    Analyse the skin region (forehead + cheeks) for:
      - Brightness (L channel in LAB)
      - Smoothness (inverse of Laplacian variance = texture)
    """
    # Build a face mask using convex hull of landmarks
    pts = np.array([[int(lm.x * w), int(lm.y * h)]
                    for lm in landmarks], dtype=np.int32)
    hull  = cv2.convexHull(pts)
    mask  = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 255)

    # Erode mask slightly to avoid edges/hair
    kernel = np.ones((15, 15), np.uint8)
    mask   = cv2.erode(mask, kernel, iterations=2)

    face_pixels = image[mask == 255]
    if len(face_pixels) < 100:
        return 50.0

    # Brightness via LAB L-channel
    lab      = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    brightness = float(np.mean(l_channel[mask == 255]))   # 0–255
    brightness_score = np.clip((brightness / 200) * 100, 0, 100)

    # Smoothness via Laplacian variance (lower variance = smoother skin)
    gray       = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap        = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var    = float(np.var(lap[mask == 255]))
    # Typical range: 0 (smooth) – 2000+ (very textured)
    smooth_score = np.clip(100 - (lap_var / 20), 0, 100)

    score = (brightness_score * 0.45) + (smooth_score * 0.55)
    return round(float(np.clip(score, 20, 100)), 1)


# ── 3. EYE INTENSITY ─────────────────────────────────────────────────────────
def _eye_aspect_ratio(landmarks, eye_indices, w, h):
    """
    Eye Aspect Ratio (EAR):
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    Higher EAR = more open / alert eyes.
    """
    pts = [_landmark_to_px(landmarks[i], w, h) for i in eye_indices]
    A = _euclidean(pts[1], pts[5])
    B = _euclidean(pts[2], pts[4])
    C = _euclidean(pts[0], pts[3])
    if C == 0:
        return 0
    return (A + B) / (2.0 * C)


def score_eye_intensity(image, landmarks, w, h):
    """
    Combines Eye Aspect Ratio (openness) + iris-to-sclera contrast.
    Large, open, contrasted eyes = high eye intensity.
    """
    left_ear  = _eye_aspect_ratio(landmarks, LEFT_EYE,  w, h)
    right_ear = _eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
    avg_ear   = (left_ear + right_ear) / 2.0

    # EAR typically 0.20 (squinting) – 0.40 (wide open)
    ear_score = np.clip((avg_ear - 0.15) / 0.25 * 100, 0, 100)

    # Iris contrast: compare darkness of iris region vs surrounding sclera
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    iris_scores = []

    for iris_indices, eye_indices in [(LEFT_IRIS, LEFT_EYE), (RIGHT_IRIS, RIGHT_EYE)]:
        try:
            # Iris centre + radius
            iris_pts = [_landmark_to_px(landmarks[i], w, h) for i in iris_indices]
            cx = int(np.mean([p[0] for p in iris_pts]))
            cy = int(np.mean([p[1] for p in iris_pts]))
            r  = max(int(_euclidean(iris_pts[0], iris_pts[2]) / 2), 2)

            iris_mask  = np.zeros_like(gray)
            cv2.circle(iris_mask, (cx, cy), r, 255, -1)

            # Sclera ring just outside iris
            sclera_mask = np.zeros_like(gray)
            cv2.circle(sclera_mask, (cx, cy), r + 8, 255, -1)
            sclera_mask = cv2.subtract(sclera_mask, iris_mask)

            iris_mean   = float(np.mean(gray[iris_mask > 0]))   if np.any(iris_mask > 0)   else 128
            sclera_mean = float(np.mean(gray[sclera_mask > 0])) if np.any(sclera_mask > 0) else 200
            contrast    = max(0, sclera_mean - iris_mean)        # higher = more defined eyes
            iris_scores.append(np.clip(contrast / 120 * 100, 0, 100))
        except Exception:
            iris_scores.append(50.0)

    contrast_score = float(np.mean(iris_scores)) if iris_scores else 50.0
    score = (ear_score * 0.5) + (contrast_score * 0.5)
    return round(float(np.clip(score, 20, 100)), 1)


# ── 4. JAWLINE SHARPNESS ─────────────────────────────────────────────────────
def score_jawline(image, landmarks, w, h):
    """
    Extracts the jawline contour from landmarks, then measures
    Canny edge strength along that region. Sharp, defined jaw = high score.
    """
    jaw_pts = np.array(
        [_landmark_to_px(landmarks[i], w, h) for i in JAW_LINE], dtype=np.int32
    )

    # Build narrow jaw mask (dilated polyline)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(mask, [jaw_pts], isClosed=False, color=255, thickness=20)

    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    jaw_edges = edges[mask == 255]
    if len(jaw_edges) == 0:
        return 50.0

    edge_density = float(np.mean(jaw_edges)) / 255.0  # 0–1
    score = np.clip(edge_density * 300, 0, 100)        # scale up
    return round(float(np.clip(score, 20, 100)), 1)


# ── TIER LABEL ───────────────────────────────────────────────────────────────
def get_tier(score):
    if score >= 90:
        return "Godly Aura",   "⚡"
    elif score >= 76:
        return "Elite Aura",   "👑"
    elif score >= 61:
        return "Rare Aura",    "💜"
    elif score >= 41:
        return "Mid Aura",     "✨"
    else:
        return "Developing",   "🌱"


# ── MAIN ENTRY POINT ─────────────────────────────────────────────────────────
def analyse(image_bytes: bytes) -> dict:
    """
    Main function called by Flask.
    Accepts raw image bytes, returns a dict with all scores.
    """
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Could not decode image. Please upload a JPG or PNG."}

    h, w = img.shape[:2]

    # Run MediaPipe Face Mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,          # enables iris landmarks 468-477
        min_detection_confidence=0.5
    ) as face_mesh:

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return {"error": "No face detected. Please upload a clear front-facing photo."}

        landmarks = results.multi_face_landmarks[0].landmark

    # ── Calculate each sub-score ────────────────────────────────────────────
    symmetry  = score_symmetry(landmarks, w, h)
    glow      = score_skin_glow(img, landmarks, w, h)
    eyes      = score_eye_intensity(img, landmarks, w, h)
    jawline   = score_jawline(img, landmarks, w, h)

    # ── Weighted final score ────────────────────────────────────────────────
    total = round(
        symmetry * 0.25 +
        glow     * 0.25 +
        eyes     * 0.25 +
        jawline  * 0.25,
        1
    )

    tier, icon = get_tier(total)

    return {
        "total":    total,
        "tier":     tier,
        "icon":     icon,
        "metrics": {
            "Face Symmetry":    symmetry,
            "Skin Glow":        glow,
            "Eye Intensity":    eyes,
            "Jawline":          jawline,
        }
    }

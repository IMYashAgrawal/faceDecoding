import cv2
import mediapipe as mp
import numpy as np
from insightface.app import FaceAnalysis

# =================== InsightFace Init ===================
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# =================== MediaPipe Init ===================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True)

# =================== Landmark Indices ===================
LANDMARKS = {
    "left_eye": 33,
    "right_eye": 263,
    "nose_tip": 1,
    "nose_left": 98,
    "nose_right": 327,
    "mouth_left": 61,
    "mouth_right": 291,
    "chin": 152,
    "forehead": 10,
    "jaw_left": 234,
    "jaw_right": 454,
}

# =================== Utility ===================
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# =================== Geometry Extraction ===================
def extract_geometry(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    face = result.multi_face_landmarks[0]
    p = {}

    for name, idx in LANDMARKS.items():
        lm = face.landmark[idx]
        p[name] = (lm.x * w, lm.y * h)

    face_width = distance(p["left_eye"], p["right_eye"]) * 2
    face_length = distance(p["forehead"], p["chin"])

    if face_width == 0 or face_length == 0:
        return None

    return np.array([
        face_length / face_width,
        distance(p["jaw_left"], p["jaw_right"]) / face_width,
        distance(p["nose_left"], p["nose_right"]) / face_width,
        distance(p["mouth_left"], p["mouth_right"]) / face_width,
    ])

# =================== Deep Embedding ===================
def extract_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    faces = app.get(img)
    if not faces:
        return None

    return faces[0].embedding

# =================== Similarity ===================
def cosine_distance(v1, v2):
    return 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# =================== Precompute ===================
def precompute_features(images):
    embeddings = {}
    geometries = {}

    for img in images:
        emb = extract_embedding(img)
        geo = extract_geometry(img)

        if emb is not None and geo is not None:
            embeddings[img] = emb
            geometries[img] = geo
        else:
            print(f"❌ Skipping {img} (face not detected)")

    return embeddings, geometries

# =================== Hybrid Score ===================
def hybrid_score(img1, img2, embeddings, geometries, w_deep=0.7, w_geo=0.3):
    deep = cosine_distance(embeddings[img1], embeddings[img2])
    geo = cosine_distance(geometries[img1], geometries[img2])
    return w_deep * deep + w_geo * geo

# =================== Emoji Mapping ===================
def score_to_emoji(score):
    if score < 0.30:
        return "🟢"   # Same person
    elif score < 0.60:
        return "🟡"   # Similar
    else:
        return "🔴"   # Different

# =================== RUN ===================
images = [
    "face1.png",
    "face2.jpg"
]

embeddings, geometries = precompute_features(images)

valid_images = list(embeddings.keys())
n = len(valid_images)

if n < 2:
    print("❌ Not enough valid faces to compare")
    exit()

print("\n😀 HYBRID FACE SIMILARITY MATRIX (EMOJI VIEW)\n")
print("Legend: 🟢 Same | 🟡 Similar | 🔴 Different | ⚪ Self\n")

# Header
header = f"{'':12}"
for img in valid_images:
    header += f"{img[:10]:>12}"
print(header)
print("-" * (12 + 12 * n))

# Matrix
for i in range(n):
    img1 = valid_images[i]
    row = f"{img1[:10]:12}"

    for j in range(n):
        img2 = valid_images[j]

        if i == j:
            row += f"{'⚪':>12}"
        else:
            score = hybrid_score(img1, img2, embeddings, geometries)
            row += f"{score_to_emoji(score):>12}"

    print(row)

print("\n🧠 Interpretation:")
print("🟢  < 0.30   → Same Person")
print("🟡  0.30–0.60 → Similar")
print("🔴  > 0.60   → Different Person")

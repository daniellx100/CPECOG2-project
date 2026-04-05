import os
import cv2
import numpy as np
import joblib
from tqdm import tqdm
from tensorflow.keras.models import load_model

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------
# CONFIG
# ----------------------------
DATASET_FOLDER = r"C:\Users\danie\Downloads\Project2\train"
IMG_SIZE = 48

# ----------------------------
# LOAD MODELS
# ----------------------------
cnn_model = load_model("cnn_clean.keras")
svm_bundle = joblib.load("fer2013_balanced_hog_lbp_svm.pkl")

svm_model = svm_bundle["pipeline"]
label_map = svm_bundle["label_map"]

# reverse label map
labels = {v:k for k,v in label_map.items()}

# ----------------------------
# FACE DETECTION
# ----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ----------------------------
# PREPROCESS
# ----------------------------
def preprocess_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        gray = gray[y:y+h, x:x+w]

    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    return gray

# ----------------------------
# LBP
# ----------------------------
def lbp(img):
    h, w = img.shape
    img_p = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REFLECT)

    center = img_p[1:h+1, 1:w+1]
    codes = np.zeros((h,w), dtype=np.uint8)

    offsets = [(-1,-1),(-1,0),(-1,1),
               (0,1),(1,1),(1,0),
               (1,-1),(0,-1)]

    for i,(dy,dx) in enumerate(offsets):
        neighbor = img_p[1+dy:h+1+dy,1+dx:w+1+dx]
        codes |= ((neighbor >= center) << i).astype(np.uint8)

    hist,_ = np.histogram(codes.ravel(), bins=256, range=(0,256), density=True)
    return hist.astype(np.float32)

# ----------------------------
# FEATURE EXTRACTION
# ----------------------------
def extract_features(img):
    face = preprocess_face(img)

    hog = cv2.HOGDescriptor((48,48),(16,16),(8,8),(8,8),9)
    hog_feat = hog.compute(face).reshape(-1)

    lbp_feat = lbp(face)

    hist = cv2.calcHist([face],[0],None,[32],[0,256]).ravel()
    hist = hist / (hist.sum() + 1e-8)

    return np.concatenate([hog_feat, lbp_feat, hist])

# ----------------------------
# PREDICTION
# ----------------------------
def predict(img):
    # CNN
    face = preprocess_face(img)
    cnn_input = face / 255.0
    cnn_input = cnn_input.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    cnn_probs = cnn_model.predict(cnn_input, verbose=0)[0]

    # SVM
    feats = extract_features(img).reshape(1, -1)
    decision = svm_model.decision_function(feats)

    exp = np.exp(decision)
    svm_probs = exp / np.sum(exp)

    # Ensemble
    final_probs = 0.7 * cnn_probs + 0.3 * svm_probs
    return np.argmax(final_probs)

# ----------------------------
# TEST LOOP
# ----------------------------
y_true = []
y_pred = []

print("\nRunning accuracy test...\n")

for label_idx, class_name in label_map.items():
    path = os.path.join(DATASET_FOLDER, class_name)
    files = os.listdir(path)

    for file in tqdm(files, desc=class_name):
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        pred = predict(img)

        y_true.append(label_idx)
        y_pred.append(pred)

# ----------------------------
# RESULTS
# ----------------------------
acc = accuracy_score(y_true, y_pred)

print("\n" + "="*50)
print("FINAL ENSEMBLE EVALUATION")
print("="*50)
print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=list(label_map.values())))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))
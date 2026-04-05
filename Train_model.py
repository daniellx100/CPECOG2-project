import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
import joblib
 
# ----------------------------
# CONFIGURATION
# ----------------------------
DATASET_FOLDER = r"C:\Users\danie\Downloads\Project2\train"   # FER2013 training folder
IMG_SIZE_SMALL = 48
IMG_SIZE_LARGE = 64
MAX_PER_CLASS = 4000       # For balancing
MODEL_FILENAME = "fer2013_balanced_hog_lbp_svm.pkl"
 
# ----------------------------
# HAAR CASCADES
# ----------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
 
# ----------------------------
# LBP FEATURE (enhanced)
# ----------------------------
def lbp_16u(img, radius=2, neighbors=16):
    h, w = img.shape
    img_p = cv2.copyMakeBorder(img, radius, radius, radius, radius, cv2.BORDER_REFLECT)
    center = img_p[radius:h+radius, radius:w+radius].astype(np.int32)
    codes = np.zeros((h, w), dtype=np.uint32)
    angles = np.linspace(0, 2*np.pi, neighbors, endpoint=False)
    for n, angle in enumerate(angles):
        dy = -int(round(radius * np.sin(angle)))
        dx = int(round(radius * np.cos(angle)))
        neighbor = img_p[radius+dy:h+radius+dy, radius+dx:w+radius+dx].astype(np.int32)
        codes |= ((neighbor >= center) << n).astype(np.uint32)
    hist, _ = np.histogram(codes.ravel(), bins=2**neighbors, range=(0, 2**neighbors), density=True)
    return hist.astype(np.float32)
 
# ----------------------------
# HOG FEATURE
# ----------------------------
def get_hog(win_size):
    block = (16, 16)
    stride = (8, 8)
    cell = (8, 8)
    nbins = 9
    return cv2.HOGDescriptor(win_size, block, stride, cell, nbins)
 
HOG_SMALL = get_hog((IMG_SIZE_SMALL, IMG_SIZE_SMALL))
HOG_LARGE = get_hog((IMG_SIZE_LARGE, IMG_SIZE_LARGE))
 
# ----------------------------
# FACE DETECTION + ALIGNMENT
# ----------------------------
def detect_face_align(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face_roi = gray[y:y+h, x:x+w]
 
    # Align eyes if detected
    eyes = eye_cascade.detectMultiScale(face_roi)
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[2], reverse=True)[:2]
        eye_centers = [(ex + ew//2, ey + eh//2) for (ex, ey, ew, eh) in eyes]
        eye_centers = sorted(eye_centers, key=lambda c: c[0])
        left_eye, right_eye = eye_centers
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        eye_center = ((left_eye[0]+right_eye[0])/2.0, (left_eye[1]+right_eye[1])/2.0)
        M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        aligned = cv2.warpAffine(face_roi, M, (face_roi.shape[1], face_roi.shape[0]))
    else:
        aligned = face_roi
 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    aligned = clahe.apply(aligned)
    return aligned
 
# ----------------------------
# AUGMENTATION
# ----------------------------
def augment_image(img):
    augmented = [img]
    # Horizontal flip
    augmented.append(cv2.flip(img, 1))
    # Small rotations
    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), angle, 1.0)
        augmented.append(cv2.warpAffine(img, M, (img.shape[1], img.shape[0])))
    # Brightness/contrast
    for alpha in [0.9, 1.1]:
        for beta in [-10, 10]:
            augmented.append(cv2.convertScaleAbs(img, alpha=alpha, beta=beta))
    # Small zooms
    for scale in [0.9, 1.1]:
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
        augmented.append(cv2.warpAffine(img, M, (w,h)))
    return augmented
 
# ----------------------------
# FEATURE EXTRACTION
# ----------------------------
def extract_features(img):
    img_s = cv2.resize(img, (IMG_SIZE_SMALL, IMG_SIZE_SMALL))
    img_l = cv2.resize(img, (IMG_SIZE_LARGE, IMG_SIZE_LARGE))
    hog_feat = HOG_SMALL.compute(img_s).reshape(-1)
    hog_feat_large = HOG_LARGE.compute(img_l).reshape(-1)
    lbp_feat = lbp_16u(img_s)
    hist = cv2.calcHist([img_s], [0], None, [32], [0,256]).ravel().astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return np.concatenate([hog_feat, hog_feat_large, lbp_feat, hist], axis=0)
 
# ----------------------------
# LOAD DATASET (BALANCED)
# ----------------------------
def load_dataset():
    X, y = [], []
    classes = sorted([c for c in os.listdir(DATASET_FOLDER) if os.path.isdir(os.path.join(DATASET_FOLDER, c))])
    label_map = {i:name for i,name in enumerate(classes)}
    print("\nLoading dataset with balancing and augmentation...\n")
    for label_idx, cls in label_map.items():
        class_path = os.path.join(DATASET_FOLDER, cls)
        files = os.listdir(class_path)
        np.random.shuffle(files)
        count = 0
        for img_name in tqdm(files, desc=cls, unit="img"):
            if count >= MAX_PER_CLASS:
                break
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            face = detect_face_align(img)
            if face is None:
                continue
            for aug in augment_image(face):
                feats = extract_features(aug)
                X.append(feats)
                y.append(label_idx)
                count += 1
                if count >= MAX_PER_CLASS:
                    break
    X, y = shuffle(np.array(X), np.array(y), random_state=42)
    return X, y, label_map
 
# ----------------------------
# MAIN TRAINING
# ----------------------------
if __name__ == "__main__":
    print("\n=====================================")
    print("  FER2013 Classical Ensemble Training")
    print("=====================================\n")
   
    X, y, label_map = load_dataset()
    print(f"\nDataset loaded: {len(X)} samples\n")
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
 
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.98, whiten=True)),
        ("svm", LinearSVC(C=5.0, class_weight="balanced", max_iter=10000))
    ])
 
    print("Training SVM...")
    pipeline.fit(X_train, y_train)
 
    print("\nEvaluating...")
    y_pred = pipeline.predict(X_test)
 
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=[label_map[i] for i in range(len(label_map))]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
 
    # Save model
    joblib.dump({
        "pipeline": pipeline,
        "label_map": label_map,
        "image_size_small": IMG_SIZE_SMALL,
        "image_size_large": IMG_SIZE_LARGE
    }, MODEL_FILENAME)
    print(f"\nModel saved as: {MODEL_FILENAME}")

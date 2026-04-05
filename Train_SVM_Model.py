import os
import cv2
import numpy as np
from tqdm import tqdm

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

import joblib

# ----------------------------
# CONFIG
# ----------------------------
DATASET_FOLDER = r"C:\Users\danie\Downloads\Project2\train"
IMG_SIZE = 48
MAX_PER_CLASS = 3000

MODEL_NAME = "fer2013_balanced_hog_lbp_svm.pkl"

# ----------------------------
# HOG
# ----------------------------
hog = cv2.HOGDescriptor(
    (48,48), (16,16), (8,8), (8,8), 9
)

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
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    hog_feat = hog.compute(img).reshape(-1)
    lbp_feat = lbp(img)

    hist = cv2.calcHist([img],[0],None,[32],[0,256]).ravel()
    hist = hist / (hist.sum() + 1e-8)

    return np.concatenate([hog_feat, lbp_feat, hist])

# ----------------------------
# LOAD DATASET
# ----------------------------
def load_dataset():
    X, y = [], []

    classes = sorted([
        c for c in os.listdir(DATASET_FOLDER)
        if os.path.isdir(os.path.join(DATASET_FOLDER, c))
    ])

    label_map = {i:name for i,name in enumerate(classes)}

    print("\nLoading dataset...\n")

    for label_idx, cls in label_map.items():
        path = os.path.join(DATASET_FOLDER, cls)
        files = os.listdir(path)

        np.random.shuffle(files)
        count = 0

        for file in tqdm(files, desc=cls):
            if count >= MAX_PER_CLASS:
                break

            img_path = os.path.join(path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            feats = extract_features(img)

            X.append(feats)
            y.append(label_idx)

            count += 1

    X, y = shuffle(np.array(X), np.array(y), random_state=42)

    return X, y, label_map

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    print("\n=====================================")
    print("   TRAINING SVM (HOG + LBP)")
    print("=====================================\n")

    X, y, label_map = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(C=1.0, max_iter=5000))
    ])

    print("\nTraining SVM...\n")
    pipeline.fit(X_train, y_train)

    print("\nEvaluating...\n")
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n🔥 Test Accuracy: {acc:.4f}")

    # ----------------------------
    # SAVE MODEL
    # ----------------------------
    joblib.dump({
        "pipeline": pipeline,
        "label_map": label_map
    }, MODEL_NAME)

    print(f"\n✅ Model saved as: {MODEL_NAME}")
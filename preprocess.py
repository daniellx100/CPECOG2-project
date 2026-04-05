import cv2
import os
import numpy as np
from tqdm import tqdm
 
# ---------------------------------------------
# Paths
# ---------------------------------------------
train_path = r"C:/Users/danie/Downloads/Project2/train"
test_path  = r"C:/Users/danie/Downloads/Project2/test"
 
IMG_SIZE = 48





# =============================================
# LBP FUNCTION
# =============================================
def lbp_8u(img):
    h, w = img.shape
    img_p = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REFLECT)
 
    center = img_p[1:h+1, 1:w+1].astype(np.int32)
    codes  = np.zeros((h, w), dtype=np.uint8)
 
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, 1),  ( 1, 1),  ( 1, 0),
        ( 1, -1), ( 0, -1)
    ]
 
    for bit, (dy, dx) in enumerate(offsets):
        neighbor = img_p[1+dy:h+1+dy, 1+dx:w+1+dx].astype(np.int32)
        codes |= ((neighbor >= center) << bit).astype(np.uint8)
 
    hist, _ = np.histogram(codes.ravel(), bins=256, range=(0, 256), density=True)
    return hist.astype(np.float32)





# =============================================
# HOG INITIALIZATION
# =============================================
def get_hog():
    win       = (IMG_SIZE, IMG_SIZE)
    block     = (16, 16)
    stride    = (8, 8)
    cell      = (8, 8)
    nbins     = 9
    return cv2.HOGDescriptor(win, block, stride, cell, nbins)
 
HOG = get_hog()





# =============================================
# FEATURE EXTRACTION
# =============================================
def extract_features(img_gray):
    img = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.equalizeHist(img)
 
    hog_feat = HOG.compute(img).reshape(-1)
    lbp_hist = lbp_8u(img)
 
    hist = cv2.calcHist([img], [0], None, [32], [0, 256]).ravel().astype(np.float32)
    hist = hist / (hist.sum() + 1e-8)
 
    return np.concatenate([hog_feat, lbp_hist, hist], axis=0).astype(np.float32)





# =============================================
# MAIN PREPROCESSING FUNCTION
# =============================================
def preprocess_images(folder_path):
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
 
    X, y = [], []
 
    classes = sorted([
        c for c in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, c))
    ])
 
    print(f"\nProcessing folder: {folder_path}")
 
    for label in classes:
        class_path = os.path.join(folder_path, label)
        files = os.listdir(class_path)
 
        print(f"\nClass: {label} ({len(files)} images)")
 
        for file in tqdm(files, desc=label, unit="img"):
            img_path = os.path.join(class_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
 
            feats = extract_features(img)
            X.append(feats)
            y.append(label)
 
    return np.array(X, dtype=np.float32), np.array(y)





# =============================================
# RUN DIRECTLY
# =============================================
if __name__ == "__main__":
    X_test, y_test = preprocess_images(test_path)
    print("\nFinished.")
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    
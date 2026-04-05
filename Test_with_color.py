import cv2
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from tkinter import Tk, filedialog

# ----------------------------
# CONFIG
# ----------------------------
IMG_SIZE = 48
USE_ENSEMBLE = False   

# ----------------------------
# LOAD MODEL
# ----------------------------
cnn_model = load_model("cnn_final_tuned.keras")  # 🔥 USE YOUR BEST MODEL
label_map = joblib.load("label_map.pkl")

# ----------------------------
# COLOR MAPPING
# ----------------------------
colors = {
    "angry": (0, 0, 255),
    "disgust": (0, 255, 0),
    "fear": (128, 0, 128),
    "happy": (0, 255, 255),
    "neutral": (200, 200, 200),
    "sad": (255, 0, 0),
    "surprise": (255, 255, 0)
}

# ----------------------------
# FACE PREPROCESS (MATCH TRAINING)
# ----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def preprocess_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        gray = gray[y:y+h, x:x+w]

    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    gray = gray / 255.0

    return gray

# ----------------------------
# PREDICT (CNN ONLY)
# ----------------------------
def predict(img):
    face = preprocess_face(img)
    face = face.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    probs = cnn_model.predict(face, verbose=0)[0]

    idx = np.argmax(probs)
    confidence = probs[idx] * 100

    return label_map[idx], confidence

# ----------------------------
# FILE PICKER
# ----------------------------
Tk().withdraw()
image_path = filedialog.askopenfilename()

img = cv2.imread(image_path)

# ----------------------------
# RUN PREDICTION
# ----------------------------
predicted_label, confidence = predict(img)
color = colors.get(predicted_label, (255,255,255))

print("\nPrediction:", predicted_label)
print("Confidence:", confidence)

cv2.putText(img, f"{predicted_label} ({confidence:.1f}%)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2)

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
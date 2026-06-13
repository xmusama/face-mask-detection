from pathlib import Path

import cv2
import numpy as np

from app_paths import IMAGE_SIZE


def read_uploaded_image(uploaded_file) -> np.ndarray:
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("File gambar tidak bisa dibaca.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def scenario_uses_enhancement(scenario_id: str) -> bool:
    return "enhanced" in scenario_id


def scenario_uses_crop(scenario_id: str) -> bool:
    return "crop" in scenario_id


def apply_clahe_rgb(image_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def apply_gaussian_denoise(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image_rgb, ksize=(3, 3), sigmaX=0.5)


def crop_face_if_detected(image_rgb: np.ndarray) -> tuple[np.ndarray, bool]:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(35, 35))
    if len(faces) == 0:
        return image_rgb, False

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    pad = int(0.18 * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image_rgb.shape[1], x + w + pad)
    y2 = min(image_rgb.shape[0], y + h + pad)
    return image_rgb[y1:y2, x1:x2], True


def detect_faces(image_rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(32, 32))
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


def crop_box_with_padding(image_rgb: np.ndarray, box: tuple[int, int, int, int], pad_ratio: float = 0.18) -> np.ndarray:
    x, y, w, h = box
    pad = int(pad_ratio * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image_rgb.shape[1], x + w + pad)
    y2 = min(image_rgb.shape[0], y + h + pad)
    return image_rgb[y1:y2, x1:x2]


def preprocess_image_for_scenario(image_rgb: np.ndarray, scenario_id: str) -> tuple[np.ndarray, bool]:
    processed = image_rgb.copy()
    face_detected = False

    if scenario_uses_crop(scenario_id):
        processed, face_detected = crop_face_if_detected(processed)
    if scenario_uses_enhancement(scenario_id):
        processed = apply_clahe_rgb(processed)
        processed = apply_gaussian_denoise(processed)

    processed = cv2.resize(processed, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return processed, face_detected


def extract_hog_features(images: np.ndarray) -> np.ndarray:
    hog = cv2.HOGDescriptor((128, 128), (16, 16), (8, 8), (8, 8), 9)
    features = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        features.append(hog.compute(gray).flatten())
    return np.array(features, dtype=np.float32)

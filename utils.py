import requests
import cv2
import numpy as np
import logging
from fastapi import HTTPException

logger = logging.getLogger(__name__)

def download_image_to_array(url: str, max_size=640) -> np.ndarray:
    """
    Download an image from a URL and resize it to a maximum size, returning it as a NumPy array.
    """
    try:
        # Disable SSL verification for local development (NOT SAFE FOR PRODUCTION)
        response = requests.get(url, timeout=10, verify=False)
        response.raise_for_status()
        image_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image.")
        h, w = img.shape[:2]
        if max(h, w) > max_size:
            if h > w:
                new_h = max_size
                new_w = int(w * (max_size / h))
            else:
                new_w = max_size
                new_h = int(h * (max_size / w))
            img = cv2.resize(img, (new_w, new_h))
        return img
    except Exception as e:
        logger.error(f"Image download failed from {url}: {e}")
        raise HTTPException(status_code=400, detail=f"Image download failed: {e}")

def detect_and_crop_face(img: np.ndarray) -> np.ndarray:
    """
    Detect and crop a face from an image using OpenCV's Haar cascade, returning the cropped NumPy array.
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            raise ValueError("No face detected.")
        x, y, w, h = faces[0]  # Use the first detected face
        cropped = img[y:y+h, x:x+w]
        return cropped
    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        raise HTTPException(status_code=422, detail=f"Face detection failed: {e}")
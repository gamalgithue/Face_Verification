import requests
import cv2
import tempfile
import logging
from deepface import DeepFace
from fastapi import HTTPException

logger = logging.getLogger(__name__)


def download_image_to_tempfile(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_file.write(response.content)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        logger.error(f"Image download failed from {url}: {e}")
        raise HTTPException(status_code=400, detail=f"Image download failed: {e}")


def detect_and_crop_face(image_path: str, backend="mtcnn") -> str:
    try:
        faces = DeepFace.extract_faces(img_path=image_path, detector_backend=backend, enforce_detection=False)
        if not faces:
            raise ValueError("No face detected.")

        face = faces[0]["facial_area"]
        img = cv2.imread(image_path)
        cropped = img[face["y"]:face["y"] + face["h"], face["x"]:face["x"] + face["w"]]

        temp_cropped = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(temp_cropped.name, cropped)
        return temp_cropped.name
    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        raise HTTPException(status_code=422, detail=f"Face detection failed: {e}")
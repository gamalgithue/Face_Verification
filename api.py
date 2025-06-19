from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import os
import logging
from utils import download_image_to_array, detect_and_crop_face

# --- ENVIRONMENT CONFIG ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU (not available on free plan)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logs

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI Init ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request Schema ---
class FaceVerificationRequest(BaseModel):
    id_url: str
    ref_url: str

# --- API Endpoint ---
@app.post("/verify")
async def verify_face(request: FaceVerificationRequest):
    """
    Verify if two face images match by downloading, cropping, and comparing them.
    """
    try:
        # Download and resize images
        id_img = download_image_to_array(request.id_url)
        ref_img = download_image_to_array(request.ref_url)

        # Detect and crop faces
        cropped_id_img = detect_and_crop_face(id_img)
        cropped_ref_img = detect_and_crop_face(ref_img)

        # Verify faces using DeepFace
        result = DeepFace.verify(
            img1_path=cropped_id_img,  # NumPy array of cropped face
            img2_path=cropped_ref_img,  # NumPy array of cropped face
            model_name="Facenet",
            detector_backend="opencv"  # Still applied, but fast on pre-cropped images
        )

        threshold = 0.6
        distance = result.get("distance", 1.0)
        is_match = distance < threshold

        return {"match": is_match}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Verification error: {e}")
        return {"error": str(e)}

# --- Health Check ---
@app.get("/")
def root():
    return {"message": "Face Verification API is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

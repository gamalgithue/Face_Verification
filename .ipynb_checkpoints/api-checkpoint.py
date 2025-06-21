from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import os
import logging
from utils import download_image_to_tempfile, detect_and_crop_face

# --- ENVIRONMENT CONFIG ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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
    temp_files = []
    try:
        id_img = download_image_to_tempfile(request.id_url)
        temp_files.append(id_img)
        ref_img = download_image_to_tempfile(request.ref_url)
        temp_files.append(ref_img)
        cropped_id_img = detect_and_crop_face(id_img)
        temp_files.append(cropped_id_img)
        cropped_ref_img = detect_and_crop_face(ref_img)
        temp_files.append(cropped_ref_img)

        result = DeepFace.verify(
            img1_path=cropped_id_img,
            img2_path=cropped_ref_img,
            model_name="Facenet",
            detector_backend="opencv"
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
    finally:
        for file in temp_files:
            if file and os.path.exists(file):
                os.remove(file)

# --- Health Check ---
@app.get("/")
def root():
    return {"message": "Face Verification API is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
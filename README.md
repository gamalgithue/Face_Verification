# Face Verification API for GP Aoun
A FastAPI-based face verification system for the GP Aoun transportation app, using DeepFace to verify driver identities by comparing ID and selfie images.

## Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the API: `python api.py`


## Testing
JSON request of image links to test
{
    "id_url": "https://drive.google.com/uc?export=download&id=1DAmGVPYAxA1w0iZEdj38YInNBBGOUN_Y",
    "ref_url": "https://drive.google.com/uc?export=download&id=120DDtafPe5lpUaf1mx96ZtL3H1p3X7xl"
}

##Respond
{
    "match": true
}

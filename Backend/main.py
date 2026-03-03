import sys
import os
import cv2
import numpy as np
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# --- DYNAMIC PATH INJECTION ---
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from inference.predictor import ClinicalPredictor

logger = logging.getLogger("DermaScan.API")

# Global variable to hold our inference engine
ai_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events.
    Loads the heavy PyTorch models into GPU memory ONCE here.
    """
    global ai_engine
    logger.info("Starting up FastAPI Server...")
    try:
        ai_engine = ClinicalPredictor()
        logger.info("DermaScanAI Engine loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load AI Engine: {e}")
        raise RuntimeError("Could not initialize ClinicalPredictor. Check weights and paths.")
    
    yield
    
    logger.info("Shutting down server, clearing memory...")


app = FastAPI(
    title="DermaScanAI API",
    description="Clinical-grade skin symptom diagnostic engine.",
    version="1.0.0",
    lifespan=lifespan
)

# --- CORS CONFIGURATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ENDPOINTS ---

@app.get("/")
async def root_health_check():
    """Simple health check to verify the server is breathing."""
    return {"status": "online", "message": "DermaScanAI API is running."}

@app.post("/api/analyze")
async def analyze_skin(file: UploadFile = File(...)):
    """
    The main clinical inference endpoint.
    Expects a multipart/form-data image upload.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # 1. Read the image file into a memory buffer
        contents = await file.read()
        
        # 2. Convert the memory buffer into a NumPy array
        nparr = np.frombuffer(contents, np.uint8)
        
        # 3. Decode the NumPy array into an OpenCV BGR image matrix
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Could not decode the image. The file might be corrupted.")

        # 4. Fire the AI Pipeline!
        logger.info(f"Processing incoming request: {file.filename}")
        report = ai_engine.predict(img_bgr)
        
        # If the predictor caught a low-quality image or missing face
        if report.get("status") in ["rejected", "error"]:
            raise HTTPException(status_code=422, detail=report)

        # 5. Return the successful diagnostic JSON
        return report

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"API Error during analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during clinical analysis.")
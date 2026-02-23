from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
import os
import tempfile
import shutil

# Import your custom AI Brain and NLP Vocal Cords
from Model.classifier.model import DermaScanInference
from Model.recommender.model import SkincareRecommender, ImprovementObserver

# Global variables to hold our AI in memory
vision_ai = None
nlp_doctor = None
observer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This runs exactly once when the server boots up. 
    It loads the heavy PyTorch weights into RAM so they are instantly ready.
    """
    global vision_ai, nlp_doctor, observer
    print("\n[System] Booting up AI Engines into RAM...")
    
    # Locate the exact path to your champion model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(base_dir, "Model", "classifier", "symptom_classifier_final1.pth")
    
    # 1. Initialize Vision Engine
    vision_ai = DermaScanInference(classifier_weights_path=weights_path)
    
    # 2. Initialize NLP Engines
    nlp_doctor = SkincareRecommender()
    observer = ImprovementObserver()
    
    print("[System] 🟢 All Engines ONLINE. Ready to receive patients.\n")
    yield
    print("\n[System] 🔴 Shutting down AI Engines...")
    # Clean up memory when the server stops

# Initialize FastAPI with the lifespan manager
app = FastAPI(title="DermaScan AI Backend", version="1.0", lifespan=lifespan)

@app.get("/")
async def health_check():
    """A simple ping to check if the server is breathing."""
    return {"status": "Active", "message": "DermaScan API is running."}

@app.post("/api/analyze")
async def analyze_skin(file: UploadFile = File(...)):
    """
    The main endpoint. Accepts an image, feeds it to the Vision AI, 
    routes the math to the NLP Doctor, and returns a JSON report.
    """
    # 1. Security Check: Reject anything that isn't an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    # 2. Save the uploaded image to a temporary file for the Vision AI to read
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 3. Trigger the Vision AI Math
        vision_report = vision_ai.analyze(temp_path)
        
        if "error" in vision_report:
            raise HTTPException(status_code=500, detail=vision_report["error"])
            
        # 4. Extract Top Symptoms for the NLP Engine
        # We grab the primary diagnosis and check if the second one is close enough to include
        sorted_probs = list(vision_report["all_probabilities"].items())
        primary_symptom, primary_conf = sorted_probs[0]
        
        diagnoses_to_pass = [(primary_symptom, primary_conf)]
        
        # If the second highest symptom is within 10% of the primary, include it for a multi-diagnosis
        if len(sorted_probs) > 1:
            secondary_symptom, secondary_conf = sorted_probs[1]
            if (primary_conf - secondary_conf) <= 10.0 and primary_symptom != "clear_face":
                diagnoses_to_pass.append((secondary_symptom, secondary_conf))
                
        # 5. Trigger the NLP Doctor
        prescription_text = nlp_doctor.generate_prescription(diagnoses_to_pass)
        
        # 6. Package and send the final response back to the frontend
        return {
            "status": "success",
            "vision_analysis": {
                "primary_diagnosis": primary_symptom,
                "confidence": primary_conf,
                "full_breakdown": vision_report["all_probabilities"]
            },
            "nlp_prescription": prescription_text
        }
        
    finally:
        # 7. Housekeeping: Delete the temporary image so your hard drive doesn't fill up
        if os.path.exists(temp_path):
            os.remove(temp_path)
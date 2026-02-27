import os
from pathlib import Path

class DirectoryLocator:
    """
    Centralized Configuration Manager for the DermaScanAI Pipeline.
    Dynamically resolves paths from the project root.
    """
    # 1. Project Root Resolution
    # Assuming config.py is located somewhere inside D:\Projects\DermaScanAI\Backend\
    # Adjust the .parent chain if config.py is deeper in the directory tree.
    ROOT_DIR = Path(__file__).resolve().parent.parent

    # 2. Pipeline Weights (The Three Brains)
    FACE_DETECTOR_WEIGHTS = os.path.join(ROOT_DIR, "Model", "detection", "face_detector_final.pth")
    IQA_GATEKEEPER_WEIGHTS = os.path.join(ROOT_DIR, "Model", "quality", "iqa_gatekeeper_b0.pth")
    SYMPTOM_CLASSIFIER_WEIGHTS = os.path.join(ROOT_DIR, "Model", "classifier", "symptom_classifier_final4.pth")

    @classmethod
    def verify_pipeline_readiness(cls):
        """
        Pre-flight diagnostic sweep. Call this once when the API boots.
        It ensures all three models exist before accepting user images.
        """
        print("--- INITIATING PIPELINE PRE-FLIGHT CHECK ---")
        
        models_to_check = {
            "Stage 1: Face Detector": cls.FACE_DETECTOR_WEIGHTS,
            "Stage 2: IQA Gatekeeper": cls.IQA_GATEKEEPER_WEIGHTS,
            "Stage 3: Symptom Classifier": cls.SYMPTOM_CLASSIFIER_WEIGHTS
        }
        
        all_systems_go = True
        
        for stage, path in models_to_check.items():
            if os.path.exists(path):
                print(f"[*] {stage} ... ONLINE")
            else:
                print(f"[!] {stage} ... OFFLINE (Missing: {path})")
                all_systems_go = False
                
        if not all_systems_go:
            raise FileNotFoundError("Critical pipeline weights are missing. Boot sequence aborted.")
            
        print("--- ALL SYSTEMS NOMINAL ---")
        return True

# If you run this file directly, it will test your paths.
if __name__ == "__main__":
    DirectoryLocator.verify_pipeline_readiness()
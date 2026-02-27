import sys
import os
import cv2
import torch
import numpy as np

# --- DYNAMIC PATH INJECTION ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- MODULE IMPORTS ---
from Backend.utils.config import DirectoryLocator
from Backend.imProcessor.crop_face import crop_primary_face
from Backend.imProcessor.resize import process_and_resize
from Backend.imProcessor.grayscale import convert_to_clinical_grayscale

# Note: Adjust these imports to match the exact file structures inside your Model folders
from Backend.Model.detection.detector import FaceDetector
from Backend.Model.quality.model import IQAModel  # <-- Corrected class name
from Backend.Model.classifier.model import SymptomClassifier 

# --- CUSTOM EXCEPTIONS ---
class LowQualityError(Exception):
    """Raised when an image fails the IQA gatekeeper threshold."""
    pass

class ClinicalPredictor:
    def __init__(self):
        print("--- BOOTING CLINICAL INFERENCE PIPELINE ---")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load Configurations
        self.face_weights = DirectoryLocator.FACE_DETECTOR_WEIGHTS
        self.iqa_weights = DirectoryLocator.IQA_GATEKEEPER_WEIGHTS
        self.classifier_weights = DirectoryLocator.SYMPTOM_CLASSIFIER_WEIGHTS

        # 2. Spin up the Engines
        print("[*] Initializing MobileNet Face Detector...")
        self.detector = FaceDetector(model_path=self.face_weights, confidence_threshold=0.6, device=self.device.type)
        
        print("[*] Initializing IQA Gatekeeper...")
        self.iqa_model = IQAModel(model_path=self.iqa_weights, device=self.device) # <-- Corrected instantiation
        
        print("[*] Initializing Final4 Symptom Classifier...")
        self.classifier = SymptomClassifier(model_path=self.classifier_weights, device=self.device)
        
        self.classes = ['acne', 'clear_face', 'darkspots', 'puffy_eyes', 'wrinkles']
        print("--- ALL ENGINES ONLINE ---")

    def predict(self, image_source):
        """
        Executes the full end-to-end diagnostic pipeline.
        """
        # Step 0: Load Image
        if isinstance(image_source, str):
            img_bgr = cv2.imread(image_source)
            if img_bgr is None:
                raise ValueError(f"[!] Error: Could not load image from {image_source}")
        else:
            img_bgr = image_source

        # Step 1: Isolate Target (Cropper)
        cropped_face = crop_primary_face(img_bgr, self.detector)
        if cropped_face is None:
            return {"error": "No human face detected."}

        # Step 2: Clinical Resizing (224x224 + PNG lock)
        resized_face = process_and_resize(cropped_face, target_size=(224, 224))

        # Step 3: IQA Gatekeeper Check
        # Ensure 'evaluate' matches the actual method name in your IQAModel class
        quality_score = self.iqa_model.evaluate(resized_face)
        print(f"[*] IQA Scan Complete. Quality Score: {quality_score:.1f}/10")
        
        if quality_score < 5.0:
            raise LowQualityError(f"Image quality ({quality_score:.1f}/10) is below the clinical threshold of 5.0. Please capture a clearer photo.")

        # Step 4: Topological Filter (Grayscale)
        clinical_tensor_input = convert_to_clinical_grayscale(resized_face)

        # Step 5: Diagnostic Inference
        # Ensure 'predict' matches the actual method name in your SymptomClassifier class
        confidences = self.classifier.predict(clinical_tensor_input)

        # Step 6: The 5% Margin Logic
        top_class = max(confidences, key=confidences.get)
        max_score = confidences[top_class]

        threshold = max_score - 5.0
        
        final_results = {
            symptom: score 
            for symptom, score in confidences.items() 
            if score >= threshold
        }

        sorted_results = dict(sorted(final_results.items(), key=lambda item: item[1], reverse=True))

        return {
            "status": "success",
            "quality_score": quality_score,
            "primary_diagnosis": top_class,
            "margin_results": sorted_results
        }

if __name__ == "__main__":
    # --- PIPELINE TEST RUN ---
    TEST_IMG = r"C:\Users\dell\Pictures\Camera Roll\WIN_20260224_22_18_22_Pro.jpg"
    
    try:
        pipeline = ClinicalPredictor()
        print(f"\nScanning: {TEST_IMG}")
        
        report = pipeline.predict(TEST_IMG)
        
        print("\n" + "="*40)
        print(" FINAL DIAGNOSTIC REPORT ")
        print("="*40)
        print(f"Quality Score: {report['quality_score']:.1f}/10")
        print("\nDetected Symptoms (Within 5% Margin):")
        for symptom, conf in report['margin_results'].items():
            print(f" - {symptom.upper()}: {conf:.2f}%")
        print("="*40)
        
    except LowQualityError as lqe:
        print(f"\n[PIPELINE HALTED] {lqe}")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
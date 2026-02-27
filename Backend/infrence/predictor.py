import os
import sys
import cv2
import torch
import numpy as np
import logging
from typing import Union, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("DermaScan.Predictor")

# --- Dynamic Path Injection ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Backend.utils.config import DirectoryLocator
from Backend.imProcessor.crop_face import crop_primary_face
from Backend.imProcessor.resize import process_and_resize
from Backend.Model.detection.detector import FaceDetector
from Backend.Model.quality.model import IQAModel
from Backend.Model.classifier.model import SymptomClassifier

# Hardcoded fallback — used only if DirectoryLocator path is missing/wrong
_CLASSIFIER_WEIGHTS_FALLBACK = r"D:\Projects\DermaScanAI\Backend\Model\classifier\symptom_classifier_phased.pth"


def _resolve_classifier_path() -> str:
    """
    Returns the classifier weights path. 
    Prefers DirectoryLocator, falls back to the hardcoded path with a warning.
    """
    try:
        path = DirectoryLocator.SYMPTOM_CLASSIFIER_WEIGHTS
        if os.path.exists(path):
            return path
        logger.warning(
            f"DirectoryLocator path not found: '{path}'. "
            f"Falling back to hardcoded path."
        )
    except AttributeError:
        logger.warning("DirectoryLocator.SYMPTOM_CLASSIFIER_WEIGHTS not defined. Using fallback.")

    if not os.path.exists(_CLASSIFIER_WEIGHTS_FALLBACK):
        raise FileNotFoundError(
            f"Classifier weights not found at fallback path: {_CLASSIFIER_WEIGHTS_FALLBACK}\n"
            f"Please retrain the model or update DirectoryLocator.SYMPTOM_CLASSIFIER_WEIGHTS."
        )
    return _CLASSIFIER_WEIGHTS_FALLBACK


class ClinicalPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Running on device: {self.device}")

        self.detector  = FaceDetector(
            model_path=DirectoryLocator.FACE_DETECTOR_WEIGHTS,
            device=self.device.type
        )
        self.iqa_model = IQAModel(
            model_path=DirectoryLocator.IQA_GATEKEEPER_WEIGHTS,
            device=self.device
        )
        self.classifier = SymptomClassifier(
            model_path=_resolve_classifier_path(),
            device=self.device
        )

        logger.info("Pipeline Fully Operational.")

    def predict(self, image_source: Union[str, np.ndarray]) -> Dict[str, Any]:
        try:
            # --- 1. Load ---
            if isinstance(image_source, str):
                if not os.path.exists(image_source):
                    return {"status": "error", "message": f"File not found: {image_source}"}
                img_bgr = cv2.imread(image_source)
            else:
                img_bgr = image_source

            if img_bgr is None:
                return {"status": "error", "message": "Failed to decode image (cv2.imread returned None)"}

            # --- 2. Detect & Crop Face ---
            cropped_face = crop_primary_face(img_bgr, self.detector)
            if cropped_face is None:
                return {"status": "error", "message": "No face detected in the image"}

            # --- 3. Standardize to 224x224 ---
            resized_face = process_and_resize(cropped_face, target_size=(224, 224))

            # --- 4. Quality Gate ---
            quality_score = self.iqa_model.evaluate(resized_face) * 10
            logger.info(f"IQA Score: {quality_score:.2f}/10")
            if quality_score < 4.0:
                return {
                    "status": "rejected",
                    "reason": f"Image quality too low ({quality_score:.2f}/10). Please use a clearer photo."
                }

            # --- 5. Classify ---
            # SymptomClassifier._sanitize_input handles BGR→RGB and dtype internally
            confidences = self.classifier.predict(resized_face)

            # --- 6. Build Result ---
            top_class = max(confidences, key=confidences.get)
            logger.info(f"Prediction: {top_class} ({confidences[top_class]:.2f}%)")

            return {
                "status":           "success",
                "quality_score":    round(quality_score, 2),
                "primary_diagnosis": top_class,
                "confidence":       round(confidences[top_class], 2),
                "all_confidences":  confidences
            }

        except Exception as e:
            logger.exception(f"Pipeline Error: {e}")
            return {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    TEST_IMAGE = r"C:\Users\dell\Pictures\Camera Roll\WhatsApp Image 2026-02-27 at 2.36.03 PM.jpeg"

    pipeline = ClinicalPredictor()
    report   = pipeline.predict(TEST_IMAGE)

    print("\n--- PREDICTION REPORT ---")
    for k, v in report.items():
        print(f"  {k:<22}: {v}")
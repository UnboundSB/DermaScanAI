import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

class DermaScanInference:
    """
    Master inference class that chains SSDLite Face Detection 
    with the custom EfficientNet-B0 Symptom Classifier.
    """
    def __init__(self, classifier_weights_path, ssdlite_weights_path=None, device=None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[System] Initializing DermaScan AI on: {self.device}")
        
        # Must perfectly match the alphabetical order from your dataset folders
        self.classes = ['acne', 'clear_face', 'darkspots', 'puffy_eyes', 'wrinkles']
        
        # 1. Load the Face Detector (SSDLite)
        self.face_detector = self._initialize_detector(ssdlite_weights_path)
        
        # 2. Load the Symptom Classifier (EfficientNet-B0)
        self.classifier = self._initialize_classifier(classifier_weights_path)
        
        # 3. Standardization Pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _initialize_detector(self, weights_path):
        """Loads SSDLite for face isolation."""
        # Using the standard PyTorch SSDLite MobileNetV3 architecture
        detector = models.detection.ssdlite320_mobilenet_v3_large(
            weights=models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        )
        
        # If you have custom trained SSDLite weights for faces, load them here
        if weights_path and os.path.exists(weights_path):
            detector.load_state_dict(torch.load(weights_path, map_location=self.device))
            
        detector.to(self.device)
        detector.eval()
        return detector

    def _initialize_classifier(self, weights_path):
        """Loads your champion EfficientNet-B0 model."""
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Missing classifier weights at: {weights_path}")
            
        model = models.efficientnet_b0(weights=None)
        # Modify the classification head to match your 5 final classes
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(self.classes))
        
        model.load_state_dict(torch.load(weights_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def extract_face(self, image_path):
        """Finds the largest face in the image and returns a cropped PIL Image."""
        original_img = Image.open(image_path).convert("RGB")
        img_tensor = transforms.ToTensor()(original_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.face_detector(img_tensor)[0]
            
        # Filter for the highest confidence bounding box
        # (Assuming class '1' is person/face in standard COCO SSDLite)
        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        
        if len(boxes) == 0 or scores[0] < 0.5:
            print("[Warning] No confident face detected. Analyzing full image.")
            return original_img
            
        # Take the box with the highest confidence score
        best_box = boxes[0]
        x1, y1, x2, y2 = map(int, best_box)
        
        # Optional: Add padding to the bounding box so we don't cut off the chin/forehead
        padding = 30
        width, height = original_img.size
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        
        face_crop = original_img.crop((x1, y1, x2, y2))
        return face_crop

    def analyze(self, image_path):
        """The master function: Extracts face, runs analysis, returns probabilities."""
        if not os.path.exists(image_path):
            return {"error": "Image file not found."}
            
        # 1. Isolate the target area
        face_img = self.extract_face(image_path)
        
        # 2. Transform for the classifier
        input_tensor = self.preprocess(face_img).unsqueeze(0).to(self.device)
        
        # 3. Neural Network Inference
        with torch.no_grad():
            logits = self.classifier(input_tensor)
            # Apply Softmax to convert raw math into percentages (0.0 to 1.0)
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
            
        probs_array = probabilities.cpu().numpy()
        
        # 4. Package the results cleanly for the backend
        results = {}
        for i, cls_name in enumerate(self.classes):
            results[cls_name] = round(float(probs_array[i]) * 100, 2)
            
        # Sort by highest confidence
        sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
        
        top_diagnosis = list(sorted_results.keys())[0]
        top_confidence = list(sorted_results.values())[0]
        
        return {
            "status": "success",
            "primary_diagnosis": top_diagnosis,
            "confidence": top_confidence,
            "all_probabilities": sorted_results
        }

# --- QUICK TEST HARNESS ---
# This only runs if you execute the file directly, allowing you to test it instantly.
if __name__ == "__main__":
    # Point this to your winning model
    WEIGHTS = "symptom_classifier_final1.pth"
    
    # Point this to a random test photo of a face (outside your dataset)
    TEST_IMAGE = r"D:\Projects\DermaScanAI\test_image.jpg" 
    
    # Initialize the engine
    derma_engine = DermaScanInference(classifier_weights_path=WEIGHTS)
    
    # Run the diagnosis
    print(f"\nAnalyzing: {TEST_IMAGE}")
    report = derma_engine.analyze(TEST_IMAGE)
    
    if "error" in report:
        print(report["error"])
    else:
        print("\n--- DIAGNOSIS REPORT ---")
        print(f"Primary Symptom: {report['primary_diagnosis'].upper()} ({report['confidence']}%)")
        print("\nFull Breakdown:")
        for symptom, conf in report['all_probabilities'].items():
            print(f"  - {symptom:<15}: {conf}%")
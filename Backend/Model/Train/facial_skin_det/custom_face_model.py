import torch
import cv2
import numpy as np
import torchvision
from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights
from torchvision.transforms import functional as F

class FaceDetector:
    def __init__(self, model_path, confidence_threshold=0.5, device=None):
        self.threshold = confidence_threshold
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        print(f"[FaceDetector] Initializing on {self.device}...")

        try:
            weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
            self.model.head.classification_head.num_classes = 2
            
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            print(f"[FaceDetector] Model loaded successfully from {model_path}")
            
        except Exception as e:
            raise RuntimeError(f"[Error] Failed to load model: {e}")

    def detect(self, image_source):
        if isinstance(image_source, str):
            image = cv2.imread(image_source)
        elif isinstance(image_source, np.ndarray):
            image = image_source
        else:
            raise TypeError("Input must be a file path string or numpy array.")

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(img_rgb).to(self.device).unsqueeze(0)

        with torch.no_grad():
            predictions = self.model(img_tensor)

        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        
        results = []
        for box, score in zip(boxes, scores):
            if score >= self.threshold:
                x1, y1, x2, y2 = box.astype(int)
                results.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "score": float(score)
                })
        return results
import torch
import cv2
import numpy as np
import torchvision
from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights
from torchvision.transforms import functional as F

class FaceDetector:
    def __init__(self, model_path, confidence_threshold=0.5, device=None):
        """
        Initializes the Face Detector.
        
        Args:
            model_path (str): Path to the .pth weights file.
            confidence_threshold (float): Minimum confidence to report a face (0.0 to 1.0).
            device (str): 'cuda' or 'cpu'. If None, automatically detects.
        """
        self.threshold = confidence_threshold
        
        # 1. Device Configuration
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        print(f"[FaceDetector] Initializing on {self.device}...")

        # 2. Load Architecture (SSDLite320 - MobileNetV3)
        # We must use the exact same structure used during training
        try:
            weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
            
            # modify the head to match our 2-class training (Background + Face)
            self.model.head.classification_head.num_classes = 2
            
            # 3. Load Trained Weights
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            
            # 4. Set to Evaluation Mode (Critical for inference!)
            self.model.to(self.device)
            self.model.eval()
            print(f"[FaceDetector] Model loaded successfully from {model_path}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"[Error] Weights not found at {model_path}")
        except Exception as e:
            raise RuntimeError(f"[Error] Failed to load model: {e}")

    def detect(self, image_source):
        """
        Detects faces in an image.
        
        Args:
            image_source: Can be a file path (str) OR a numpy array (cv2 image).
            
        Returns:
            list of dicts: [{'box': [x1, y1, x2, y2], 'score': 0.95}, ...]
        """
        # A. Handle Input (Path vs Array)
        if isinstance(image_source, str):
            image = cv2.imread(image_source)
            if image is None:
                raise ValueError(f"Could not open image at {image_source}")
        elif isinstance(image_source, np.ndarray):
            image = image_source
        else:
            raise TypeError("Input must be a file path string or numpy array.")

        # B. Preprocess
        # OpenCV uses BGR, PyTorch expects RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to Tensor (scales to 0-1 automatically)
        img_tensor = F.to_tensor(img_rgb).to(self.device)
        
        # Add Batch Dimension (C, H, W) -> (1, C, H, W)
        img_tensor = img_tensor.unsqueeze(0)

        # C. Inference
        with torch.no_grad():
            predictions = self.model(img_tensor)

        # D. Post-Process
        # SSDLite returns a list of dictionaries (one per image in batch)
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

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    # Define paths
    WEIGHTS = r"D:\Projects\DermaScanAI\Backend\Model\detection\face_detector_final.pth"
    TEST_IMG = r"D:\Projects\DermaScanAI\datasets\face_detection\processed_640\images\0_Parade_marchingband_1_20.png" # Change to a real image path

    try:
        # 1. Initialize
        detector = FaceDetector(model_path=WEIGHTS, confidence_threshold=0.5)
        
        # 2. Run Detection
        if os.path.exists(TEST_IMG):
            faces = detector.detect(TEST_IMG)
            
            print(f"\n[Result] Found {len(faces)} faces:")
            for i, face in enumerate(faces):
                print(f"  Face {i+1}: Confidence {face['score']:.2f}, Box {face['box']}")
                
            # Optional: Visualization code if running locally
            img = cv2.imread(TEST_IMG)
            for face in faces:
                x1, y1, x2, y2 = face['box']
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{face['score']:.2f}", (x1, y1-10), 
                           cv2.LINE_AA, 0.5, (0, 255, 0), 1)
            
            cv2.imwrite("test_output.jpg", img)
            print("[Info] Saved visualization to 'test_output.jpg'")
        else:
            print("[Warning] Test image not found. Please set TEST_IMG to a valid path.")

    except Exception as e:
        print(e)
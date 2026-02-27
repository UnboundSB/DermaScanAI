import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2

class IQAModel:
    """
    Image Quality Assessment (IQA) Gatekeeper.
    Evaluates image clarity before allowing it into the diagnostic pipeline.
    """
    def __init__(self, model_path, device=None):
        # 1. Setup Device (Inherits from orchestrator if provided)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Initializing IQA Gatekeeper on: {self.device}")
        
        # 2. Verify Absolute Path (Provided by DirectoryLocator)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[!] Critical Error: IQA weights not found at: {model_path}")
            
        # 3. Initialize Architecture (EfficientNet-B0)
        self.model = models.efficientnet_b0(weights=None)
        
        # Rebuild the regression head to match your training architecture
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, 1)
        )
        
        # 4. Load Weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()  # CRITICAL: Switches off Dropout/BatchNorm updates for inference speed
        
        # 5. Pre-define Transforms
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(), # Added to handle numpy arrays seamlessly
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # 6. Warmup 
        self._warmup()

    def _warmup(self):
        """Runs a dummy tensor to prime the GPU/CPU for zero-latency first requests."""
        dummy_input = torch.zeros(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            self.model(dummy_input)

    def evaluate(self, image_source):
        """
        Evaluates the quality of the incoming image matrix.
        Renamed to 'evaluate' to match the orchestrator's expectations.
        """
        try:
            # Handle OpenCV Numpy Arrays (This is what the resizer hands off)
            if isinstance(image_source, np.ndarray):
                # Convert BGR (OpenCV) to RGB before it hits the PIL/Torch transforms
                image = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
            # Handle direct file paths
            elif isinstance(image_source, str):
                image = cv2.imread(image_source)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                raise TypeError("Input must be a numpy array or file path.")

            # Preprocess (Numpy -> PIL -> Tensor -> Normalize)
            input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                output = self.model(input_tensor)
                score = output.item()

            return max(0.0, score)

        except Exception as e:
            print(f"[!] IQA Inference Error: {e}")
            return 0.0

if __name__ == "__main__":
    # --- ISOLATED MODULE TEST ---
    # Update this path to where your weights actually are for local testing
    WEIGHTS_PATH = r"D:\Projects\DermaScanAI\Backend\Model\quality\iqa_gatekeeper_b0.pth"
    TEST_IMG = r"C:\Users\dell\Pictures\Camera Roll\WIN_20260224_22_18_22_Pro.jpg"
    
    if os.path.exists(WEIGHTS_PATH) and os.path.exists(TEST_IMG):
        gatekeeper = IQAModel(model_path=WEIGHTS_PATH)
        
        # Test feeding a raw numpy array just like the orchestrator will
        raw_cv2_image = cv2.imread(TEST_IMG)
        
        score = gatekeeper.evaluate(raw_cv2_image)
        print(f"\n[SUCCESS] Quality Score: {score:.2f}/10")
    else:
        print("[!] Missing weights or test image for local execution.")
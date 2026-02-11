import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

class IQAModel:
    def __init__(self, model_filename="iqa_gatekeeper_b0.pth"):
        # 1. Setup Device (Fastest available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 2. Locate Model File (Relative to this script)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, model_filename)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at: {model_path}")
        print('model found')
        # 3. Initialize Architecture (EfficientNet-B0)
        # We load without weights=IMAGENET first to be faster, then load our own.
        self.model = models.efficientnet_b0(weights=None)
        
        # Rebuild the regression head to match training
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, 1)
        )
        
        # 4. Load Weights and Optimize
        # map_location ensures it loads on CPU first if GPU is busy, then moves
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()  # CRITICAL: Switches off Dropout/BatchNorm updates for inference speed
        
        # 5. Pre-define Transforms (Done once, not per image)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # 6. Warmup (Optional but recommended)
        # Runs a dummy pass so the first user request isn't slow due to lazy loading
        self._warmup()

    def _warmup(self):
        dummy_input = torch.zeros(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            self.model(dummy_input)

    def predict(self, image_source):
        """
        Args:
            image_source: Can be a file path (str) or a PIL Image object.
        Returns:
            float: Quality score (0.0 to 1.0+)
        """
        try:
            # Handle file path vs PIL object
            if isinstance(image_source, str):
                image = Image.open(image_source).convert("RGB")
            else:
                image = image_source.convert("RGB")

            # Preprocess
            input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            # Inference (No Gradients = Fast)
            with torch.no_grad():
                output = self.model(input_tensor)
                score = output.item()

            return max(0.0, score)

        except Exception as e:
            print(f"Inference Error: {e}")
            return 0.0

# --- Usage Example (for testing) ---
if __name__ == "__main__":
    # Initialize once
    gatekeeper = IQAModel()
    
    # Run prediction
    # Replace with a real path to test
    test_path = "test_image.jpg" 
    print('start')
    if os.path.exists(test_path):
        score = gatekeeper.predict(test_path)
        print(f"Quality Score: {score:.4f}")
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms


class SymptomClassifier:
    def __init__(self, model_path, device=None):
        self.device = torch.device(device if isinstance(device, str) else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[*] Initializing SymptomClassifier on: {self.device}")

        self.classes = ['acne', 'clear_face', 'darkspots', 'puffy_eyes', 'wrinkles']

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[!] Weight file not found at: {model_path}")

        # Build model architecture (must match training exactly)
        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, len(self.classes))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # FIX: Removed Grayscale — must mirror the fixed training transforms exactly.
        # Grayscale was discarding color which is the primary diagnostic signal
        # for acne (redness), dark spots (pigmentation), and puffy eyes (discoloration).
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"[*] Loaded weights from: {model_path}")
        print(f"[*] Classes: {self.classes}")

    def _sanitize_input(self, img: np.ndarray) -> np.ndarray:
        """
        Guarantees the image handed to the PyTorch pipeline is:
          - dtype  : uint8
          - range  : 0-255
          - channels: RGB (3ch)
        Handles all the weird formats OpenCV and resizers can produce.
        """
        # --- 1. Fix dtype / range ---
        if img.dtype != np.uint8:
            if img.max() <= 1.1:                        # float [0,1]
                img = (img * 255).clip(0, 255).astype(np.uint8)
            else:                                        # float [0,255]
                img = img.clip(0, 255).astype(np.uint8)

        # --- 2. Fix channel count ---
        if len(img.shape) == 2:                         # grayscale HxW
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:                         # BGRA
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 3:                         # BGR  (standard OpenCV)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def predict(self, img_array: np.ndarray) -> dict:
        """
        Parameters
        ----------
        img_array : np.ndarray
            Raw image as returned by OpenCV (BGR, any dtype).

        Returns
        -------
        dict  {class_name: confidence_percentage}
            e.g. {'acne': 72.31, 'clear_face': 5.12, ...}
        """
        img_sanitized  = self._sanitize_input(img_array)
        input_tensor   = self.preprocess(img_sanitized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits        = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]

        probs = probabilities.cpu().numpy()
        return {cls: round(float(p) * 100, 2) for cls, p in zip(self.classes, probs)}
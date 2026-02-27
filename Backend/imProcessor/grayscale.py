import cv2
import numpy as np


def convert_to_clinical_grayscale(img_array):
    """
    Transforms a BGR image array into a 3-channel grayscale array.
    This preserves the 'EfficientNet' input shape (224, 224, 3) 
    while stripping out deceptive color information.
    """
    if img_array is None or img_array.size == 0:
        print("[!] Error: Grayscaler received an empty image array.")
        return None

    # 1. Convert to single-channel grayscale (Luminance only)
    # We use the REC.709 formula (standard for digital displays)
    gray_single = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # 2. Re-stack into 3 identical channels
    # EfficientNet was pre-trained on ImageNet (RGB), so it REQUIRES 3 channels.
    # We duplicate the grayscale channel to satisfy the architecture 
    # while keeping the data purely topological.
    gray_3channel = cv2.merge([gray_single, gray_single, gray_single])

    return gray_3channel

if __name__ == "__main__":
    import os
    # --- ISOLATED MODULE TEST ---
    print("--- TESTING CLINICAL GRAYSCALER ---")
    
    # Path to your test image from the camera roll
    TEST_IMG_PATH = r"C:\Users\dell\Pictures\Camera Roll\WIN_20260224_22_18_22_Pro.jpg"
    
    if os.path.exists(TEST_IMG_PATH):
        raw_bgr = cv2.imread(TEST_IMG_PATH)
        
        # Run Transformation
        clinical_gray = convert_to_clinical_grayscale(raw_bgr)
        
        if clinical_gray is not None:
            print(f"[SUCCESS] Image converted to 3-channel grayscale. Shape: {clinical_gray.shape}")
            
            # Show the difference
            cv2.imshow("Original BGR", cv2.resize(raw_bgr, (500, 500)))
            cv2.imshow("Clinical Grayscale (Topological View)", cv2.resize(clinical_gray, (500, 500)))
            
            print("\nPress any key on the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print(f"[!] Test image not found at {TEST_IMG_PATH}")
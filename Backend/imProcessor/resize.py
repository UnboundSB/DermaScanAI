import cv2
import numpy as np

def _conv_to_png(img_array):
    """
    Forces the numpy array through a lossless PNG encoding/decoding cycle in memory.
    This locks the pixel data into a lossless state before any geometric transformations occur.
    """
    # 1. Encode the raw array into a PNG memory buffer
    success, encoded_buffer = cv2.imencode('.png', img_array)
    if not success:
        raise ValueError("[!] Critical Error: Failed to encode image matrix to PNG format.")
    
    # 2. Decode the PNG buffer back into a numpy array
    png_array = cv2.imdecode(encoded_buffer, cv2.IMREAD_COLOR)
    return png_array

def process_and_resize(img_array, target_size=(224, 224)):
    """
    Receives the cropped face array, enforces PNG format, 
    and shrinks it down to the exact 224x224 dimensions required by EfficientNet.
    """
    if img_array is None or img_array.size == 0:
        print("[!] Error: Resizer received an empty image array.")
        return None

    # Step 1: Autonomous PNG Conversion
    png_ready_array = _conv_to_png(img_array)

    # Step 2: Clinical Resizing
    # Using INTER_AREA is mandatory here. It prevents 'moiré' patterns and preserves 
    # the micro-textures of wrinkles and acne when shrinking the image matrix.
    resized_array = cv2.resize(png_ready_array, target_size, interpolation=cv2.INTER_AREA)
    
    return resized_array

if __name__ == "__main__":
    # --- ISOLATED MODULE TEST ---
    print("--- TESTING RESIZE MODULE ---")
    
    # Create a dummy image (e.g., a large 1000x1000 noisy image mimicking a face crop)
    print("[*] Generating dummy 1000x1000 raw image array...")
    dummy_crop = np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)
    
    print(f"[*] Original Shape: {dummy_crop.shape}")
    
    # Run it through the pipeline
    final_224_image = process_and_resize(dummy_crop)
    
    if final_224_image is not None:
        print(f"[SUCCESS] Image converted to PNG structure and resized to: {final_224_image.shape}")
        
        # Save a physical copy just to prove the PNG conversion holds
        cv2.imwrite("test_resized_output.png", final_224_image)
        print("[*] Test output saved to disk as 'test_resized_output.png'")
import os
import cv2

# --- CONFIGURATION ---
BASE_DIR = r"D:\Projects\DermaScanAI\datasets\skin_ageing_symptoms"

# We read FROM the garbage folder
GARBAGE_DIR = os.path.join(BASE_DIR, "deleted_garbage")
# We rescue TO the training folder
RESCUE_DIR = os.path.join(BASE_DIR, "dataset_ready_for_training")

CLASSES = ["acne", "darkspots", "wrinkles", "puffy_eyes", "clear_face"]

def main():
    print("--- STARTING GARBAGE RESCUE MISSION ---")
    print("Controls:")
    print("  [r]     : Rotate 90° Clockwise")
    print("  [a]     : Accept & Rescue (Moves back to training folder)")
    print("  [Space] : Skip (Leaves in garbage bin)")
    print("  [d]     : Delete (Permanently destroys file from disk)")
    print("  [q/Esc] : Quit")
    print("---------------------------------------")
    
    cv2.namedWindow("Rescue Mission", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Rescue Mission", 600, 600)

    for cls in CLASSES:
        garbage_folder = os.path.join(GARBAGE_DIR, cls)
        rescue_folder = os.path.join(RESCUE_DIR, cls)
        
        if not os.path.exists(garbage_folder): 
            continue
            
        os.makedirs(rescue_folder, exist_ok=True)
        images = [f for f in os.listdir(garbage_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images: continue
        print(f"\nReviewing '{cls}' ({len(images)} rejected images)...")
        
        for img_name in images:
            img_path = os.path.join(garbage_folder, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            while True:
                # Create a copy just for displaying text so we don't save text onto the image
                display_img = img.copy()
                
                # Black overlay for text readability
                cv2.rectangle(display_img, (0, 0), (display_img.shape[1], 40), (0, 0, 0), -1)
                info_text = f"[{cls}] {img_name} | [r]Rot | [a]Rescue | [Space]Skip | [d]Del | [q]Quit"
                cv2.putText(display_img, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                
                cv2.imshow("Rescue Mission", display_img)
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord('r'):
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    print(f"Rotated: {img_name}")
                    
                elif key == ord('a'):
                    # Rescue: Save the image to the training folder and remove from garbage
                    rescue_path = os.path.join(rescue_folder, img_name)
                    cv2.imwrite(rescue_path, img)
                    os.remove(img_path)
                    print(f"[RESCUED] -> {cls}/{img_name}")
                    break
                    
                elif key == ord(' '):
                    # Skip: Leave it in the garbage, go to next
                    print(f"Skipped: {img_name} (Left in garbage)")
                    break
                    
                elif key == ord('d'):
                    # Delete permanently from your hard drive
                    os.remove(img_path)
                    print(f"[TRASHED] Permanently deleted: {img_name}")
                    break
                    
                elif key == ord('q') or key == 27:
                    print("\nExiting rescue mission...")
                    cv2.destroyAllWindows()
                    return

    cv2.destroyAllWindows()
    print("\n--- RESCUE MISSION COMPLETE ---")

if __name__ == "__main__":
    main()
import os
import cv2
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- CONFIGURATION (VERIFIED) ---
RAW_IMG_DIR = r"D:\Projects\DermaScanAI\datasets\face_detection\raw\WIDER_train\images"
RAW_ANNO_FILE = r"D:\Projects\DermaScanAI\datasets\face_detection\raw\wider_face_split\wider_face_train_bbx_gt.txt"

# Output Location
BASE_PROC_DIR = r"D:\Projects\DermaScanAI\datasets\face_detection\processed_640"
PROC_IMG_DIR = os.path.join(BASE_PROC_DIR, "images")
PROC_LABEL_DIR = os.path.join(BASE_PROC_DIR, "labels")
EDA_DIR = os.path.join(BASE_PROC_DIR, "eda_reports")

# Settings
TARGET_SIZE = (640, 640)
PNG_COMPRESSION = 3  # 0-9 (3 is optimized for speed/size)

def parse_wider_annotations(anno_file):
    """
    Parses WIDER FACE text file.
    Returns a list of entries.
    """
    if not os.path.exists(anno_file):
        print(f"[Error] Annotation file not found at: {anno_file}")
        return []

    print(f"[Init] Reading annotations from: {anno_file}")
    with open(anno_file, 'r') as f:
        lines = f.readlines()

    entries = []
    i = 0
    while i < len(lines):
        filename = lines[i].strip()
        try:
            num_faces = int(lines[i+1].strip())
        except ValueError:
            i += 1
            continue
            
        boxes = []
        if num_faces > 0:
            for j in range(num_faces):
                # x, y, w, h
                coords = list(map(float, lines[i + 2 + j].strip().split()))
                # Valid box check (width and height must be > 0)
                if coords[2] > 0 and coords[3] > 0:
                    boxes.append(coords[:4])
            i += 2 + num_faces
        else:
            i += 3 

        if len(boxes) > 0:
            entries.append({"filename": filename, "boxes": boxes})
            
    return entries

def process_single_image(entry):
    """
    1. Loads Image
    2. Resizes to 640x640
    3. Normalizes Coordinates (YOLO Format)
    4. Saves PNG & TXT
    Returns: Lightweight stats (list of box dimensions) or None
    """
    img_path = os.path.join(RAW_IMG_DIR, entry['filename'])
    
    # Check if exists
    if not os.path.exists(img_path):
        # WIDER FACE often uses subfolders like '0--Parade/img.jpg'
        # Verification:
        # print(f"Missing: {img_path}") 
        return None

    # Load
    img = cv2.imread(img_path)
    if img is None:
        return None

    h_orig, w_orig = img.shape[:2]
    
    # Resize
    # INTER_LINEAR is faster/lighter than INTER_AREA
    img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    
    # Calculate Scale Factors
    scale_x = TARGET_SIZE[0] / w_orig
    scale_y = TARGET_SIZE[1] / h_orig
    
    label_lines = []
    box_stats = [] # Store width/height for EDA
    
    for box in entry['boxes']:
        x, y, w, h = box
        
        # 1. Scale Box to 640x640 space
        x_new = x * scale_x
        y_new = y * scale_y
        w_new = w * scale_x
        h_new = h * scale_y
        
        # 2. Normalize to 0-1 (YOLO Format)
        # Center X, Center Y, Width, Height
        cx = (x_new + w_new / 2) / TARGET_SIZE[0]
        cy = (y_new + h_new / 2) / TARGET_SIZE[1]
        wn = w_new / TARGET_SIZE[0]
        hn = h_new / TARGET_SIZE[1]
        
        # Clamp (Safety to avoid > 1.0)
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        wn = max(0, min(1, wn))
        hn = max(0, min(1, hn))
        
        # YOLO: <class_id> <x_center> <y_center> <width> <height>
        label_lines.append(f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")
        
        # Stats (Absolute pixels in 640x640 space)
        box_stats.append((w_new, h_new))

    # Save Image (PNG)
    # Replace slashes in filename (0--Parade/img.jpg -> 0--Parade_img.jpg)
    flat_name = entry['filename'].replace("/", "_")
    name_no_ext = os.path.splitext(flat_name)[0]
    
    save_img_path = os.path.join(PROC_IMG_DIR, name_no_ext + ".png")
    cv2.imwrite(save_img_path, img_resized, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
    
    # Save Label (TXT)
    save_lbl_path = os.path.join(PROC_LABEL_DIR, name_no_ext + ".txt")
    with open(save_lbl_path, 'w') as f:
        f.write("\n".join(label_lines))
        
    return box_stats

def generate_eda(all_box_stats):
    print("--- GENERATING EDA PLOTS ---")
    if not os.path.exists(EDA_DIR):
        os.makedirs(EDA_DIR)
        
    # Unpack stats
    widths = [b[0] for b in all_box_stats]
    heights = [b[1] for b in all_box_stats]
    
    sns.set_theme(style="whitegrid")
    
    # PLOT 1: Face Size Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(widths, bins=50, color="blue", alpha=0.6, label="Width")
    sns.histplot(heights, bins=50, color="red", alpha=0.4, label="Height")
    plt.title(f"Face Dimensions in {TARGET_SIZE} Image")
    plt.xlabel("Pixels")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(os.path.join(EDA_DIR, "1_face_size_dist.png"))
    plt.close()
    
    # PLOT 2: Aspect Ratio (Width vs Height)
    plt.figure(figsize=(8, 8))
    # Sample only first 2000 points to keep plot fast
    plt.scatter(widths[:2000], heights[:2000], alpha=0.1, s=10, c='purple')
    plt.plot([0, 200], [0, 200], 'k--', label="Square (1:1)")
    plt.title("Face Aspect Ratio (Sampled)")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.legend()
    plt.savefig(os.path.join(EDA_DIR, "2_aspect_ratio.png"))
    plt.close()
    
    # VISUAL SANITY CHECK
    # Draw boxes on the LAST processed image
    if os.listdir(PROC_IMG_DIR):
        last_files = sorted(os.listdir(PROC_IMG_DIR))[-1]
        last_img_path = os.path.join(PROC_IMG_DIR, last_files)
        last_lbl_path = os.path.join(PROC_LABEL_DIR, last_files.replace(".png", ".txt"))
        
        if os.path.exists(last_img_path) and os.path.exists(last_lbl_path):
            chk_img = cv2.imread(last_img_path)
            h, w = chk_img.shape[:2]
            with open(last_lbl_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = list(map(float, line.split()))
                cx, cy, wn, hn = parts[1:]
                
                # De-normalize
                w_box = int(wn * w)
                h_box = int(hn * h)
                x_box = int((cx * w) - (w_box/2))
                y_box = int((cy * h) - (h_box/2))
                
                cv2.rectangle(chk_img, (x_box, y_box), (x_box+w_box, y_box+h_box), (0, 255, 0), 2)
                
            cv2.imwrite(os.path.join(EDA_DIR, "3_sanity_check.jpg"), chk_img)
            print(f"-> Saved visual sanity check: {os.path.join(EDA_DIR, '3_sanity_check.jpg')}")

def run_pipeline():
    # Make Dirs
    for d in [PROC_IMG_DIR, PROC_LABEL_DIR, EDA_DIR]:
        os.makedirs(d, exist_ok=True)

    # 1. Parse
    entries = parse_wider_annotations(RAW_ANNO_FILE)
    if not entries:
        print("No entries found. Check your paths!")
        return

    print(f"Found {len(entries)} images. Processing sequentially...")
    
    all_stats = []
    count = 0
    
    # 2. Process Loop
    for entry in tqdm(entries):
        stats = process_single_image(entry)
        
        if stats is not None:
            all_stats.extend(stats)
            count += 1
            
        # RAM SAVER: Clean memory every 200 images
        if count % 200 == 0:
            gc.collect()

    print(f"Processed {count} images.")
    
    # 3. Generate Reports
    generate_eda(all_stats)
    print("DONE.")

if __name__ == "__main__":
    run_pipeline()
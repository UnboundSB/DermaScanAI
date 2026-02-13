import os
import requests
import tarfile
from tqdm import tqdm # pip install tqdm (standard progress bar)

def download_lfw_dataset(target_folder="lfw_dataset"):
    # Official UMass Amherst LFW dataset URL
    url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    filename = "lfw.tgz"

    print(f"[Init] Preparing to download Labeled Faces in the Wild (LFW)...")
    
    # 1. Download
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    if not os.path.exists(filename):
        print(f"[Download] Fetching {filename} (~180MB)...")
        block_size = 1024 # 1 Kibibyte
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        
        with open(filename, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
    else:
        print(f"[Info] {filename} already exists. Skipping download.")

    # 2. Extract
    print(f"[Extract] Unpacking to '{target_folder}/'...")
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    try:
        with tarfile.open(filename, "r:gz") as tar:
            # The tar contains a root folder 'lfw', we strip it to keep things clean
            # Actually, standard extraction is safer to avoid path issues
            tar.extractall(path=target_folder)
            
        print("-" * 40)
        print(f"[Success] Dataset ready in: {os.path.abspath(target_folder)}")
        print(f"[Stats] Contains 13,000+ images of 5,749 identities.")
        
    except Exception as e:
        print(f"[Error] Extraction failed: {e}")

if __name__ == "__main__":
    # You might need to install requests and tqdm if you haven't
    # pip install requests tqdm
    download_lfw_dataset()
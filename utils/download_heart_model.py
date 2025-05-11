import os
import gdown

MODEL_DIR = "models"
MODEL_NAME = "unet_finetuned.h5"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# Google Drive direct download URL using file ID
GOOGLE_DRIVE_ID = "1didPGIAJ9IZaRmLjSXy8QEVu4NVbPmdB"
GOOGLE_DRIVE_URL = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"

def download_heart_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print("Downloading Heart model...")
        gdown.download(GOOGLE_DRIVE_URL, MODEL_PATH, quiet=False)
        print("Download completed.")
    else:
        print("Heart model already exists.")
    
    return MODEL_PATH

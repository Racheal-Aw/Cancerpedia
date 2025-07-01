import gdown
import zipfile
import os

# Google Drive file ID (replace with your own if needed)
file_id = "1K6x4FU4A4aBIP7_agPKtLRXojtgbgoUq"
zip_file = "embedding_model_cancer.zip"
output_folder = "embedding_model_cancer"

# Download the file if it doesn't exist
if not os.path.exists(zip_file):
    print("⬇️ Downloading model zip...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", zip_file, quiet=False)
else:
    print("✅ Zip file already exists.")

# Extract the zip if not already done
if not os.path.exists(output_folder):
    print("📦 Extracting model...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(output_folder)
    os.remove(zip_file)
    print(f"✅ Extracted to '{output_folder}/' and removed zip file.")
else:
    print("✅ Model already extracted.")

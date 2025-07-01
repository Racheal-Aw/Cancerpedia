import os
import gdown
import zipfile
import streamlit as st


from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage

# --- 1. Download & unzip embedding model if needed ---

file_id = "1K6x4FU4A4aBIP7_agPKtLRXojtgbgoUq"  # Replace with your actual Google Drive file ID (just the ID)
zip_file = "embedding_model_cancer.zip"
output_folder = "embedding_model_cancer"

if not os.path.exists(zip_file) and not os.path.exists(output_folder):
    print("⬇️ Downloading model zip...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", zip_file, quiet=False)
else:
    print("✅ Zip file or extracted folder already exists.")

if not os.path.exists(output_folder):
    print("📦 Extracting model...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(output_folder)
    os.remove(zip_file)
    print(f"✅ Extracted to '{output_folder}/' and removed zip file.")
else:
    print("✅ Model already extracted.")

# embeddings
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings_folder = "./embedding_model_cancer"  # make sure your local folder name matches here

embeddings = HuggingFaceEmbedding(
    model_name=embedding_model,
    cache_folder=embeddings_folder,
)


# Load vector index from storage
storage_context = StorageContext.from_defaults(persist_dir="vector_index")
vector_index = load_index_from_storage(storage_context, embed_model=embeddings)

st.write("Embedding and vector index loaded successfully!")


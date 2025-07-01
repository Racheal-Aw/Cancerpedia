import os
import gdown
import zipfile
import streamlit as st

from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole


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


# --- 2. Setup embedding and vector index ---

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_cache_folder = f"./{output_folder}/"

embeddings = HuggingFaceEmbedding(
    model_name=embedding_model_name,
    cache_folder=embedding_cache_folder,
)

storage_context = StorageContext.from_defaults(persist_dir="vector_index")
vector_index = load_index_from_storage(storage_context, embed_model=embeddings)

retriever = vector_index.as_retriever(similarity_top_k=2)


# --- 3. Setup LLM and Chat Engine ---

model_name = "llama3-70b-8192"
llm = Groq(
    model=model_name,
    # token=st.secrets["GROQ_API_KEY"],  # Uncomment and set if needed
)

prefix_messages = [
    ChatMessage(role=MessageRole.SYSTEM, content="You are a kind and helpful chatbot having a conversation with a human."),
    ChatMessage(role=MessageRole.SYSTEM, content=(
        "You may use background information to improve your answers. "
        "You may also answer new questions on unrelated topics if asked. "
        "But do not say things like 'According to the text' or refer to any document. "
        "Answer naturally and directly, as if you're speaking from your own knowledge."
    )),
    ChatMessage(role=MessageRole.SYSTEM, content="Keep your answers short, clear, and conversational."),
]

memory = ChatMemoryBuffer.from_defaults()

@st.cache_resource
def init_bot():
    return ContextChatEngine(
        llm=llm, retriever=retriever, memory=memory, prefix_messages=prefix_messages
    )

rag_bot = init_bot()


# --- 4. Streamlit UI ---

st.title("Cancerpedia Chat")
st.caption("Your friendly, trustworthy cancer knowledge companion.")

st.markdown("""
👋 **Welcome to Cancerpedia Chat!**

Curious about cancer, treatments, or medical terms you've come across?  
This chatbot is here to help you explore verified cancer education materials in plain language.  
Whether you're a patient, caregiver, or just learning, you're in the right place.

💡 *Try asking things like:*  
- "What is chemotherapy?"  
- "How do cancer cells grow?"  
- "What does BRCA1 mean?"  

📚 *Powered by a curated library of cancer education resources.*
""")

# Sidebar: Quick Search + FAQ
with st.sidebar:
    st.markdown("## 🔍 Quick Search")
    search_input = st.text_input("Enter a keyword or topic")

    st.markdown("## 📌 Frequently Asked")
    faq_questions = [
        "What is cancer?",
        "How does immunotherapy work?",
        "Can cancer be inherited?",
        "What are the early signs?",
        "How is cancer staged?",
        "What is palliative care?",
        "What are common cancer treatments?",
        "How can I support a loved one with cancer?",
        "What is the role of a pathologist in cancer diagnosis?",
        "What is the difference between benign and malignant tumors?",
        "What lifestyle changes can help prevent cancer?",
        "What is the role of clinical trials in cancer treatment?",
        "How can I manage side effects of cancer treatment?",
        "What is the importance of early detection?",
        "What are the most common types of cancer?",
        "What are common signs and symptoms of cancer?",
        "What is the role of nutrition in cancer care?",
        "How can I cope with a cancer diagnosis?",
        "When should I see a doctor about possible cancer symptoms?",
        "What is the difference between chemotherapy and radiation therapy?",
        "What are the potential side effects of cancer treatment?",
        "Are unexplained weight loss or fatigue signs of cancer?"
    ]
    selected_faq = st.selectbox("Select a common question:", [""] + faq_questions)

    # Set session state input based on sidebar selections
    if search_input:
        st.info(f"Searching for: *{search_input}*")
        st.session_state['user_input'] = search_input
    elif selected_faq:
        st.info(f"You selected: *{selected_faq}*")
        st.session_state['user_input'] = selected_faq


# --- 5. Display chat history ---

for message in rag_bot.chat_history:
    with st.chat_message(message.role):
        st.markdown(message.blocks[0].text)


# --- 6. Process user input from sidebar or chat input ---

def process_user_message(user_text):
    st.chat_message("human").markdown(user_text)
    with st.spinner("Know all about cancer..."):
        answer = rag_bot.chat(user_text)
        response = answer.response
        with st.chat_message("assistant"):
            st.markdown(response)

# Process sidebar input if available
if 'user_input' in st.session_state and st.session_state['user_input']:
    process_user_message(st.session_state['user_input'])
    st.session_state['user_input'] = ""  # reset after processing

# Process chat input box
if prompt := st.chat_input("Curious minds wanted!"):
    process_user_message(prompt)

           

           

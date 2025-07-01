import os

from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
import streamlit as st

# llm
model = "llama3-70b-8192"

llm = Groq(
    model=model,
    # token=st.secrets["GROQ_API_KEY"], # when you're running local
)

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
    
# embeddings
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings_folder = "./embedding_model/"

embeddings = HuggingFaceEmbedding(
    model_name=embedding_model,
    cache_folder=embeddings_folder,
)

# load Vector Database
# allow_dangerous_deserialization is needed. Pickle files can be modified to deliver a malicious payload that results in execution of arbitrary code on your machine
storage_context = StorageContext.from_defaults(persist_dir="vector_index")
vector_index = load_index_from_storage(storage_context, embed_model=embeddings)

# retriever
retriever = vector_index.as_retriever(similarity_top_k=2)

# prompt
prefix_messages = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content ="You are a kind and helpful chatbot having a conversation with a human."
    ),
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=(
            "You may use background information to improve your answers."
            "You may also answer new questions on unrelated topics if asked."
            "But do not say things like 'According to the text' or refer to any document. "
            "Answer naturally and directly, as if you're speaking from your own knowledge."
        )
    ),
    ChatMessage(
        role=MessageRole.SYSTEM,
        content ="Keep your answers short, clear, and conversational."
    ),
]

# memory
memory = ChatMemoryBuffer.from_defaults()


# bot with memory
@st.cache_resource
def init_bot():
    return ContextChatEngine(
        llm=llm, retriever=retriever, memory=memory, prefix_messages=prefix_messages
    )


rag_bot = init_bot()

##### streamlit #####

st.title("Cancerpedia Chat")
st.caption("Your friendly, trustworthy cancer knowledge companion.")

# Welcome Message
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

with st.sidebar:
    st.markdown("## 🔍 Quick Search")  # Sidebar header
    search_input = st.text_input("Enter a keyword or topic")  # Text input field

    if search_input:
        st.info(f"Searching for: *{search_input}*")  # Feedback message
        st.session_state['user_input'] = search_input  # Store in session state

with st.sidebar:
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
        "What are common signs and symptoms of cancer",
        "What is the role of nutrition in cancer care?",
        "How can I cope with a cancer diagnosis?"
        "When should I see a doctor about possible cancer symptoms?",
        "What is the difference between chemotherapy and radiation therapy?",
        "What are the potential side effects of cancer treatment?",
        "Are unexplained weight loss or fatigue signs of cancer?"
    ]

    selected_faq = st.selectbox("Select a common question:", [""] + faq_questions)

    if selected_faq:
        st.info(f"You selected: *{selected_faq}*")
        st.session_state['user_input'] = selected_faq


   
# Display chat messages from history on app rerun
for message in rag_bot.chat_history:
    with st.chat_message(message.role):
        st.markdown(message.blocks[0].text)

# React to user input
if prompt := st.chat_input("Curious minds wanted!"):
    # Display user message in chat message container
    st.chat_message("human").markdown(prompt)

    # Begin spinner before answering question so it's there for the duration
    with st.spinner("Know all about cancer..."):
        # send question to chain to get answer
        answer = rag_bot.chat(prompt)

        # extract answer from dictionary returned by chain
        response = answer.response

        # Display chatbot response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)

if selected_faq:
    st.chat_message("human").markdown(selected_faq)
    with st.spinner("Know all about cancer..."):
        answer = rag_bot.chat(selected_faq)
        response = answer.response
        with st.chat_message("assistant"):
            st.markdown(response)

if search_input:
    st.chat_message("human").markdown(search_input)
    with st.spinner("Know all about cancer..."):
        answer = rag_bot.chat(search_input)
        response = answer.response
        with st.chat_message("assistant"):
            st.markdown(response)
           

           

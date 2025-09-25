import os
import json
import pickle
import requests
import numpy as np
import faiss
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# --- Load API key ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Better Call Advoca!", page_icon="logo.png", layout="wide")

# --- Custom Theme (accent color #1475bf) ---
st.markdown("""
<style>
/* Accent color for buttons */
button[kind="primary"] {
    background-color: #1475bf !important;
    border-color: #1475bf !important;
    color: white !important;
}
button[kind="primary"]:hover {
    background-color: #125c99 !important;
    border-color: #125c99 !important;
}
/* Chat bubbles with accent borders */
[data-testid="stChatMessage-user"] {
    background-color: #e9f5ff;
    border: 1px solid #1475bf;
    border-radius: 12px;
    padding: 10px;
}
[data-testid="stChatMessage-assistant"] {
    background-color: #f7f7f8;
    border: 1px solid #1475bf;
    border-radius: 12px;
    padding: 10px;
}
/* Sidebar titles + powered by */
section[data-testid="stSidebar"] .stMarkdown h1, 
section[data-testid="stSidebar"] .stMarkdown h2, 
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #1475bf !important;
}
.powered-by {
    text-align: center;
    color: #1475bf;
    font-weight: bold;
}
/* Chat input box */
[data-testid="stChatInput"] > div {
    border: 2px solid #ccc !important;
    border-radius: 8px !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border: 2px solid #1475bf !important;
    box-shadow: 0 0 5px #1475bf !important;
}
/* Hide default footer */
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Sidebar with logo + branding ---
st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.title("Better Call Advoca!")
st.sidebar.markdown(
    "Your AI-powered legal assistant\n\n⚠️ *Advoca is an AI assistant, not a substitute for legal counsel.*"
)

# --- Powered by Galácticos ---
st.sidebar.markdown("---")
st.sidebar.markdown("<div class='powered-by'>⚡ Powered by Galácticos</div>", unsafe_allow_html=True)

# --- Main Header with logo ---
col1, col2 = st.columns([0.15, 0.85])
with col1:
    st.image("logo.png", width=80)
with col2:
    st.title("Better Call Advoca! – Legal Assistant")
st.caption("Ask your legal questions below 👇")

# -------------------------
# Paths for dataset & cache
# -------------------------
DATASET_DIR = "dataset"
INDEX_FILE = "faiss_index.bin"
EMB_FILE = "embeddings.npy"
DOC_FILE = "documents.pkl"
PROC_FILE = "processed_files.json"

# --- Cache Embedder ---
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# --- Cache FAISS Index + Documents ---
@st.cache_resource
def load_index_and_docs():
    if os.path.exists(INDEX_FILE):
        faiss_index = faiss.read_index(INDEX_FILE)
    else:
        faiss_index = faiss.IndexFlatL2(384)

    if os.path.exists(DOC_FILE):
        with open(DOC_FILE, "rb") as f:
            documents = pickle.load(f)
    else:
        documents = []

    return faiss_index, documents

faiss_index, documents = load_index_and_docs()

# --- Preprocess new files only once ---
@st.cache_data
def process_new_files():
    if os.path.exists(PROC_FILE):
        with open(PROC_FILE, "r") as f:
            processed_files = json.load(f)
    else:
        processed_files = {}

    new_data_added = False
    embeddings = []

    if os.path.exists(EMB_FILE):
        embeddings = np.load(EMB_FILE)
    else:
        embeddings = np.zeros((0, 384), dtype="float32")

    for fname in os.listdir(DATASET_DIR):
        path = os.path.join(DATASET_DIR, fname)
        if fname not in processed_files and fname.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                if lines:
                    docs_emb = embedder.encode(lines).astype("float32")
                    embeddings = np.vstack([embeddings, docs_emb])
                    faiss_index.add(docs_emb)
                    documents.extend(lines)
                    processed_files[fname] = len(lines)
                    new_data_added = True

    if new_data_added:
        np.save(EMB_FILE, embeddings)
        faiss.write_index(faiss_index, INDEX_FILE)
        with open(DOC_FILE, "wb") as f:
            pickle.dump(documents, f)
        with open(PROC_FILE, "w") as f:
            json.dump(processed_files, f)

    return True

process_new_files()

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Gemini Call ---
def call_gemini(question, context):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    headers = {"Content-Type": "application/json", "X-Goog-Api-Key": GEMINI_API_KEY}

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            "You are a helpful legal assistant called Better Call Advoca!.\n"
                            "Use the following retrieved legal cases/articles to answer the question.\n"
                            "Always include a disclaimer: 'Advoca is an AI assistant, not a substitute for legal counsel.'\n\n"
                            f"Retrieved context:\n{context}\n\n"
                            f"Question: {question}"
                        )
                    }
                ]
            }
        ]
    }

    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code == 200:
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return f"⚠️ Error {resp.status_code}: {resp.text}"

# --- User input ---
if user_input := st.chat_input("Enter your legal question..."):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if documents:
        query_vec = embedder.encode([user_input]).astype("float32")
        D, I = faiss_index.search(query_vec, k=3)
        retrieved_cases = [documents[i] for i in I[0] if i < len(documents)]
        retrieved_text = "\n\n".join(retrieved_cases)
    else:
        retrieved_text = "⚠️ No documents available in the dataset."

    answer = call_gemini(user_input, retrieved_text)

    st.session_state["messages"].append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

import streamlit as st
from pypdf import PdfReader
import chromadb
import ollama
import hashlib
import uuid

st.set_page_config(page_title="RAG Study Assistant", page_icon="📚", layout="wide")
st.title("📚 RAG Study Assistant")
st.caption("Upload PDF notes an" \
"d ask questions grounded in your documents.")

CHAT_MODEL = "qwen2.5:3b"
EMBED_MODEL = "nomic-embed-text"
DB_DIR = "./chroma_db"

@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(name="study_notes")

def extract_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((page_num, text))
    return pages

def chunk_text(text, chunk_size=900, overlap=150):
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks

def embed(text):
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]

def index_pdf(uploaded_file):
    collection = get_collection()
    pages = extract_pdf(uploaded_file)
    if not pages:
        return 0, "No extractable text found. This may be a scanned PDF."
    count = 0
    for page_num, page_text in pages:
        for part_num, chunk in enumerate(chunk_text(page_text)):
            doc_id = str(uuid.uuid4())
            collection.add(
                ids=[doc_id],
                documents=[chunk],
                embeddings=[embed(chunk)],
                metadatas=[{
                    "filename": uploaded_file.name,
                    "page": page_num,
                    "part": part_num
                }]
            )
            count += 1
    return count, None

def answer_question(question):
    collection = get_collection()
    if collection.count() == 0:
        return "Please upload and index a PDF first.", []
    results = collection.query(
        query_embeddings=[embed(question)],
        n_results=min(4, collection.count())
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    if not docs:
        return "I couldn't find relevant information in your uploaded notes.", []
    context = "\n\n".join(
        f"[Source: {m.get('filename', 'PDF')}, page {m.get('page', '?')}]\n{d}"
        for d, m in zip(docs, metas)
    )
    prompt = f"""You are a study assistant. Answer the question using ONLY the context below.
If the context does not contain the answer, say: "I couldn't find that in the uploaded notes."
Keep the answer clear and student-friendly. Mention relevant page numbers.

CONTEXT:
{context}

QUESTION: {question}
"""
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    sources = [
        f"{m.get('filename', 'PDF')} — page {m.get('page', '?')}"
        for m in metas
    ]
    return response["message"]["content"], list(dict.fromkeys(sources))

with st.sidebar:
    st.header("1. Upload notes")
    uploaded_files = st.file_uploader(
        "Choose PDF files", type=["pdf"], accept_multiple_files=True
    )
    if st.button("📥 Process PDFs", disabled=not uploaded_files, use_container_width=True):
        total = 0
        for file in uploaded_files:
            with st.spinner(f"Processing {file.name}..."):
                try:
                    count, error = index_pdf(file)
                    if error:
                        st.warning(f"{file.name}: {error}")
                    else:
                        total += count
                        st.success(f"{file.name}: {count} text chunks indexed.")
                except Exception as e:
                    st.error(f"Could not process {file.name}: {e}")
        if total:
            st.success(f"Done! Indexed {total} chunks.")

    st.divider()
    st.header("2. Collection")
    try:
        count = get_collection().count()
        st.metric("Indexed text chunks", count)
    except Exception:
        st.write("Collection will be created when you process a PDF.")
    st.caption(f"Models: {CHAT_MODEL} + {EMBED_MODEL}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your uploaded notes...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching your notes and preparing an answer..."):
            try:
                answer, sources = answer_question(question)
                st.markdown(answer)
                if sources:
                    with st.expander("📌 Sources"):
                        for source in sources:
                            st.write("• " + source)
            except Exception as e:
                answer = (
                    "I couldn't connect to Ollama. Make sure Ollama is installed "
                    "and running, and that both models are downloaded.\n\n"
                    f"Technical detail: `{e}`"
                )
                st.error(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

# RAG Study Assistant (Windows)

A beginner-friendly PDF question-answering app using:
- Streamlit for the interface
- PyPDF for PDF text extraction
- ChromaDB for local vector storage
- Ollama for local embeddings and chat model

## 1. Install prerequisites
1. Install Python 3.11 or 3.12 from https://www.python.org/downloads/
   - During setup, tick **Add Python to PATH**.
2. Install Ollama for Windows from https://ollama.com/download
3. Open Command Prompt and download the models:

```bat
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

## 2. Set up the project
Open Command Prompt in this folder and run:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Run
```bat
streamlit run app.py
```
The app opens in your browser. Upload a text-based PDF, click **Process PDFs**, then ask a question.

## Notes
- First model downloads need an internet connection.
- This starter app works with text-based PDFs. Scanned/image-only PDFs need OCR, which is not included yet.
- Answers are designed to use uploaded notes only, but AI can still make mistakes. Verify important information.
- The vector database is saved locally in `chroma_db`.

# Chat Bot - with RAG and Llama
***currently in it's early stages***

Lightweight RAG (Retrieval-Augmented Generation) chatbot that answers questions based on the contents of PDF documents. It uses a local LLM powered by `llama.cpp`, and combines it with vector-based document retrieval (via FAISS) to provide accurate, context-aware responses.

This project is designed for privacy-conscious environments — no external API calls are made, and all model inference happens locally.

---

## 🧩 What This Project Does

1. **Ingest PDF files**: 
   - Parses content from one or more PDFs using `PyMuPDF` or OCR (Tesseract) for scanned files.
   - Embeds text chunks using `sentence-transformers` or HuggingFace models.
   - Stores embeddings in a local FAISS vector store.

2. **Run a local LLM**: 
   - Uses `llama-cpp-python` to run models like `llama-2-7b.Q4_K_M.gguf` locally (no internet needed).
   - You can plug in any compatible `.gguf` model.

3. **RAG pipeline**: 
   - When a user asks a question, relevant chunks are retrieved from the vector store.
   - These are passed into the LLM for generation of an answer grounded in the source content.

---

## 🛠️ How It Works

- **LangChain** is used to orchestrate document loaders, embeddings, retrievers, and LLMs.
- **HuggingFace Embeddings** provide compact representations of document chunks.
- **FAISS** enables efficient similarity search over embedded documents.
- **Llama.cpp** runs the LLM fully offline, making this a self-contained Q&A system.

---

## ⚡ Quickstart

1. **Install requirements** (preferably in a virtual environment):

   ```bash
   pip install -r requirements.txt
   ```

2. **Download a LLaMA model** (e.g., `llama-2-7b.Q4_K_M.gguf`) and place it in the `models/` directory.
   
   Use this wget cmd to download this model:
   ```
   wget -O <CWD>/models/llama-2-7b.Q4_K_M.gguf https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf

3. **Ingest your documents**:

   ```bash
   python src/ingest.py
   ```

4. **Run the chatbot**:

   ```bash
   python src/app.py
   ```



## 📌 Notes

- Make sure your system has enough RAM (ideally 8GB+) for the LLM.
- For scanned PDFs, Tesseract must be installed system-wide for OCR to work.
- Works cross platform, tuned for Arch (btw :D)

---

## ✅ To-Do


- Table understanding and structured data extraction
- UI frontend or web chat interface (streamlit/gradio)
- Model selection CLI flag

## Screenshots
![image](https://github.com/user-attachments/assets/c2a7cd59-f720-4fc3-963d-58fa1f392927)

![image](https://github.com/user-attachments/assets/8d22ad32-ae0e-45d2-a7df-5cca2dc68885)


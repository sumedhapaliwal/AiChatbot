from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from ocr_utils import load_images_with_ocr
import os

DATA_DIR = f"{os.getcwd()}/docs/"
VECTOR_DIR = f"{os.getcwd()}/vector_store/"

def load_docs():
    docs = []
    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)
        if file.endswith(".txt"):
            loader = TextLoader(path)
            docs.extend(loader.load())
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        elif file.endswith((".png", ".jpg", ".jpeg")):
            docs.extend(load_images_with_ocr(path))
    return docs

def ingest():
    docs = load_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_DIR)
    print("Ingestion complete. Vector store saved.")

    # testing if the pdf got parsed
    # db = FAISS.load_local(f"{os.getcwd()}/vector_store", embeddings, allow_dangerous_deserialization=True)
    # print(db.similarity_search("test", k=2))

if __name__ == "__main__":
    ingest()
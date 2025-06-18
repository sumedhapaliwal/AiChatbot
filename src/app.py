from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
import os

VECTOR_DIR = f"{os.getcwd()}/vector_store"

def load_chain():
    # Load embeddings + vector DB
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.load_local(
        VECTOR_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Define a prompt template
    prompt_template = """
    You are an expert assistant. Use the below context to answer the question.

    Context:
    {context}

    Question:
    {question}

    Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    llm = LlamaCpp(
        model_path="models/llama-2-7b.Q4_K_M.gguf",
        n_ctx=4096,             
        n_batch=64,          
        f16_kv=True,
        use_mlock=True, 
        verbose=True
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return chain

def main():
    chain = load_chain()

    print("🧠 RAG Chatbot Ready. Ask your question:\n")
    while True:
        query = input("> ")
        if query.lower() in ["exit", "quit"]:
            break
        result = chain.invoke({"query": query})

        print("\n💬 Answer:\n", result['result'])
        print("\n📄 Source Chunks:")
        for doc in result['source_documents']:
            print(f"📄 Page {doc.metadata.get('page_label', '?')} | {doc.page_content[:200]}...\n")

if __name__ == "__main__":
    main()
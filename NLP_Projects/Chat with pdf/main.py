import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def setup_qa_system(file_path):

    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 200)
    chunks = text_splitter.split_documents(docs)
    
    embeddings = OllamaEmbeddings(model= "llama2")
    vector_store = FAISS.from_documents(chunks , embeddings)

    retriever = vector_store.as_retriever()
    llm = Ollama(model= "llama2")
    
    qa_chain = RetrievalQA.from_chain_type(llm , retriever = retriever)

    return qa_chain

if __name__ == "__main__":

    qa_chain = setup_qa_system("Attention Is All You Need.pdf")

    while True:
        question = input("\n Ask a question: ")

        if question.lower() == "exit":
            break

        answer = qa_chain.invoke(question)
        print("\n Answer: \n", answer['result'])
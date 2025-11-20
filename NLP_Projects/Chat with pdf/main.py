!pip install -r requirements.txt

pip install pypdf

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
api_key = os.getenv("MISTRALAI_API_KEY")

os.environ["MISTRALAI_API_KEY"] = api_key

def setup_qa_system(file_path):

    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 200)
    chunks = text_splitter.split_documents(docs)

    embeddings = MistralAIEmbeddings(
                model="mistral-embed",
                mistral_api_key=api_key
                )
    vector_store = FAISS.from_documents(chunks , embeddings)

    retriever = vector_store.as_retriever()
    llm = ChatMistralAI(
            model="mistral-large-latest",
            temperature=0,
            max_retries=2,
            mistral_api_key=api_key
        )

    qa_chain = RetrievalQA.from_chain_type(llm , retriever = retriever)

    return qa_chain

if __name__ == "__main__":

    qa_chain = setup_qa_system("/content/Attention is all you need.pdf")

    while True:
        question = input("\n Ask a question: ")

        if question.lower() == "exit":
            break

        answer = qa_chain.invoke(question)
        print("\n Answer: \n", answer['result'])


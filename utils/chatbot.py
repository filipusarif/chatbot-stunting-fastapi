import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

load_dotenv()

# Konfigurasi Path
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

def get_models(api_key: str):
    """Inisialisasi LLM dan Embeddings dengan API Key dari Laravel"""
    llm = ChatNVIDIA(
        model="meta/llama3-70b-instruct", 
        nvidia_api_key=api_key, 
        temperature=0.5
    )
    embeddings = NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5", 
        nvidia_api_key=api_key
    )
    return llm, embeddings

def get_retriever(api_key: str):
    """Memuat atau membuat vector store secara dinamis"""
    _, embeddings = get_models(api_key)
    
    # 1. Jika database sudah ada, muat langsung
    if os.path.exists(DB_FAISS_PATH):
        vectorstore = FAISS.load_local(
            DB_FAISS_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True 
        )
        return vectorstore.as_retriever()
    
    # 2. Jika belum ada, buat baru (Proses indexing)
    print("Database tidak ditemukan. Melakukan indexing data...")
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        return None

    pdf_loader = DirectoryLoader(DATA_PATH, glob="./*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(DATA_PATH, glob="./*.txt", loader_cls=TextLoader)
    
    docs = pdf_loader.load() + txt_loader.load()
    if not docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    
    os.makedirs("vectorstore", exist_ok=True)
    vectorstore.save_local(DB_FAISS_PATH)
    
    return vectorstore.as_retriever()

def get_chat_response_with_rag(user_input: str, api_key: str) -> str:
    """Fungsi utama yang dipanggil oleh FastAPI"""
    try:
        # Inisialisasi model dan retriever dengan key yang dikirim
        llm, _ = get_models(api_key)
        retriever = get_retriever(api_key)

        if not retriever:
            return "Maaf, asisten belum memiliki basis data pengetahuan. Mohon hubungi admin untuk mengisi folder data/."

        system_prompt = (
            "Anda adalah asisten kesehatan ahli stunting. "
            "Gunakan konteks berikut untuk menjawab pertanyaan: {context}. "
            "Jika informasi tidak ada dalam konteks, katakan bahwa Anda tidak tahu, "
            "namun tetap berikan saran umum kesehatan anak yang relevan. "
            "Jawablah dengan bahasa Indonesia yang ramah."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # Bangun RAG Chain secara dinamis
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Eksekusi
        response = rag_chain.invoke({"input": user_input})
        return response["answer"]

    except Exception as e:
        # Tangani jika API Key salah atau kedaluwarsa
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            return "Error: API Key NVIDIA tidak valid atau sudah habis. Mohon perbarui di halaman Admin."
        return f"Terjadi kesalahan teknis: {error_msg}"
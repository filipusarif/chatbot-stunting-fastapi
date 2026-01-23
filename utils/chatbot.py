import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:

    raise ImportError("Gagal memuat LangChain chains. Pastikan paket 'langchain' terinstal sempurna.")
load_dotenv()

# Path
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Model Init
llm = ChatNVIDIA(model="meta/llama3-70b-instruct", temperature=0.5)
embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")

def create_or_load_vector_store():
    if os.path.exists(DB_FAISS_PATH):
        print("Memuat Vector Store dari lokal...")
        vectorstore = FAISS.load_local(
            DB_FAISS_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True 
        )
    else:
        print("Database tidak ditemukan. Membuat database baru dari folder data/...")
        
        pdf_loader = DirectoryLoader(DATA_PATH, glob="./*.pdf", loader_cls=PyPDFLoader)
        txt_loader = DirectoryLoader(DATA_PATH, glob="./*.txt", loader_cls=TextLoader)
        
        docs = pdf_loader.load() + txt_loader.load()
        
        if not docs:
            print("Peringatan: Tidak ada file ditemukan di folder data/")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        
        os.makedirs("vectorstore", exist_ok=True)
        vectorstore.save_local(DB_FAISS_PATH)
        print("Database berhasil dibuat dan disimpan secara permanen.")
    
    return vectorstore.as_retriever()

retriever = create_or_load_vector_store()

def get_chat_response_with_rag(user_input: str):
    if not retriever:
        return "Maaf, sistem pengetahuan saya sedang dipersiapkan. Silakan coba lagi nanti."

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

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": user_input})
    return response["answer"]
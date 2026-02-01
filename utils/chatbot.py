import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from utils.detection import detector
import json
import re

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
    
    if os.path.exists(DB_FAISS_PATH):
        vectorstore = FAISS.load_local(
            DB_FAISS_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True 
        )
        return vectorstore.as_retriever()
    
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
    try:
        llm, _ = get_models(api_key)
        retriever = get_retriever(api_key)

        extraction_prompt = f"""
        [SYSTEM: OUTPUT ONLY VALID JSON. NO PREAMBLE.]
        Tugas: Klasifikasikan pesan user dan ekstrak data jika ada.

        ATURAN IS_DETECTION:
        - SET "is_detection": true HANYA JIKA user memberikan data angka (umur/tinggi/berat) ATAU secara eksplisit meminta dilakukan pengecekan/analisis pada kasus tertentu.
        - SET "is_detection": false JIKA user bertanya secara umum (misal: "Halo", "Apa itu stunting?", "Bagaimana cara cek stunting?").
        
        Ekstrak 7 fitur medis jika ada: [Gender, Umur, BB_Lahir, TB_Lahir, BB_Skrg, TB_Skrg, ASI]

        ATURAN KETAT:
        - "data_lengkap" HANYA boleh true jika SEMUA 7 fitur memiliki angka (BUKAN null).
        - Jika ada satu saja yang tidak disebutkan user, set "data_lengkap": false dan masukkan ke "data_kurang".
        - JANGAN mengarang angka. Gunakan null jika tidak ada di pesan.

        Kebutuhan Data: [Gender(0:Pr, 1:Lk), Umur(bln), BB_Lahir(kg), TB_Lahir(cm), BB_Skrg(kg), TB_Skrg(cm), ASI(0:Tdk, 1:Ya)]

        FORMAT JSON:
        {{
          "is_detection": true,
          "data_lengkap": false,
          "data_kurang": ["field"],
          "features": [7 numbers/null]
        }}

        Pesan User: "{user_input}"
        """
        
        extraction_response = llm.invoke(extraction_prompt)
        content = extraction_response.content if hasattr(extraction_response, 'content') else str(extraction_response)
        # print("--- DEBUG RAW LLM OUTPUT ---")
        # print(content)
        # print("-----------------------------")
        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
                data = json.loads(json_str)
            else:
                data = {"is_detection": False}
        except Exception as parse_error:
            print(f"!!! JSON Parsing Error: {parse_error}")
            data = {"is_detection": False}


        is_det = str(data.get("is_detection")).lower() == 'true'
        features = data.get("features", [])
        
        has_null = features is None or (isinstance(features, list) and (None in features or len(features) < 7))
        
        is_full = (str(data.get("data_lengkap")).lower() == 'true') and not has_null

        # --- LOGIKA FOLLOW-UP ---
        if is_det and not is_full:
            missing = data.get("data_kurang", [])
            if not missing:
                missing = ["Data kelahiran (Berat/Tinggi) atau status ASI"]
            
            missing_fields = ", ".join(missing)
            return f"Saya siap membantu menganalisis status stunting si kecil. Namun, saya butuh data: **{missing_fields}**."
        
        detection_result = "User bertanya secara umum."
        if is_det and is_full:
            features = data.get("features")
            if features and len(features) == 7:
                try:
                    res = detector.predict(features)
                    detection_result = f"HASIL ANALISIS MODEL CNN: Anak terindikasi {res['status']} ({res['probability']*100}%)."
                except Exception as pred_error:
                    print(f"!!! Prediction Error: {pred_error}")
                    detection_result = "Gagal menjalankan model prediksi."

        system_prompt = (
            "Anda adalah asisten kesehatan ahli stunting yang ramah. "
            "Informasi Tambahan: {detection_info}. "
            "Gunakan konteks dokumen: {context}. "
            "Jawablah dengan bahasa Indonesia yang empati dan profesional."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # print("RAG Chain created successfully.", detection_result)
        response = rag_chain.invoke({
            "input": user_input,
            "detection_info": detection_result 
        })
        
        return response["answer"]

    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg:
            return "Koneksi ke NVIDIA AI terputus. Mohon cek API Key di dashboard admin."
        return f"Terjadi kesalahan teknis: {error_msg}"


# def get_chat_response_with_rag(user_input: str, api_key: str) -> str:
#     """Fungsi utama yang dipanggil oleh FastAPI"""
#     try:
#         # Inisialisasi model dan retriever dengan key yang dikirim
#         llm, _ = get_models(api_key)
#         retriever = get_retriever(api_key)

#         if not retriever:
#             return "Maaf, asisten belum memiliki basis data pengetahuan. Mohon hubungi admin untuk mengisi folder data/."

#         system_prompt = (
#             "Anda adalah asisten kesehatan ahli stunting. "
#             "Gunakan konteks berikut untuk menjawab pertanyaan: {context}. "
#             "Jika informasi tidak ada dalam konteks, katakan bahwa Anda tidak tahu, "
#             "namun tetap berikan saran umum kesehatan anak yang relevan. "
#             "Jawablah dengan bahasa Indonesia yang ramah."
#         )
        
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", system_prompt),
#             ("human", "{input}"),
#         ])

#         # Bangun RAG Chain secara dinamis
#         question_answer_chain = create_stuff_documents_chain(llm, prompt)
#         rag_chain = create_retrieval_chain(retriever, question_answer_chain)

#         # Eksekusi
#         response = rag_chain.invoke({"input": user_input})
#         return response["answer"]

#     except Exception as e:
#         # Tangani jika API Key salah atau kedaluwarsa
#         error_msg = str(e)
#         if "401" in error_msg or "Unauthorized" in error_msg:
#             return "Error: API Key NVIDIA tidak valid atau sudah habis. Mohon perbarui di halaman Admin."
#         return f"Terjadi kesalahan teknis: {error_msg}"
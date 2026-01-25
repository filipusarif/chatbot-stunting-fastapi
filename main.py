from fastapi import FastAPI
from pydantic import BaseModel
from utils.chatbot import get_chat_response_with_rag
from utils.detection import detector
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Stunting Care AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    token: str

class DetectionRequest(BaseModel):
    usia_bulan: int
    berat_badan: float
    tinggi_badan: float
    jenis_kelamin: str

@app.get("/")
def home():
    return {"status": "API is running"}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    answer = get_chat_response_with_rag(request.message, request.token)
    return {"reply": answer}

class DetectionRequest(BaseModel):
    gender: int           # 1 ,0
    age: int              # bulan
    birth_weight: float   # kg
    birth_length: float   # cm
    body_weight: float    # kg
    body_length: float    # cm
    breastfeeding: int    # 1 , 0

@app.post("/api/detect")
async def detect_stunting(data: DetectionRequest):
    features = [
        data.gender, 
        data.age, 
        data.birth_weight, 
        data.birth_length, 
        data.body_weight, 
        data.body_length, 
        data.breastfeeding
    ]
    
    result = detector.predict(features)
    
    if result["status"] == "Stunting":
        result["message"] = "Berdasarkan data, anak terindikasi stunting. Segera konsultasikan ke tenaga medis."
    else:
        result["message"] = "Pertumbuhan anak Anda terlihat normal. Tetap jaga pola makan dan nutrisi."
        
    return result
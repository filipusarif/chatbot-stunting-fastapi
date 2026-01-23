from fastapi import FastAPI
from pydantic import BaseModel
from utils.chatbot import get_chat_response_with_rag
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
    response = get_chat_response_with_rag(request.message)
    return {"reply": response}

@app.post("/api/detect")
async def detect_endpoint(request: DetectionRequest):
    status = "Normal"
    if request.tinggi_badan < 70 and request.usia_bulan > 12:
        status = "Indikasi Stunting"
        
    return {
        "status_stunting": status,
        "saran": "Segera konsultasikan ke puskesmas terdekat untuk pemeriksaan Z-Score."
    }
# 🩺 StuntingCare AI Backend (FastAPI + RAG)

Backend ini bertanggung jawab untuk memproses logika **Artificial Intelligence** menggunakan metode **RAG (Retrieval-Augmented Generation)**. Sistem ini menghubungkan dokumen pengetahuan lokal dengan model **NVIDIA NIM** (Llama 3) untuk memberikan jawaban akurat seputar stunting.

---

### 🚀 Fitur Utama

* **Dynamic RAG Pipeline**: Mengintegrasikan LangChain dengan NVIDIA NIM (LLM & Embeddings).
* **Vector Store Management**: Menggunakan FAISS untuk pencarian dokumen cepat secara lokal.
* **Multi-format Loader**: Mendukung pembacaan data dari file `.pdf` dan `.txt`.
* **Dynamic Token Injection**: Menerima API Token NVIDIA secara dinamis dari Laravel (Frontend) untuk keamanan dan fleksibilitas.

---

### 🛠️ Tech Stack & Requirements

| Komponen | Teknologi |
| --- | --- |
| **Language** | Python 3.10+ |
| **Framework** | FastAPI |
| **AI Framework** | LangChain |
| **LLM** | NVIDIA NIM (Meta Llama 3 70B) |
| **Vector Database** | FAISS |
| **Server** | Uvicorn |

---

### 📥 Instalasi (Local Development)

Pastikan Anda berada di lingkungan Linux (Arch Linux) Anda dan ikuti langkah-langkah berikut:

1. **Clone Repository & Masuk ke Folder BE**
```bash
git clone https://github.com/filipusarif/chatbot-stunting-fastapi.git
cd chatbot-stunting-fastapi

```


2. **Buat Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate

```


3. **Install Dependencies**
```bash
pip install -r requirements.txt

```


4. **Konfigurasi Environment Variable**
Buat file `.env` di root folder:
```env
NVIDIA_API_KEY=your_initial_key_here

```



---

### 📂 Struktur Proyek

```text
.
├── data/               # Folder sumber dokumen (PDF/TXT)
├── vectorstore/        # Folder penyimpanan database FAISS
│   └── db_faiss/       # Index FAISS yang sudah di-generate
├── main.py             # Entry point FastAPI & Routes
├── chatbot.py          # Logika RAG & LangChain
├── requirements.txt    # Daftar library Python
└── .env                # Konfigurasi API Key (Private)

```

---

### 🔌 API Endpoints

#### **POST** `/api/chat`

Endpoint utama untuk mendapatkan jawaban dari AI.

* **Request Body**:
```json
{
  "message": "Apa itu stunting?",
  "token": "nvapi-xxxxxxxxxxxx"
}

```


* **Response**:
```json
{
  "reply": "Stunting adalah gangguan pertumbuhan..."
}

```



---

### 🛠️ Cara Menjalankan

Jalankan server Uvicorn dengan mode *reload* untuk pengembangan:

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

```

---

> **Catatan :**
> Pastikan folder `data/` sudah terisi minimal satu file dokumen sebelum menjalankan chat pertama kali agar sistem bisa melakukan *indexing* otomatis ke FAISS.


import requests
import pandas as pd
from evaluate import load
from dotenv import load_dotenv
import os
load_dotenv()
# 1. Konfigurasi API
API_URL = "http://127.0.0.1:8000/api/chat"
API_TOKEN = os.getenv("NVIDIA_API_KEY")

# 2. Dataset Uji (20 Data dari Artikel Kemenkes)
test_dataset = [
    {"q": "Apa yang dimaksud dengan stunting?", "ref": "Stunting adalah masalah kurang gizi kronis yang disebabkan oleh asupan gizi yang kurang dalam waktu cukup lama akibat pemberian makanan yang tidak sesuai dengan kebutuhan gizi."},
    {"q": "Apa penyebab stunting?", "ref": "Stunting terkait dengan banyak penyebab, antara lain aktor asupan gizi ibu dan anak, status kesehatan balita, ketahanan pangan, lingkungan sosial dan kesehatan, lingkungan pemukiman, kemiskinan, dan lain-lain."},
    {"q": "Bagaimana diagnosis stunting?", "ref": "Diagnosis stunting pertama-tama dilakukan dengan melakukan tanya jawab oleh petugas kesehatan seputaran asupan makan anak, riwayat pemberian ASI, riwayat kehamilan dan persalinan, serta lingkungan tempat tinggal anak. Setelah itu akan dilakukan pemeriksaan fisik berupa mengukur panjang atau tinggi badan, berat badan, lingkar kepala dan lingkar lengan anak. Seorang anak dapat di diagnosis stunting bila tinggi badannya berada di bawah garis merah (-2 SD) berdasarkan kurva pertumbuhan WHO."},
    {"q": "Bagaimana penanganan stunting pada anak?", "ref": "Penanganan stunting dapat meliputi pengobatan penyakit penyebabnya, perbaikan nutrisi, pemberian suplemen, serta penerapan pola hidup bersih dan sehat."},
    {"q": "Bagaimana pemberian makan bayi dan anak agar tidak stunting?", "ref": "Memberikan ASI Eksklusif sampai bayi berusia 6 bulan. Melanjutkan pemberian ASI disertai Makanan Pendamping ASI (MP ASI)."},
    {"q": "Bagaimana cara mendiagnosis stunting secara fisik?", "ref": "Diagnosis dilakukan dengan mengukur panjang atau tinggi badan anak, berat badan, lingkar kepala, dan lingkar lengan."},
    {"q": "Apa indikator diagnosis stunting menurut kurva pertumbuhan WHO?", "ref": "Seorang anak didiagnosis stunting bila tinggi badannya berada di bawah garis merah atau minus 2 Standar Deviasi (-2 SD) berdasarkan kurva pertumbuhan WHO."},
    {"q": "Apa yang dimaksud dengan periode 1000 HPK?", "ref": "1.000 Hari Pertama Kehidupan (HPK) adalah periode kunci pencegahan stunting yang dimulai sejak bayi dalam kandungan hingga anak berusia 23 bulan."},
    {"q": "Jelaskan upaya pencegahan stunting dengan metode ABCDE!", "ref": "A: Aktif minum Tablet Tambah Darah, B: Bumil teratur periksa kehamilan, C: Cukupi konsumsi protein hewani, D: Datang ke Posyandu setiap bulan, E: Eksklusif ASI 6 bulan."},
    {"q": "Berapa kali minimal ibu hamil harus melakukan pemeriksaan kehamilan (ANC)?", "ref": "Ibu hamil disarankan melakukan pemeriksaan minimal 6 kali: 1 kali pada trimester pertama, 2 kali pada trimester kedua, dan 3 kali pada trimester ketiga."},
    {"q": "Apa saja yang bisa dilakukan ibu hamil untuk mencegah stunting?", "ref": "Mengonsumsi makanan tinggi protein, rutin periksa kehamilan, dan mengonsumsi tablet tambah darah minimal 90 tablet selama masa kehamilan."},
    {"q": "Bagaimana cara mencegah stunting pada periode bayi usia 0-6 bulan?", "ref": "Melakukan inisiasi menyusu dini (IMD), memberikan kolostrum, serta memberikan ASI eksklusif secara penuh selama enam bulan pertama."},
    {"q": "Apa saja syarat pemberian MP-ASI yang baik setelah bayi berusia 6 bulan?", "ref": "MP-ASI harus tepat waktu, adekuat (jumlah dan tekstur cukup), bervariasi, aman secara kebersihan, dan diberikan dengan cara yang benar."},
    {"q": "Apa saja contoh sumber protein hewani untuk MP-ASI yang diprioritaskan?", "ref": "Ikan, ayam, daging sapi, hati ayam/sapi, udang, telur, susu, dan hasil olahannya."},
    {"q": "Bagaimana variasi makanan dalam MP-ASI yang disarankan?", "ref": "Terdiri dari makanan pokok (beras, kentang, jagung), protein hewani, protein nabati (kacang-kacangan), serta buah dan sayur."},
    {"q": "Sebutkan intervensi gizi untuk anak usia 6-23 bulan (Baduta)?", "ref": "Melanjutkan ASI hingga usia 23 bulan, memberikan MP-ASI bergizi, menyediakan obat cacing, suplementasi vitamin A, zinc, dan imunisasi lengkap."},
    {"q": "Apa pengaruh pola asuh orang tua terhadap kejadian stunting?", "ref": "Pola asuh yang tidak optimal dalam pemberian makan, kurangnya perhatian dalam tumbuh kembang, serta rendahnya pengetahuan gizi dapat meningkatkan risiko stunting."},
    {"q": "Apa saja pengobatan yang dilakukan jika anak sudah menderita stunting?", "ref": "Mengobati penyakit yang mendasarinya, memberikan nutrisi tambahan kaya protein hewani dan kalori, serta memberikan suplemen vitamin dan mineral."},
    {"q": "Mengapa sanitasi dan air bersih sangat penting untuk mencegah stunting?", "ref": "Karena rendahnya akses air bersih dan sanitasi memicu infeksi penyakit yang dapat menghambat penyerapan gizi dan mengganggu pertumbuhan anak."},
    {"q": "Apa tanda utama anak balita mengalami stunting berdasarkan tinggi badannya?", "ref": "Anak ditandai dengan tinggi badan yang sangat pendek hingga melampaui defisit 2 SD di bawah median standar pertumbuhan umurnya."}
]

# 3. Load Metrics
bleu = load("bleu")
rouge = load("rouge")

results = []

print("Memulai evaluasi... Harap tunggu.")

for i, data in enumerate(test_dataset):
    # Hit API FastAPI
    try:
        response = requests.post(API_URL, json={"message": data["q"], "token": API_TOKEN})
        ai_reply = response.json().get("reply", "")
        
        # Hitung Skor per Baris
        res_bleu = bleu.compute(predictions=[ai_reply], references=[[data["ref"]]])
        res_rouge = rouge.compute(predictions=[ai_reply], references=[data["ref"]])
        
        results.append({
            "No": i+1,
            "Pertanyaan": data["q"],
            "Jawaban AI": ai_reply[:100] + "...", # Potong untuk tampilan tabel
            "BLEU": round(res_bleu['bleu'], 4),
            "ROUGE-L": round(res_rouge['rougeL'], 4)
        })
        print(f"Data {i+1} selesai.")
    except Exception as e:
        print(f"Gagal memproses data {i+1}: {e}")

# 4. Tampilkan Hasil
df = pd.DataFrame(results)
print("\n--- HASIL EVALUASI ---")
print(df[["No", "BLEU", "ROUGE-L"]])

print("\n--- RATA-RATA SKOR ---")
print(f"Rata-rata BLEU: {df['BLEU'].mean():.4f}")
print(f"Rata-rata ROUGE-L: {df['ROUGE-L'].mean():.4f}")

# Simpan ke CSV untuk lampiran skripsi
df.to_csv("hasil_evaluasi_stuntingcare.csv", index=False)
print("\nHasil lengkap telah disimpan di 'hasil_evaluasi_stuntingcare.csv'")
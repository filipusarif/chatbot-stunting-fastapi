import pandas as pd
import matplotlib.pyplot as plt

# 1. Memuat data dari CSV
try:
    df = pd.read_csv('./evaluates/hasil_evaluasi_stuntingcare1.csv')
    print("Data berhasil dimuat.")
except FileNotFoundError:
    print("Error: File 'hasil_evaluasi_stuntingcare.csv' tidak ditemukan.")
    exit()

# Set gaya visual agar lebih modern
plt.style.use('seaborn-v0_8-muted')

# --- VISUALISASI 1: BAR CHART ---
plt.figure(figsize=(14, 7))
x = df['No']
width = 0.35

plt.bar(x - width/2, df['BLEU'], width, label='BLEU', color='#3498db', alpha=0.8)
plt.bar(x + width/2, df['ROUGE-L'], width, label='ROUGE-L', color='#e74c3c', alpha=0.8)

# Tambahkan detail grafik
plt.xlabel('Nomor Sampel Pertanyaan', fontsize=12, fontweight='bold')
plt.ylabel('Skor (0.0 - 1.0)', fontsize=12, fontweight='bold')
plt.title('Perbandingan Skor BLEU dan ROUGE-L per Sampel Uji', fontsize=14, fontweight='bold', pad=20)
plt.xticks(x)
plt.ylim(0, 1.1) # Batas atas sedikit dilebihkan untuk ruang legend
plt.legend(loc='upper right', frameon=True, shadow=True)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Simpan Bar Chart
plt.tight_layout()
plt.savefig('bar_chart_evaluasi.png', dpi=300)
print("Bar chart disimpan sebagai 'bar_chart_evaluasi.png'")


# --- VISUALISASI 2: LINE CHART ---
plt.figure(figsize=(14, 7))

plt.plot(df['No'], df['BLEU'], marker='o', linestyle='-', linewidth=2, label='BLEU', color='#2980b9')
plt.plot(df['No'], df['ROUGE-L'], marker='s', linestyle='-', linewidth=2, label='ROUGE-L', color='#c0392b')

# Tambahkan detail grafik
plt.xlabel('Nomor Sampel Pertanyaan', fontsize=12, fontweight='bold')
plt.ylabel('Skor (0.0 - 1.0)', fontsize=12, fontweight='bold')
plt.title('Tren Konsistensi Performa Model (BLEU vs ROUGE-L)', fontsize=14, fontweight='bold', pad=20)
plt.xticks(df['No'])
plt.ylim(0, 1.1)
plt.legend(loc='upper right', frameon=True, shadow=True)
plt.grid(True, linestyle='--', alpha=0.5)

# Simpan Line Chart
plt.tight_layout()
plt.savefig('line_chart_evaluasi.png', dpi=300)
print("Line chart disimpan sebagai 'line_chart_evaluasi.png'")

plt.show()
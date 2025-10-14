# 🇮🇩 Dashboard Analisis Kemiskinan di Indonesia

Sebuah aplikasi web interaktif yang dibangun dengan **Streamlit** untuk memvisualisasikan dan menganalisis data terkait kemiskinan dan indikator pembangunan lainnya di seluruh provinsi di Indonesia.

## 🚀 Fitur Utama

-   **Peta Choropleth Interaktif**: Visualisasi sebaran data (seperti persentase kemiskinan) di seluruh provinsi. Peta diwarnai berdasarkan nilai metrik untuk perbandingan yang mudah.
-   **Popup Detail Provinsi**: Klik pada provinsi mana pun di peta untuk melihat rincian data statistik yang komprehensif, termasuk:
    -   Indeks Pembangunan Manusia (IPM)
    -   Persentase Penduduk Miskin
    -   Rata-rata Lama Sekolah
    -   Pengeluaran per Kapita
    -   Dan metrik sosial-ekonomi lainnya.
-   **Analisis Data Eksplorasi (EDA)**: Halaman terpisah untuk analisis mendalam, termasuk:
    -   **Heatmap Korelasi**: Untuk memahami hubungan linear antar variabel.
    -   **Scatter Plot**: Untuk melihat distribusi fitur terhadap tingkat kemiskinan.
-   **Pemuatan Data Dinamis**: Data peta (GeoJSON) dimuat langsung dari URL eksternal, membuat aplikasi lebih portabel dan ringan.
-   **Caching Cerdas**: Menggunakan `@st.cache_data` dari Streamlit untuk memastikan data hanya dimuat dan diproses sekali, menghasilkan performa aplikasi yang sangat cepat.

---

## 🛠️ Teknologi yang Digunakan

-   **Framework**: Python, Streamlit
-   **Manipulasi Data**: Pandas, NumPy
-   **Visualisasi**: Folium, Streamlit-Folium, Seaborn, Matplotlib
-   **Networking**: Requests

---

## ⚙️ Instalasi dan Penggunaan

Ikuti langkah-langkah berikut untuk menjalankan aplikasi ini secara lokal.

### 1. Prasyarat

Pastikan Anda telah menginstal Python (versi 3.9 atau lebih tinggi).

### 2. Clone Repositori

```bash
git clone https://github.com/naufalhajid/Poverty-in-Indonesia.git
cd Poverty in Indonesia
```

### 3. Buat dan Aktifkan Virtual Environment (Direkomendasikan)

# Untuk Windows
python -m venv venv
.\venv\Scripts\activate

# Untuk macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Instal Dependensi

Instal semua pustaka yang diperlukan menggunakan file `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 5. Jalankan Aplikasi

Setelah semua dependensi terinstal, jalankan aplikasi Streamlit dengan perintah berikut:

```bash
streamlit run app.py
```

Aplikasi akan terbuka secara otomatis di browser default Anda.

---

## 📂 Struktur Proyek

```
.
├── app.py                  # Kode utama aplikasi Streamlit
├── requirements.txt        # Daftar dependensi Python
└── data/
    └── df_cleaned.csv      # Dataset utama yang berisi indikator sosial-ekonomi
```

---
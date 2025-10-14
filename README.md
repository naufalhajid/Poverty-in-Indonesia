<div align="center">
  <h1>🇮🇩 Dashboard Analisis Kemiskinan di Indonesia</h1>
  <p>
    Aplikasi web interaktif untuk memvisualisasikan dan menganalisis data kemiskinan serta indikator pembangunan lainnya di seluruh provinsi di Indonesia.
  </p>
  <a href="#"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App"></a>
</div>

---

## 🎯 Tujuan Proyek

Proyek ini bertujuan untuk menyediakan platform yang mudah digunakan bagi para analis, peneliti, atau masyarakat umum untuk:
1.  **Memahami sebaran geografis** indikator sosial-ekonomi di Indonesia.
2.  **Menganalisis hubungan** antar berbagai variabel seperti tingkat kemiskinan, IPM, dan pengeluaran per kapita.
3.  Menyajikan data yang kompleks dalam format visual yang **intuitif dan interaktif**.

---

## ✨ Fitur Unggulan

🗺️ **Peta Choropleth Interaktif**
-   Visualisasi data provinsi yang diwarnai berdasarkan tingkat kemiskinan.
-   Legenda dinamis untuk interpretasi data yang mudah.
-   Efek *highlight* saat kursor diarahkan ke sebuah provinsi.

🖱️ **Tooltip & Popup Detail**
-   **Tooltip Cepat**: Arahkan kursor ke provinsi untuk melihat nama dan persentase kemiskinan secara instan.
-   **Popup Komprehensif**: Klik pada provinsi untuk menampilkan jendela detail berisi berbagai metrik penting seperti IPM, Rata-rata Lama Sekolah, Umur Harapan Hidup, dan lainnya.

📊 **Halaman Analisis Data Eksplorasi (EDA)**
-   **Heatmap Korelasi**: Membantu mengidentifikasi hubungan positif atau negatif antar variabel.
-   **Scatter Plot**: Memvisualisasikan hubungan antara tingkat kemiskinan dengan fitur-fitur lainnya.
-   **Pratinjau Data**: Tampilan tabel dari dataset yang telah dibersihkan.

⚡ **Arsitektur Aplikasi yang Efisien**
-   **Pemuatan Data dari URL**: GeoJSON dimuat secara dinamis dari repositori eksternal, membuat aplikasi lebih ringan.
-   **Caching Cerdas**: Menggunakan decorator `@st.cache_data` untuk memuat dan memproses data hanya sekali, memastikan navigasi antar halaman yang sangat cepat.

---

## 📚 Sumber Data dan Pra-pemrosesan

*   **Dataset Utama**: File `data/df_cleaned.csv` berisi kumpulan data indikator sosial-ekonomi di tingkat **Kabupaten/Kota**.
*   **Data Geografis**: File `prov 34 simplified.geojson` dimuat dari URL publik untuk menyediakan batas-batas wilayah provinsi.
*   **Pra-pemrosesan Otomatis**: Aplikasi secara cerdas melakukan langkah-langkah berikut saat dijalankan:
    1.  **Pembersihan Nama Kolom**: Mengubah nama kolom yang panjang menjadi alias yang lebih pendek dan mudah dibaca.
    2.  **Agregasi Data**: Mengagregasi data dari tingkat Kabupaten/Kota ke tingkat **Provinsi** dengan menghitung nilai rata-rata (`.mean()`). Ini adalah langkah krusial agar data dapat dipetakan ke GeoJSON provinsi.
    3.  **Injeksi Data**: Menyuntikkan data yang telah diagregasi ke dalam properti GeoJSON untuk digunakan oleh *tooltip* dan *popup*.

---

## 🛠️ Teknologi yang Digunakan

-   **Framework Aplikasi**: `Streamlit`
-   **Manipulasi Data**: `Pandas`, `NumPy`
-   **Visualisasi Peta**: `Folium`, `streamlit-folium`
-   **Visualisasi Statistik**: `Seaborn`, `Matplotlib`
-   **Permintaan Jaringan**: `Requests`

---

## ⚙️ Panduan Instalasi dan Penggunaan Lokal

#### 1. Prasyarat
Pastikan Anda memiliki **Python 3.9** atau versi yang lebih baru terinstal di sistem Anda.

#### 2. Clone Repositori
Buka terminal Anda dan jalankan perintah berikut untuk meng-clone proyek ini:
```bash
git clone https://github.com/naufalhajid/Poverty-in-Indonesia.git
```

#### 3. Masuk ke Direktori Proyek
```bash
# Gunakan tanda kutip jika nama folder mengandung spasi
cd "Poverty in Indonesia"
```

#### 4. Buat dan Aktifkan Virtual Environment (Sangat Direkomendasikan)
Ini akan mengisolasi dependensi proyek Anda dari instalasi Python global.
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

#### 5. Instal Dependensi
Instal semua pustaka yang dibutuhkan dengan satu perintah:
```bash
pip install -r requirements.txt
```

#### 6. Jalankan Aplikasi
Jalankan perintah berikut di terminal Anda:
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
# Dashboard Analisis Kemiskinan di Indonesia

Aplikasi Streamlit untuk mengeksplorasi indikator kemiskinan dan pembangunan di Indonesia melalui peta provinsi interaktif, ringkasan statistik, heatmap korelasi, dan scatter plot.

Project ini diposisikan sebagai **dashboard publik eksploratif**, bukan aplikasi prediksi machine learning. Artefak model di folder `Model/` berasal dari notebook eksperimen dan belum digunakan oleh aplikasi utama.

## Fitur

- Peta choropleth provinsi berbasis Folium.
- Tooltip dan popup detail untuk setiap provinsi.
- Ringkasan provinsi dengan tingkat kemiskinan tertinggi dan terendah.
- Halaman EDA dengan statistik deskriptif, heatmap korelasi, scatter plot, dan pratinjau data.
- Validasi data dasar saat aplikasi dijalankan.
- GeoJSON lokal sebagai sumber utama agar batas provinsi tidak bergantung penuh pada URL eksternal.

## Data

- Dataset utama: `data/df_cleaned.csv`
- GeoJSON provinsi: `data/prov 34.geojson`
- Dataset mentah: `data/Klasifikasi Tingkat Kemiskinan di Indonesia.csv`

Dataset bersih berisi data kabupaten/kota dan diagregasi ke tingkat provinsi dengan rata-rata sederhana. Karena itu, angka provinsi di dashboard sebaiknya dibaca sebagai rata-rata kabupaten/kota dalam dataset, bukan estimasi berbobot populasi.

## Struktur Project

```text
.
├── app.py
├── requirements.txt
├── data/
│   ├── df_cleaned.csv
│   ├── Klasifikasi Tingkat Kemiskinan di Indonesia.csv
│   └── prov 34.geojson
├── Model/
│   ├── scaler.pkl
│   └── xgb_poverty_model.pkl
├── Notebook/
│   └── Poverty_in_Indonesia.ipynb
└── tests/
    └── test_data_contract.py
```

## Menjalankan Lokal

Pastikan Python 3.9 atau versi lebih baru sudah terpasang.

```bash
git clone https://github.com/naufalhajid/Poverty-in-Indonesia.git
cd Poverty-in-Indonesia
python -m venv venv
```

Aktifkan virtual environment:

```bash
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

Instal dependency dan jalankan aplikasi:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Verifikasi

Jalankan pengecekan sintaks dan kontrak data:

```bash
python -m py_compile app.py
python -m unittest discover -s tests
```

## Catatan Deployment

- Pastikan folder `data/` ikut terdeploy.
- Aplikasi memakai `data/prov 34.geojson` sebagai sumber batas provinsi utama.
- Basemap Folium tetap memakai tile eksternal, sehingga koneksi internet masih dibutuhkan agar layer peta dasar tampil lengkap.
- `requirements.txt` dibuat untuk runtime dashboard. Dependency machine learning berat seperti `xgboost` tidak diperlukan selama fitur prediksi belum diaktifkan.

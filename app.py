# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import os

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Prediksi Kemiskinan di Indonesia",
    page_icon="🇮🇩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- KONSTANTA PATH ---
DATA_PATH = 'data/df_cleaned.csv'
GEOJSON_PATH = 'data/prov 34.geojson' # Sesuaikan dengan nama file GeoJSON Anda
MODEL_PATH = 'model/xgb_poverty_model.pkl'
SCALER_PATH = 'model/scaler.pkl'
GEOJSON_PROVINCE_KEY = 'feature.properties.name' # Sesuaikan dengan properti di file GeoJSON Anda


@st.cache_data
def preprocess_data(df):
    """
    Membersihkan, memproses, dan mengagregasi DataFrame.
    """
    # Peta untuk mengganti nama kolom yang panjang
    column_mapping = {
        'Provinsi': 'Provinsi',
        'Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)': 'Persentase Kemiskinan (P0)',
        'Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)': 'Pengeluaran Per Kapita',
        'Rata-rata Lama Sekolah Penduduk 15+ (Tahun)': 'Rata-Rata Lama Sekolah',
        'Indeks Pembangunan Manusia': 'APM SMP',  # Menggunakan IPM sebagai proksi untuk APM SMP karena keduanya mengukur pembangunan manusia
        # 'Kepadatan Penduduk' tidak memiliki proksi langsung di CSV, akan diabaikan jika tidak ada.
    }

    # Filter kolom yang ada di DataFrame untuk menghindari KeyError
    valid_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
    df_processed = df[list(valid_columns.keys())].copy()
    df_processed.rename(columns=valid_columns, inplace=True)

    # Konversi Pengeluaran Per Kapita dari tahunan ke bulanan (dibagi 12)
    if 'Pengeluaran Per Kapita' in df_processed.columns:
        df_processed['Pengeluaran Per Kapita'] = df_processed['Pengeluaran Per Kapita'] / 12

    # Agregasi data ke tingkat provinsi untuk peta
    df_provinsi = df_processed.groupby('Provinsi').mean(numeric_only=True).reset_index()

    return df_processed, df_provinsi


# --- FUNGSI UNTUK MEMUAT DATA GEOJSON ---
@st.cache_data
def load_geojson(geojson_path):
    """
    Memuat data GeoJSON dari file.
    """
    try:
        with open(geojson_path) as f:
            return f.read()
    except FileNotFoundError:
        return None

# --- FUNGSI UNTUK MEMUAT DATA DAN MODEL ---

@st.cache_data
def load_data(data_path):
    """
    Memuat dan memproses data kemiskinan dari file CSV.
    Menggunakan cache untuk performa.
    """
    try:
        df = pd.read_csv(data_path)
        # Lakukan pra-pemrosesan setelah memuat
        df_processed, df_provinsi = preprocess_data(df)
        return df_processed, df_provinsi
    except FileNotFoundError:
        return None, None

@st.cache_resource
def load_model_and_scaler(model_path, scaler_path):
    """
    Memuat model prediksi dan scaler.
    Menggunakan cache resource karena model adalah objek berat.
    """
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError:
        return None, None

# --- HALAMAN ANALISIS DATA EKSPLORASI (EDA) ---

def run_eda_page():
    """
    Menjalankan halaman untuk Analisis Data Eksplorasi.
    """
    st.header("📊 Analisis Data Eksplorasi Kemiskinan")
    st.write("Halaman ini menampilkan analisis dari data yang digunakan untuk melatih model.")

    # Memuat data riil
    df_processed, _ = load_data(DATA_PATH)

    if df_processed is not None:
        st.subheader("Pratinjau Data")
        st.dataframe(df_processed.head())

        st.subheader("Heatmap Korelasi Antar Variabel")
        st.write("Heatmap ini menunjukkan bagaimana variabel-variabel saling berhubungan. Nilai mendekati 1 atau -1 menunjukkan korelasi yang kuat.")
        
        # Membuat plot korelasi
        fig, ax = plt.subplots(figsize=(12, 8))
        # Memilih hanya kolom numerik untuk menghindari ValueError
        numeric_df = df_processed.select_dtypes(include=np.number)
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

        st.subheader("Distribusi Fitur Terhadap Persentase Kemiskinan (P0)")
        st.write("Scatter plot di bawah ini memvisualisasikan hubungan antara setiap fitur input dengan persentase kemiskinan.")
        
        # Kolom fitur untuk visualisasi
        feature_cols = ['Pengeluaran Per Kapita', 'Rata-Rata Lama Sekolah', 'APM SMP', 'Kepadatan Penduduk']
        cols = st.columns(2)

        # Loop untuk membuat scatter plot secara dinamis
        for i, feature in enumerate(feature_cols):
            with cols[i % 2]:
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.scatterplot(data=df_processed, x=feature, y='Persentase Kemiskinan (P0)', ax=ax)
                ax.set_title(f'P0 vs {feature}')
                st.pyplot(fig)
    else:
        st.error(f"❌ **File tidak ditemukan:** Pastikan file `{os.path.basename(DATA_PATH)}` berada di dalam folder `data/`.")

# --- HALAMAN PREDIKSI KEMISKINAN ---

def run_prediction_page():
    """
    Menjalankan halaman untuk prediksi kemiskinan.
    """
    st.header("🔮 Prediksi Persentase Kemiskinan")
    st.write("Masukkan nilai-nilai fitur di bawah ini untuk memprediksi persentase kemiskinan (P0) di suatu wilayah.")

    
    # Memuat model dan scaler
    model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)

    if model is not None and scaler is not None:
        # Membuat form input di sidebar
        with st.sidebar:
            st.header("Input Fitur Prediksi")
            # Slider untuk input pengguna
            # Rentang nilai disesuaikan dari analisis data (misal: min dan max dari data training)
            pengeluaran = st.slider('Pengeluaran Per Kapita (Ribu Rupiah/Bulan)', 800.0, 2500.0, 1200.0, 50.0)
            lama_sekolah = st.slider('Rata-Rata Lama Sekolah (Tahun)', 5.0, 12.0, 8.5, 0.1)
            apm_smp = st.slider('APM SMP (%)', 60.0, 100.0, 90.0, 0.5)
            kepadatan_penduduk = st.slider('Kepadatan Penduduk (Jiwa/km²)', 10.0, 1000.0, 150.0, 10.0)

        # Tombol untuk melakukan prediksi
        if st.button('🚀 Prediksi Persentase Kemiskinan'):
            # Membuat DataFrame dari input
            input_data = pd.DataFrame({
                'Pengeluaran Per Kapita': [pengeluaran],
                'Rata-Rata Lama Sekolah': [lama_sekolah],
                'APM SMP': [apm_smp],
                'Kepadatan Penduduk': [kepadatan_penduduk]
            })

            # Melakukan penskalaan pada data input
            scaled_input = scaler.transform(input_data)

            # Melakukan prediksi
            prediction = model.predict(scaled_input)
            predicted_p0 = prediction[0]

            # Menampilkan hasil prediksi
            st.success(f"✅ **Prediksi Berhasil Dibuat!**")
            st.metric(
                label="Prediksi Persentase Kemiskinan (P0)",
                value=f"{predicted_p0:.2f}%"
            )
            st.info("**Catatan:** Prediksi ini didasarkan pada model XGBoost yang dilatih pada data historis. Hasil ini adalah estimasi dan bukan angka absolut.")

    else:
        st.error(f"❌ **File model/scaler tidak ditemukan:** Pastikan file `{os.path.basename(MODEL_PATH)}` dan `{os.path.basename(SCALER_PATH)}` berada di dalam folder `model/`.")

# --- HALAMAN VISUALISASI PETA ---

def run_map_page():
    """
    Menjalankan halaman untuk visualisasi peta interaktif.
    """
    st.header("🗺️ Peta Sebaran Kemiskinan di Indonesia")
    st.write("Peta ini memvisualisasikan persentase kemiskinan (P0) di setiap provinsi. Klik pada sebuah provinsi untuk melihat detailnya.")

    # Memuat data
    _, df_provinsi = load_data(DATA_PATH)
    geojson_data = load_geojson(GEOJSON_PATH)

    if df_provinsi is not None and geojson_data is not None:
        # Validasi: Pastikan kolom yang diperlukan untuk peta ada di DataFrame
        required_map_cols = ['Provinsi', 'Persentase Kemiskinan (P0)']
        if not all(col in df_provinsi.columns for col in required_map_cols):
            st.error(f"❌ **Data Tidak Lengkap:** DataFrame tidak memiliki kolom yang diperlukan untuk membuat peta: `{', '.join(required_map_cols)}`.")
            st.info("Mohon periksa kembali file `df_cleaned.csv` dan pastikan nama kolomnya sesuai dengan yang diharapkan dalam fungsi `preprocess_data`.")
            return

        # Membuat peta folium
        # Koordinat tengah Indonesia
        map_center = [-2.548926, 118.0148634]
        m = folium.Map(location=map_center, zoom_start=5)

        # Membuat peta Choropleth
        choropleth = folium.Choropleth(
            geo_data=GEOJSON_PATH,
            name='choropleth',
            data=df_provinsi,
            columns=['Provinsi', 'Persentase Kemiskinan (P0)'], # Gunakan data provinsi yang sudah diagregasi
            key_on=GEOJSON_PROVINCE_KEY,
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Persentase Kemiskinan (%)'
        ).add_to(m)

        # Menambahkan popup dengan informasi detail
        # Pastikan kolom 'Provinsi' di df cocok dengan 'feature.properties.state' di GeoJSON
        df_provinsi_indexed = df_provinsi.set_index('Provinsi')
        for feature in choropleth.geojson.data['features']:
            province_name = feature['properties'].get(GEOJSON_PROVINCE_KEY.split('.')[-1])
            if province_name in df_provinsi_indexed.index:
                province_data = df_provinsi_indexed.loc[province_name]
                popup_content = f"""
                <b>Provinsi: {province_name}</b><br>
                Persentase Kemiskinan: {province_data['Persentase Kemiskinan (P0)']:.2f}%<br>
                Pengeluaran Per Kapita: Rp {province_data['Pengeluaran Per Kapita']:,.0f}<br>
                Rata-Rata Lama Sekolah: {province_data['Rata-Rata Lama Sekolah']:.1f} tahun<br>
                APM SMP: {province_data['APM SMP']:.1f}%<br>
                Kepadatan Penduduk: {province_data['Kepadatan Penduduk']:.1f} jiwa/km²
                """
                folium.GeoJson(
                    feature,
                    popup=folium.Popup(popup_content)
                ).add_to(m)

        # Menampilkan peta di Streamlit
        st_folium(m, width=None, height=500, use_container_width=True)
    else:
        st.error(f"❌ **File tidak ditemukan:** Pastikan `{os.path.basename(DATA_PATH)}` dan `{os.path.basename(GEOJSON_PATH)}` ada di dalam folder `data/`.")

# --- FUNGSI UTAMA (MAIN) ---

def main():
    """
    Fungsi utama untuk menjalankan aplikasi Streamlit.
    """
    st.title("🇮🇩 Dashboard Analisis dan Prediksi Kemiskinan di Indonesia")

    # Navigasi sidebar
    st.sidebar.title("Navigasi")
    page_options = ["Prediksi Kemiskinan", "Visualisasi Peta", "Analisis Data Eksplorasi"]
    selected_page = st.sidebar.selectbox("Pilih Halaman", page_options)

    # Menjalankan halaman yang dipilih
    if selected_page == "Prediksi Kemiskinan":
        run_prediction_page()
    elif selected_page == "Visualisasi Peta":
        run_map_page()
    elif selected_page == "Analisis Data Eksplorasi":
        run_eda_page()

if __name__ == "__main__":
    main()

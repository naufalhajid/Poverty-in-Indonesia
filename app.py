# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import folium
import altair as alt
import requests
from streamlit_folium import st_folium
import os

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Analisis Kemiskinan di Indonesia",
    page_icon="🇮🇩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- KONSTANTA PATH ---
DATA_PATH = 'data/df_cleaned.csv'
GEOJSON_URL = 'https://raw.githubusercontent.com/JfrAziz/indonesia-district/refs/heads/master/prov%2034%20simplified.geojson'
GEOJSON_PROVINCE_KEY = 'feature.properties.name' # Kunci pada GeoJSON untuk mencocokkan nama provinsi



@st.cache_data
def preprocess_data(df):
    """
    Membersihkan, memproses, dan mengagregasi DataFrame.
    """
    # Peta untuk mengganti nama kolom yang panjang
    column_mapping = {
        'Provinsi': 'Provinsi',
        'Indeks Pembangunan Manusia': 'Indeks Pembangunan Manusia',
        'Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)': 'Persentase Kemiskinan (P0)',
        'Rata-rata Lama Sekolah Penduduk 15+ (Tahun)': 'Rata-Rata Lama Sekolah',
        'Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)': 'Pengeluaran Per Kapita',
        'Umur Harapan Hidup (Tahun)': 'Umur Harapan Hidup',
        'Persentase rumah tangga yang memiliki akses terhadap sanitasi layak': 'Akses Sanitasi Layak',
        'Persentase rumah tangga yang memiliki akses terhadap air minum layak': 'Akses Air Minum Layak',
        'Tingkat Pengangguran Terbuka': 'Tingkat Pengangguran Terbuka',
        'Tingkat Partisipasi Angkatan Kerja': 'Tingkat Partisipasi Angkatan Kerja',
        'PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)': 'PDRB',
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
def load_geojson(url):
    """
    Memuat data GeoJSON dari URL.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Akan raise error jika status code bukan 200
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Gagal memuat data GeoJSON dari URL: {e}")
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

# --- HALAMAN ANALISIS DATA EKSPLORASI (EDA) ---

def run_eda_page():
    """
    Menjalankan halaman untuk Analisis Data Eksplorasi.
    """
    st.header("📊 Analisis Data Eksplorasi Kemiskinan")
    st.write("Halaman ini menampilkan analisis dari data yang digunakan untuk melatih model.")
    st.header("📊 Analisis Data Eksplorasi")
    st.write("Jelajahi hubungan dan distribusi data yang digunakan dalam analisis ini.")

    # Memuat data riil
    df_processed, _ = load_data(DATA_PATH)

    if df_processed is not None:
        st.subheader("Pratinjau Data")
        st.dataframe(df_processed.head())
        # Menggunakan tab untuk layout yang lebih bersih
        tab1, tab2, tab3 = st.tabs(["📈 Ringkasan Statistik", "🔗 Hubungan Antar Variabel", "⚖️ Distribusi Fitur"])

        st.subheader("Heatmap Korelasi Antar Variabel")
        st.write("Heatmap ini menunjukkan bagaimana variabel-variabel saling berhubungan. Nilai mendekati 1 atau -1 menunjukkan korelasi yang kuat.")
        
        # Membuat plot korelasi
        fig, ax = plt.subplots(figsize=(12, 8))
        # Memilih hanya kolom numerik untuk menghindari ValueError
        numeric_df = df_processed.select_dtypes(include=np.number)
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)
        with tab1:
            st.subheader("Metrik Utama")
            # Menghitung metrik utama
            avg_poverty = df_processed['Persentase Kemiskinan (P0)'].mean()
            max_poverty_row = df_processed.loc[df_processed['Persentase Kemiskinan (P0)'].idxmax()]
            min_poverty_row = df_processed.loc[df_processed['Persentase Kemiskinan (P0)'].idxmin()]

        st.subheader("Distribusi Fitur Terhadap Persentase Kemiskinan (P0)")
        st.write("Scatter plot di bawah ini memvisualisasikan hubungan antara setiap fitur input dengan persentase kemiskinan.")
        
        # Kolom fitur untuk visualisasi
        all_feature_cols = ['Pengeluaran Per Kapita', 'Rata-Rata Lama Sekolah', 'Indeks Pembangunan Manusia']
        # Filter fitur yang benar-benar ada di DataFrame untuk menghindari error
        available_feature_cols = [col for col in all_feature_cols if col in df_processed.columns]
            col1, col2, col3 = st.columns(3)
            col1.metric("Rata-rata Kemiskinan", f"{avg_poverty:.2f}%")
            col2.metric("Kemiskinan Tertinggi", f"{max_poverty_row['Persentase Kemiskinan (P0)']:.2f}%", f"{max_poverty_row['Kab/Kota']}")
            col3.metric("Kemiskinan Terendah", f"{min_poverty_row['Persentase Kemiskinan (P0)']:.2f}%", f"{min_poverty_row['Kab/Kota']}")

        cols = st.columns(2)
            st.subheader("Pratinjau Data")
            st.dataframe(df_processed.head())

        # Loop untuk membuat scatter plot secara dinamis
        for i, feature in enumerate(available_feature_cols):
            with cols[i % 2]:
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.scatterplot(data=df_processed, x=feature, y='Persentase Kemiskinan (P0)', ax=ax, alpha=0.6)
                ax.set_title(f'P0 vs {feature}')
                st.pyplot(fig)
            with st.expander("Lihat Ringkasan Statistik Lengkap"):
                st.dataframe(df_processed.describe())

        with tab2:
            st.subheader("Scatter Plot Interaktif")
            st.write("Pilih variabel untuk sumbu X dan Y untuk melihat hubungannya.")
            
            numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("Pilih Variabel Sumbu X", options=numeric_cols, index=numeric_cols.index('Pengeluaran Per Kapita'))
            with col2:
                y_axis = st.selectbox("Pilih Variabel Sumbu Y", options=numeric_cols, index=numeric_cols.index('Persentase Kemiskinan (P0)'))

            scatter_plot = alt.Chart(df_processed).mark_circle(size=60, opacity=0.7).encode(
                x=alt.X(x_axis, scale=alt.Scale(zero=False)),
                y=alt.Y(y_axis, scale=alt.Scale(zero=False)),
                tooltip=['Provinsi', 'Kab/Kota', x_axis, y_axis]
            ).interactive()

            st.altair_chart(scatter_plot, use_container_width=True)

            st.subheader("Heatmap Korelasi")
            st.write("Heatmap ini menunjukkan korelasi linear antar variabel numerik.")
            fig, ax = plt.subplots(figsize=(14, 10))
            numeric_df = df_processed.select_dtypes(include=np.number)
            corr_matrix = numeric_df.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, annot_kws={"size": 8})
            st.pyplot(fig)

        with tab3:
            st.subheader("Histogram Distribusi Fitur")
            st.write("Pilih fitur untuk melihat distribusinya.")
            
            hist_feature = st.selectbox("Pilih Fitur", options=numeric_cols, index=numeric_cols.index('Persentase Kemiskinan (P0)'))
            
            histogram = alt.Chart(df_processed).mark_bar().encode(
                alt.X(hist_feature, bin=alt.Bin(maxbins=30), title=hist_feature),
                alt.Y('count()', title='Jumlah Kabupaten/Kota'),
                tooltip=[alt.Tooltip(hist_feature, bin=True), 'count()']
            ).properties(
                title=f'Distribusi {hist_feature}'
            )
            
            st.altair_chart(histogram, use_container_width=True)

    else:
        st.error(f"❌ **File tidak ditemukan:** Pastikan file `{os.path.basename(DATA_PATH)}` berada di dalam folder `data/`.")

# --- HALAMAN VISUALISASI PETA ---

def run_map_page():
    """
    Menjalankan halaman untuk visualisasi peta interaktif.
    """
    st.header("🗺️ Peta Sebaran Kemiskinan di Indonesia")
    st.write("Peta ini memvisualisasikan persentase kemiskinan (P0) di setiap provinsi. Klik pada sebuah provinsi untuk melihat detailnya.")

    # Memuat data
    df_processed, df_provinsi = load_data(DATA_PATH)
    geojson_data = load_geojson(GEOJSON_URL)

    if df_provinsi is not None and geojson_data is not None and df_processed is not None:
        # Validasi: Pastikan kolom yang diperlukan untuk peta ada di DataFrame
        required_map_cols = ['Provinsi', 'Persentase Kemiskinan (P0)']
        if not all(col in df_provinsi.columns for col in required_map_cols):
            st.error(f"❌ **Data Tidak Lengkap:** DataFrame tidak memiliki kolom yang diperlukan untuk membuat peta: `{', '.join(required_map_cols)}`.")
            st.info("Mohon periksa kembali file `df_cleaned.csv` dan pastikan nama kolomnya sesuai dengan yang diharapkan dalam fungsi `preprocess_data`.")
            return

        # --- 4. Tambahkan Data Relevan ke GeoJSON untuk Tooltip dan Popup ---
        df_provinsi_indexed = df_provinsi.set_index('Provinsi')
        for feature in geojson_data['features']:
            # Normalisasi nama provinsi dari GeoJSON
            province_name = feature['properties'].get('name', '').upper().strip()
            
            if province_name in df_provinsi_indexed.index:
                province_data = df_provinsi_indexed.loc[province_name]
                # Menyuntikkan data yang sudah diformat ke properti GeoJSON
                feature['properties']['IPM_DISPLAY'] = f"{province_data.get('Indeks Pembangunan Manusia', 0):.2f}"
                feature['properties']['PENDUDUK_MISKIN'] = f"{province_data.get('Persentase Kemiskinan (P0)', 0):.2f}%"
                feature['properties']['LAMA_SEKOLAH'] = f"{province_data.get('Rata-Rata Lama Sekolah', 0):.2f} Tahun"
                feature['properties']['PENGELUARAN_KAPITA'] = f"{province_data.get('Pengeluaran Per Kapita', 0):,.0f} Ribu Rupiah"
                feature['properties']['UMUR_HARAPAN_HIDUP'] = f"{province_data.get('Umur Harapan Hidup', 0):.2f} Tahun"
                feature['properties']['SANITASI_LAYAK'] = f"{province_data.get('Akses Sanitasi Layak', 0):.2f}%"
                feature['properties']['AIR_MINUM_LAYAK'] = f"{province_data.get('Akses Air Minum Layak', 0):.2f}%"
                feature['properties']['PENGANGGURAN'] = f"{province_data.get('Tingkat Pengangguran Terbuka', 0):.2f}%"
                feature['properties']['ANGKATAN_KERJA'] = f"{province_data.get('Tingkat Partisipasi Angkatan Kerja', 0):.2f}%"
                feature['properties']['PDRB'] = f"{province_data.get('PDRB', 0):,.0f} Rupiah"
            else:
                # Menangani jika provinsi dari GeoJSON tidak ada di data CSV
                for prop in ['IPM_DISPLAY', 'PENDUDUK_MISKIN', 'LAMA_SEKOLAH', 'PENGELUARAN_KAPITA', 'UMUR_HARAPAN_HIDUP', 'SANITASI_LAYAK', 'AIR_MINUM_LAYAK', 'PENGANGGURAN', 'ANGKATAN_KERJA', 'PDRB']:
                    feature['properties'][prop] = "Data Tidak Tersedia"

        # Membuat peta folium
        # Koordinat tengah Indonesia
        m = folium.Map(location=[-2.5, 118.0], zoom_start=5, tiles='OpenStreetMap')

        # Membuat peta Choropleth
        folium.Choropleth(
            geo_data=geojson_data,
            name='Poverty Choropleth',
            data=df_provinsi,
            columns=['Provinsi', 'Persentase Kemiskinan (P0)'],
            key_on=GEOJSON_PROVINCE_KEY,
            fill_color='YlOrRd',
            fill_opacity=0.8,
            line_opacity=0.4,
            line_color='black',
            line_weight=0.5,
            legend_name='Persentase Penduduk Miskin (%)',
            highlight=True
        ).add_to(m)

        # --- 5. Buat Objek Tooltip dan Popup ---
        tooltip = folium.features.GeoJsonTooltip(
            fields=['name', 'PENDUDUK_MISKIN'],
            aliases=['Provinsi:', 'Penduduk Miskin:'],
            localize=True,
            sticky=False,
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
        )

        popup = folium.features.GeoJsonPopup(
            fields=[
                'name',
                'IPM_DISPLAY',
                'PENDUDUK_MISKIN',
                'LAMA_SEKOLAH',
                'PENGELUARAN_KAPITA',
                'UMUR_HARAPAN_HIDUP',
                'SANITASI_LAYAK',
                'AIR_MINUM_LAYAK',
                'PENGANGGURAN',
                'ANGKATAN_KERJA',
                'PDRB'
                ],
            aliases=[
                'Provinsi:',
                'Indeks Pembangunan Manusia:',
                'Persentase Penduduk Miskin:',
                'Rata-rata Lama Sekolah:',
                'Pengeluaran per Kapita Disesuaikan:',
                'Umur Harapan Hidup:',
                'Akses Sanitasi Layak:',
                'Akses Air Minum Layak:',
                'Tingkat Pengangguran Terbuka:',
                'Tingkat Partisipasi Angkatan Kerja:',
                'PDRB atas Dasar Harga Konstan:'
                ],
            localize=True,
            sticky=False,
            labels=True,
            max_width=400
        )
        
        # --- 6. Buat Objek GeoJson dengan Tooltip dan Popup ---
        geojson_layer_interactive = folium.features.GeoJson(
            geojson_data,
            name='Detail Provinsi',
            style_function=lambda x: {'fillColor': 'transparent', 'color': 'transparent', 'weight': 0},
            control=False,
            tooltip=tooltip, # Pass the tooltip object
            popup=popup      # Pass the popup object
        )
        geojson_layer_interactive.add_to(m)

        folium.LayerControl().add_to(m)

        # Menampilkan peta di Streamlit
        st_folium(m, width=None, height=500, use_container_width=True)
    else:
        st.error(f"❌ **File tidak ditemukan:** Pastikan file `{os.path.basename(DATA_PATH)}` ada di dalam folder `data/` atau URL GeoJSON valid.")

# --- FUNGSI UTAMA (MAIN) ---

def main():
    """
    Fungsi utama untuk menjalankan aplikasi Streamlit.
    """
    st.title("🇮🇩 Dashboard Analisis Kemiskinan di Indonesia")

    # Navigasi sidebar
    st.sidebar.title("Navigasi")
    page_options = ["Visualisasi Peta", "Analisis Data Eksplorasi"]
    selected_page = st.sidebar.selectbox("Pilih Halaman", page_options)

    # Menjalankan halaman yang dipilih
    if selected_page == "Visualisasi Peta":
        run_map_page()
    elif selected_page == "Analisis Data Eksplorasi":
        run_eda_page()

if __name__ == "__main__":
    main()

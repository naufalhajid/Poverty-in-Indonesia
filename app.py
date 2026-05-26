from pathlib import Path
import copy
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "poverty_dashboard_matplotlib"))

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import folium
import requests
from streamlit_folium import st_folium

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Analisis Kemiskinan di Indonesia",
    page_icon="🇮🇩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- KONSTANTA PATH ---
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'data' / 'df_cleaned.csv'
LOCAL_GEOJSON_PATH = BASE_DIR / 'data' / 'prov 34.geojson'
GEOJSON_URL = 'https://raw.githubusercontent.com/JfrAziz/indonesia-district/refs/heads/master/prov%2034%20simplified.geojson'
GEOJSON_PROVINCE_KEY = 'feature.properties.name'

COLUMN_MAPPING = {
    'Provinsi': 'Provinsi',
    'Kab/Kota': 'Kab/Kota',
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

REQUIRED_COLUMNS = [
    'Provinsi',
    'Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)',
]

NUMERIC_COLUMNS = [
    'Indeks Pembangunan Manusia',
    'Persentase Kemiskinan (P0)',
    'Rata-Rata Lama Sekolah',
    'Pengeluaran Per Kapita',
    'Umur Harapan Hidup',
    'Akses Sanitasi Layak',
    'Akses Air Minum Layak',
    'Tingkat Pengangguran Terbuka',
    'Tingkat Partisipasi Angkatan Kerja',
    'PDRB',
]

# --- CUSTOM CSS ---
def load_css():
    """Load custom CSS for better UI"""
    st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 10px;
    }
    h2 {
        color: #2c3e50;
        padding-top: 20px;
    }
    .stAlert {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def preprocess_data(df):
    """
    Membersihkan, memproses, dan mengagregasi DataFrame.
    """
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Kolom wajib tidak ditemukan: {', '.join(missing_columns)}")

    # Filter kolom yang ada di DataFrame
    valid_columns = {k: v for k, v in COLUMN_MAPPING.items() if k in df.columns}
    df_processed = df[list(valid_columns.keys())].copy()
    df_processed.rename(columns=valid_columns, inplace=True)

    for column in NUMERIC_COLUMNS:
        if column in df_processed.columns:
            df_processed[column] = pd.to_numeric(
                df_processed[column].astype(str).str.strip().str.replace(',', '.', regex=False),
                errors='coerce'
            )

    # Konversi Pengeluaran Per Kapita dari tahunan ke bulanan
    if 'Pengeluaran Per Kapita' in df_processed.columns:
        df_processed['Pengeluaran Per Kapita'] = df_processed['Pengeluaran Per Kapita'] / 12

    # Normalisasi nama provinsi
    if 'Provinsi' in df_processed.columns:
        df_processed['Provinsi'] = df_processed['Provinsi'].astype(str).str.upper().str.strip()

    invalid_provinces = df_processed['Provinsi'].isin(['', 'NAN', 'NONE'])
    if df_processed['Provinsi'].isna().any() or invalid_provinces.any():
        raise ValueError("Kolom Provinsi memiliki nilai kosong.")

    invalid_numeric_columns = [
        col for col in NUMERIC_COLUMNS
        if col in df_processed.columns and df_processed[col].isna().any()
    ]
    if invalid_numeric_columns:
        raise ValueError(f"Kolom numerik memiliki nilai tidak valid: {', '.join(invalid_numeric_columns)}")

    # Agregasi data ke tingkat provinsi dengan rata-rata sederhana kabupaten/kota.
    df_provinsi = df_processed.groupby('Provinsi').mean(numeric_only=True).reset_index()

    return df_processed, df_provinsi

@st.cache_data
def load_geojson(local_path, fallback_url=None):
    """
    Memuat data GeoJSON lokal terlebih dahulu, lalu URL cadangan jika perlu.
    """
    local_path = Path(local_path)
    if local_path.exists():
        try:
            with local_path.open(encoding='utf-8') as file:
                geojson_data = json.load(file)
            validate_geojson(geojson_data)
            return geojson_data
        except (OSError, json.JSONDecodeError, ValueError) as e:
            st.error(f"❌ **Gagal memuat GeoJSON lokal:** {str(e)}")
            if fallback_url is None:
                return None

    if fallback_url is None:
        st.error(f"❌ **File GeoJSON tidak ditemukan:** `{local_path}`")
        return None

    try:
        response = requests.get(fallback_url, timeout=10)
        response.raise_for_status()
        geojson_data = response.json()
        validate_geojson(geojson_data)
        return geojson_data
    except requests.exceptions.Timeout:
        st.error("⏱️ **Timeout:** Koneksi ke server GeoJSON terlalu lama. Coba refresh halaman.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ **Gagal memuat GeoJSON:** {str(e)}")
        return None
    except ValueError as e:
        st.error(f"❌ **Format GeoJSON tidak valid:** {str(e)}")
        return None

def validate_geojson(geojson_data):
    """Validate the minimal GeoJSON contract needed by the dashboard."""
    if not isinstance(geojson_data, dict):
        raise ValueError("GeoJSON harus berupa object JSON.")

    if geojson_data.get('type') != 'FeatureCollection':
        raise ValueError("GeoJSON harus bertipe FeatureCollection.")

    features = geojson_data.get('features')
    if not isinstance(features, list) or not features:
        raise ValueError("GeoJSON tidak memiliki daftar features.")

    missing_name = [
        idx for idx, feature in enumerate(features, start=1)
        if not isinstance(feature, dict) or not feature.get('properties', {}).get('name')
    ]
    if missing_name:
        raise ValueError(f"Properti name kosong pada feature: {missing_name[:5]}")

@st.cache_data
def load_data(data_path):
    """
    Memuat dan memproses data kemiskinan dari file CSV.
    """
    try:
        data_path = Path(data_path)
        if not data_path.exists():
            st.error(f"❌ **File tidak ditemukan:** `{data_path}`")
            st.info("💡 **Tips untuk deployment:**\n"
                   "- Pastikan folder `data/` dan file `df_cleaned.csv` ada di repository\n"
                   "- Upload file melalui GitHub repository Anda\n"
                   "- Periksa struktur folder: `your_repo/data/df_cleaned.csv`")
            return None, None
        
        df = pd.read_csv(data_path)
        df_processed, df_provinsi = preprocess_data(df)
        return df_processed, df_provinsi
    except (OSError, pd.errors.ParserError, ValueError) as e:
        st.error(f"❌ **Error memuat data:** {str(e)}")
        return None, None

def create_correlation_heatmap(df):
    """Create correlation heatmap with proper sizing"""
    fig, ax = plt.subplots(figsize=(14, 10))
    numeric_df = df.select_dtypes(include=np.number)
    
    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
                   fmt=".2f", ax=ax, square=True,
                   linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        return fig
    else:
        st.warning("⚠️ Data tidak cukup untuk membuat heatmap korelasi.")
        return None

def create_scatter_plots(df):
    """Create scatter plots for features vs poverty"""
    all_feature_cols = [
        'Pengeluaran Per Kapita', 
        'Rata-Rata Lama Sekolah', 
        'Indeks Pembangunan Manusia',
        'Umur Harapan Hidup',
        'Tingkat Pengangguran Terbuka'
    ]
    
    available_features = [col for col in all_feature_cols if col in df.columns]
    
    if 'Persentase Kemiskinan (P0)' not in df.columns:
        st.warning("⚠️ Kolom 'Persentase Kemiskinan (P0)' tidak ditemukan.")
        return
    
    # Create grid layout
    cols_per_row = 2
    rows = (len(available_features) + cols_per_row - 1) // cols_per_row
    
    for row in range(rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            feature_idx = row * cols_per_row + col_idx
            if feature_idx < len(available_features):
                feature = available_features[feature_idx]
                with cols[col_idx]:
                    fig, ax = plt.subplots(figsize=(7, 5))
                    sns.scatterplot(data=df, x=feature, y='Persentase Kemiskinan (P0)', 
                                  ax=ax, alpha=0.6, s=50)
                    ax.set_title(f'Hubungan {feature} dengan Kemiskinan', 
                               fontsize=12, fontweight='bold')
                    ax.set_xlabel(feature, fontsize=10)
                    ax.set_ylabel('Persentase Kemiskinan (%)', fontsize=10)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

def run_eda_page():
    """
    Halaman Analisis Data Eksplorasi.
    """
    st.header("📊 Analisis Data Eksplorasi Kemiskinan")
    st.markdown("---")
    
    with st.spinner("Memuat data..."):
        df_processed, df_provinsi = load_data(DATA_PATH)

    if df_processed is not None:
        # Statistics Overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Provinsi", df_provinsi['Provinsi'].nunique() if df_provinsi is not None else 0)
        with col2:
            avg_poverty = df_provinsi['Persentase Kemiskinan (P0)'].mean() if 'Persentase Kemiskinan (P0)' in df_provinsi.columns else 0
            st.metric("Rata-rata Kemiskinan", f"{avg_poverty:.2f}%")
        with col3:
            total_data = len(df_processed)
            st.metric("Total Data Points", f"{total_data:,}")
        with col4:
            total_features = len(df_processed.select_dtypes(include=np.number).columns)
            st.metric("Jumlah Fitur", total_features)
        st.caption("Catatan: ringkasan provinsi menggunakan rata-rata sederhana dari kabupaten/kota yang tersedia.")

        # Data Preview
        st.subheader("📋 Pratinjau Data")
        with st.expander("Lihat Data"):
            st.dataframe(df_processed.head(20), use_container_width=True)
            
            # Download button
            csv = df_processed.to_csv(index=False)
            st.download_button(
                label="📥 Download Data CSV",
                data=csv,
                file_name="data_kemiskinan_indonesia.csv",
                mime="text/csv"
            )

        # Descriptive Statistics
        st.subheader("📈 Statistik Deskriptif")
        with st.expander("Lihat Statistik"):
            st.dataframe(df_processed.describe(), use_container_width=True)

        # Correlation Heatmap
        st.subheader("🔥 Heatmap Korelasi Antar Variabel")
        st.write("Heatmap menunjukkan korelasi antar variabel. Nilai mendekati 1 (merah) = korelasi positif kuat, "
                "mendekati -1 (biru) = korelasi negatif kuat.")
        
        fig = create_correlation_heatmap(df_processed)
        if fig:
            st.pyplot(fig)
            plt.close()

        # Scatter Plots
        st.subheader("📊 Distribusi Fitur vs Persentase Kemiskinan")
        st.write("Visualisasi hubungan antara setiap fitur dengan tingkat kemiskinan.")
        create_scatter_plots(df_processed)
        
    else:
        st.error("❌ **Gagal memuat data.** Periksa kembali file dan path-nya.")

def create_folium_map(df_provinsi, geojson_data):
    """Create interactive folium map"""
    geojson_data = copy.deepcopy(geojson_data)

    # Prepare GeoJSON with data
    df_provinsi_indexed = df_provinsi.set_index('Provinsi')
    
    for feature in geojson_data['features']:
        province_name = feature['properties'].get('name', '').upper().strip()
        
        if province_name in df_provinsi_indexed.index:
            province_data = df_provinsi_indexed.loc[province_name]
            feature['properties']['IPM_DISPLAY'] = f"{province_data.get('Indeks Pembangunan Manusia', 0):.2f}"
            feature['properties']['PENDUDUK_MISKIN'] = f"{province_data.get('Persentase Kemiskinan (P0)', 0):.2f}%"
            feature['properties']['LAMA_SEKOLAH'] = f"{province_data.get('Rata-Rata Lama Sekolah', 0):.2f} Tahun"
            feature['properties']['PENGELUARAN_KAPITA'] = f"{province_data.get('Pengeluaran Per Kapita', 0):,.0f} Ribu Rupiah/Bulan"
            feature['properties']['UMUR_HARAPAN_HIDUP'] = f"{province_data.get('Umur Harapan Hidup', 0):.2f} Tahun"
            feature['properties']['SANITASI_LAYAK'] = f"{province_data.get('Akses Sanitasi Layak', 0):.2f}%"
            feature['properties']['AIR_MINUM_LAYAK'] = f"{province_data.get('Akses Air Minum Layak', 0):.2f}%"
            feature['properties']['PENGANGGURAN'] = f"{province_data.get('Tingkat Pengangguran Terbuka', 0):.2f}%"
            feature['properties']['ANGKATAN_KERJA'] = f"{province_data.get('Tingkat Partisipasi Angkatan Kerja', 0):.2f}%"
            feature['properties']['PDRB'] = f"{province_data.get('PDRB', 0):,.0f} Rupiah"
        else:
            for prop in ['IPM_DISPLAY', 'PENDUDUK_MISKIN', 'LAMA_SEKOLAH', 'PENGELUARAN_KAPITA', 
                        'UMUR_HARAPAN_HIDUP', 'SANITASI_LAYAK', 'AIR_MINUM_LAYAK', 
                        'PENGANGGURAN', 'ANGKATAN_KERJA', 'PDRB']:
                feature['properties'][prop] = "Data Tidak Tersedia"

    # Create map
    m = folium.Map(
        location=[-2.5, 118.0], 
        zoom_start=5, 
        tiles='OpenStreetMap',
        control_scale=True
    )

    # Choropleth layer
    folium.Choropleth(
        geo_data=geojson_data,
        name='Peta Kemiskinan',
        data=df_provinsi,
        columns=['Provinsi', 'Persentase Kemiskinan (P0)'],
        key_on=GEOJSON_PROVINCE_KEY,
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.5,
        line_color='black',
        line_weight=1,
        legend_name='Persentase Penduduk Miskin (%)',
        highlight=True,
        nan_fill_color='lightgray'
    ).add_to(m)

    # Interactive layer with tooltip and popup
    tooltip = folium.features.GeoJsonTooltip(
        fields=['name', 'PENDUDUK_MISKIN'],
        aliases=['Provinsi:', 'Penduduk Miskin:'],
        localize=True,
        sticky=False,
        style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px; border-radius: 5px;"
    )

    popup = folium.features.GeoJsonPopup(
        fields=[
            'name', 'IPM_DISPLAY', 'PENDUDUK_MISKIN', 'LAMA_SEKOLAH',
            'PENGELUARAN_KAPITA', 'UMUR_HARAPAN_HIDUP', 'SANITASI_LAYAK',
            'AIR_MINUM_LAYAK', 'PENGANGGURAN', 'ANGKATAN_KERJA', 'PDRB'
        ],
        aliases=[
            'Provinsi:', 'IPM:', 'Penduduk Miskin:', 'Lama Sekolah:',
            'Pengeluaran per Kapita:', 'Harapan Hidup:', 'Sanitasi Layak:',
            'Air Minum Layak:', 'Pengangguran:', 'Partisipasi Angkatan Kerja:', 'PDRB:'
        ],
        localize=True,
        sticky=False,
        labels=True,
        max_width=450,
        style="font-family: arial; font-size: 12px;"
    )
    
    geojson_layer = folium.features.GeoJson(
        geojson_data,
        name='Detail Provinsi',
        style_function=lambda x: {
            'fillColor': 'transparent', 
            'color': 'transparent', 
            'weight': 0
        },
        control=False,
        tooltip=tooltip,
        popup=popup
    )
    geojson_layer.add_to(m)

    folium.LayerControl().add_to(m)
    
    return m

def run_map_page():
    """
    Halaman Visualisasi Peta Interaktif.
    """
    st.header("🗺️ Peta Sebaran Kemiskinan di Indonesia")
    st.markdown("---")
    st.info("💡 **Tip:** Klik pada provinsi untuk melihat detail lengkap atau hover untuk info cepat.")

    with st.spinner("Memuat peta..."):
        df_processed, df_provinsi = load_data(DATA_PATH)
        geojson_data = load_geojson(LOCAL_GEOJSON_PATH, GEOJSON_URL)

    if df_provinsi is not None and geojson_data is not None:
        required_cols = ['Provinsi', 'Persentase Kemiskinan (P0)']
        if not all(col in df_provinsi.columns for col in required_cols):
            st.error(f"❌ **Kolom yang diperlukan tidak ditemukan:** {required_cols}")
            return

        geojson_provinces = {
            feature['properties']['name'].upper().strip()
            for feature in geojson_data.get('features', [])
        }
        data_provinces = set(df_provinsi['Provinsi'])
        missing_in_geojson = sorted(data_provinces - geojson_provinces)
        missing_in_data = sorted(geojson_provinces - data_provinces)
        if missing_in_geojson or missing_in_data:
            st.warning(
                "⚠️ Ada perbedaan nama provinsi antara data dan GeoJSON. "
                f"Tidak ada di GeoJSON: {missing_in_geojson or '-'}; "
                f"Tidak ada di data: {missing_in_data or '-'}."
            )

        # Display map
        m = create_folium_map(df_provinsi, geojson_data)
        st_folium(m, width=None, height=600, use_container_width=True)
        st.caption("Catatan: angka provinsi dihitung sebagai rata-rata sederhana kabupaten/kota dalam dataset.")

        # Summary statistics
        st.subheader("📊 Ringkasan Statistik Provinsi")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔴 5 Provinsi Kemiskinan Tertinggi**")
            top_5 = df_provinsi.nlargest(5, 'Persentase Kemiskinan (P0)')[['Provinsi', 'Persentase Kemiskinan (P0)']]
            st.dataframe(top_5.reset_index(drop=True), use_container_width=True)
        
        with col2:
            st.markdown("**🟢 5 Provinsi Kemiskinan Terendah**")
            bottom_5 = df_provinsi.nsmallest(5, 'Persentase Kemiskinan (P0)')[['Provinsi', 'Persentase Kemiskinan (P0)']]
            st.dataframe(bottom_5.reset_index(drop=True), use_container_width=True)
    else:
        st.error("❌ **Gagal memuat data peta.** Periksa koneksi internet dan ketersediaan file.")

def main():
    """
    Fungsi utama aplikasi.
    """
    load_css()
    
    # Header
    st.title("🇮🇩 Dashboard Analisis Kemiskinan di Indonesia")
    st.markdown("*Dashboard interaktif untuk menganalisis data kemiskinan di Indonesia*")
    st.markdown("---")

    # Sidebar
    st.sidebar.title("📍 Navigasi")
    st.sidebar.markdown("---")
    
    page_options = {
        "🗺️ Visualisasi Peta": "map",
        "📊 Analisis Data Eksplorasi": "eda"
    }
    
    selected_page = st.sidebar.radio(
        "Pilih Halaman:",
        list(page_options.keys())
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 Tentang")
    st.sidebar.info(
        "Dashboard publik untuk eksplorasi indikator kemiskinan dan pembangunan "
        "di tingkat kabupaten/kota serta agregasi provinsi."
    )
    
    st.sidebar.markdown("### 🔗 Sumber Data")
    st.sidebar.markdown("- Dataset indikator sosial-ekonomi lokal")
    st.sidebar.markdown("- Batas provinsi lokal `data/prov 34.geojson`")

    # Route to selected page
    if page_options[selected_page] == "map":
        run_map_page()
    elif page_options[selected_page] == "eda":
        run_eda_page()

if __name__ == "__main__":
    main()

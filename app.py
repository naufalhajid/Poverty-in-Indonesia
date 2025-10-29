import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import folium
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
GEOJSON_PROVINCE_KEY = 'feature.properties.name'

# --- CUSTOM CSS ---
def load_css():
    """Load custom CSS for better UI"""
    st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Headers */
    h1 {
        color: #1f77b4;
        padding-bottom: 10px;
        font-weight: 700;
    }
    h2 {
        color: #2c3e50;
        padding-top: 20px;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 10px;
    }
    h3 {
        color: #34495e;
        margin-top: 15px;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: #f0f2f6;
        border-radius: 8px;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 600;
    }
    
    /* Custom info box */
    .info-box {
        background-color: #e3f2fd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 10px 0;
    }
    
    .insight-box {
        background-color: #fff3e0;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
        margin: 10px 0;
    }
    
    .conclusion-box {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        margin: 10px 0;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

def create_info_box(content, box_type="info"):
    """Create styled info boxes"""
    box_class = f"{box_type}-box"
    st.markdown(f'<div class="{box_class}">{content}</div>', unsafe_allow_html=True)

@st.cache_data
def preprocess_data(df):
    """Membersihkan, memproses, dan mengagregasi DataFrame."""
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

    valid_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
    df_processed = df[list(valid_columns.keys())].copy()
    df_processed.rename(columns=valid_columns, inplace=True)

    if 'Pengeluaran Per Kapita' in df_processed.columns:
        df_processed['Pengeluaran Per Kapita'] = df_processed['Pengeluaran Per Kapita'] / 12

    if 'Provinsi' in df_processed.columns:
        df_processed['Provinsi'] = df_processed['Provinsi'].str.upper().str.strip()

    df_provinsi = df_processed.groupby('Provinsi').mean(numeric_only=True).reset_index()

    return df_processed, df_provinsi

@st.cache_data
def load_geojson(url):
    """Memuat data GeoJSON dari URL dengan error handling."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.error("⏱️ **Timeout:** Koneksi ke server GeoJSON terlalu lama.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ **Gagal memuat GeoJSON:** {str(e)}")
        return None

@st.cache_data
def load_data(data_path):
    """Memuat dan memproses data kemiskinan dari file CSV."""
    try:
        if not os.path.exists(data_path):
            st.error(f"❌ **File tidak ditemukan:** `{data_path}`")
            return None, None
        
        df = pd.read_csv(data_path)
        df_processed, df_provinsi = preprocess_data(df)
        return df_processed, df_provinsi
    except Exception as e:
        st.error(f"❌ **Error memuat data:** {str(e)}")
        return None, None

def create_interactive_correlation_heatmap(df):
    """Create interactive correlation heatmap using Plotly"""
    numeric_df = df.select_dtypes(include=np.number)
    
    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Korelasi")
        ))
        
        fig.update_layout(
            title="Matriks Korelasi Antar Variabel",
            width=900,
            height=700,
            xaxis_tickangle=-45,
            font=dict(size=11)
        )
        
        return fig
    return None

def create_scatter_plotly(df, x_col, y_col='Persentase Kemiskinan (P0)', title=None):
    """Create interactive scatter plot using Plotly"""
    if title is None:
        title = f'Hubungan {x_col} dengan Kemiskinan'
    
    fig = px.scatter(
        df, 
        x=x_col, 
        y=y_col,
        hover_data=['Provinsi'] if 'Provinsi' in df.columns else None,
        trendline="ols",
        title=title,
        labels={x_col: x_col, y_col: 'Persentase Kemiskinan (%)'}
    )
    
    fig.update_traces(marker=dict(size=8, opacity=0.6))
    fig.update_layout(
        height=400,
        hovermode='closest',
        showlegend=True
    )
    
    return fig

def create_distribution_plot(df, column):
    """Create distribution plot using Plotly"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df[column],
        name='Histogram',
        nbinsx=30,
        opacity=0.7
    ))
    
    fig.update_layout(
        title=f'Distribusi {column}',
        xaxis_title=column,
        yaxis_title='Frekuensi',
        height=350,
        showlegend=True
    )
    
    return fig

def create_bar_chart_top_bottom(df, column, n=5):
    """Create bar chart for top and bottom provinces"""
    top_n = df.nlargest(n, column)
    bottom_n = df.nsmallest(n, column)
    
    combined = pd.concat([top_n, bottom_n])
    combined['Kategori'] = ['Tertinggi']*n + ['Terendah']*n
    
    fig = px.bar(
        combined,
        x=column,
        y='Provinsi',
        color='Kategori',
        orientation='h',
        title=f'Top {n} dan Bottom {n} Provinsi - {column}',
        color_discrete_map={'Tertinggi': '#ef5350', 'Terendah': '#66bb6a'}
    )
    
    fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
    
    return fig

def run_home_page():
    """Halaman Beranda dengan Ringkasan Eksekutif"""
    st.title("🇮🇩 Dashboard Analisis Kemiskinan di Indonesia")
    st.markdown("### *Platform Interaktif untuk Memahami Dinamika Kemiskinan*")
    st.markdown("---")
    
    # Executive Summary
    st.header("📋 Ringkasan Eksekutif")
    
    create_info_box("""
    <h3>🎯 Tentang Dashboard Ini</h3>
    <p>Dashboard ini dirancang untuk memberikan pemahaman komprehensif tentang kemiskinan di Indonesia 
    melalui analisis data multi-dimensi. Kami mengintegrasikan berbagai indikator pembangunan manusia 
    untuk mengidentifikasi pola, tren, dan faktor-faktor yang mempengaruhi tingkat kemiskinan.</p>
    """, "info")
    
    df_processed, df_provinsi = load_data(DATA_PATH)
    
    if df_processed is not None and df_provinsi is not None:
        # Key Statistics
        st.subheader("📊 Statistik Utama")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_prov = df_provinsi['Provinsi'].nunique()
            st.metric("📍 Total Provinsi", total_prov)
        
        with col2:
            avg_poverty = df_provinsi['Persentase Kemiskinan (P0)'].mean()
            st.metric("📉 Rata-rata Kemiskinan", f"{avg_poverty:.2f}%")
        
        with col3:
            avg_ipm = df_provinsi['Indeks Pembangunan Manusia'].mean()
            st.metric("📈 Rata-rata IPM", f"{avg_ipm:.2f}")
        
        with col4:
            total_data = len(df_processed)
            st.metric("📊 Total Data Points", f"{total_data:,}")
        
        st.markdown("---")
        
        # Quick Insights
        st.subheader("💡 Insight Kunci")
        
        col1, col2 = st.columns(2)
        
        with col1:
            create_info_box("""
            <h4>🔴 Provinsi dengan Kemiskinan Tertinggi</h4>
            <p>Identifikasi area yang membutuhkan intervensi prioritas untuk program pengentasan kemiskinan.</p>
            """, "insight")
            
            top_poverty = df_provinsi.nlargest(5, 'Persentase Kemiskinan (P0)')[['Provinsi', 'Persentase Kemiskinan (P0)']]
            st.dataframe(top_poverty.reset_index(drop=True), use_container_width=True)
        
        with col2:
            create_info_box("""
            <h4>🟢 Provinsi dengan Kemiskinan Terendah</h4>
            <p>Belajar dari best practices provinsi-provinsi dengan tingkat kemiskinan rendah.</p>
            """, "insight")
            
            bottom_poverty = df_provinsi.nsmallest(5, 'Persentase Kemiskinan (P0)')[['Provinsi', 'Persentase Kemiskinan (P0)']]
            st.dataframe(bottom_poverty.reset_index(drop=True), use_container_width=True)
        
        st.markdown("---")
        
        # Visualization: Poverty vs IPM
        st.subheader("📊 Visualisasi: Hubungan IPM dengan Kemiskinan")
        
        create_info_box("""
        <p><strong>Apa yang ditunjukkan grafik ini?</strong></p>
        <p>Grafik scatter plot ini menunjukkan hubungan negatif kuat antara Indeks Pembangunan Manusia (IPM) 
        dengan tingkat kemiskinan. Semakin tinggi IPM suatu provinsi, semakin rendah tingkat kemiskinannya.</p>
        <p><strong>Mengapa ini penting?</strong> IPM mengukur pencapaian rata-rata dalam dimensi kunci pembangunan 
        manusia: umur panjang dan sehat, pengetahuan, dan standar hidup layak. Korelasi negatif yang kuat 
        menunjukkan bahwa investasi dalam pendidikan, kesehatan, dan ekonomi adalah kunci pengentasan kemiskinan.</p>
        """, "info")
        
        fig = create_scatter_plotly(
            df_provinsi, 
            'Indeks Pembangunan Manusia', 
            'Persentase Kemiskinan (P0)',
            'Korelasi IPM dengan Tingkat Kemiskinan'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Conclusion Box
        create_info_box("""
        <h3>🎯 Kesimpulan Awal</h3>
        <p>Dashboard ini mengungkap bahwa kemiskinan di Indonesia bukan hanya masalah ekonomi, tetapi 
        juga masalah pembangunan manusia yang komprehensif. Provinsi dengan IPM tinggi cenderung memiliki 
        tingkat kemiskinan yang lebih rendah, menekankan pentingnya pendekatan multi-dimensi dalam 
        pengentasan kemiskinan.</p>
        <p><strong>Navigasi:</strong> Gunakan sidebar untuk mengeksplorasi analisis lebih mendalam tentang 
        peta geografis dan analisis data eksplorasi.</p>
        """, "conclusion")

def run_eda_page():
    """Halaman Analisis Data Eksplorasi dengan Penjelasan Edukatif"""
    st.header("📊 Analisis Data Eksplorasi (EDA)")
    st.markdown("### *Memahami Data Melalui Visualisasi Interaktif*")
    st.markdown("---")
    
    # Educational Introduction
    create_info_box("""
    <h3>📚 Apa itu Exploratory Data Analysis (EDA)?</h3>
    <p>EDA adalah proses menganalisis data untuk menemukan pola, anomali, dan insight menggunakan 
    visualisasi dan statistik deskriptif. Ini adalah langkah krusial sebelum membuat model atau 
    mengambil keputusan berbasis data.</p>
    <p><strong>Tujuan EDA dalam konteks kemiskinan:</strong></p>
    <ul>
        <li>Memahami distribusi dan variasi data kemiskinan</li>
        <li>Mengidentifikasi hubungan antar variabel</li>
        <li>Menemukan outliers dan anomali</li>
        <li>Validasi asumsi untuk analisis lanjutan</li>
    </ul>
    """, "info")
    
    with st.spinner("Memuat data..."):
        df_processed, df_provinsi = load_data(DATA_PATH)

    if df_processed is not None:
        # Statistics Overview
        st.subheader("📈 Statistik Deskriptif")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Provinsi", df_provinsi['Provinsi'].nunique())
        with col2:
            avg_poverty = df_provinsi['Persentase Kemiskinan (P0)'].mean()
            st.metric("Rata-rata Kemiskinan", f"{avg_poverty:.2f}%")
        with col3:
            std_poverty = df_provinsi['Persentase Kemiskinan (P0)'].std()
            st.metric("Std Dev Kemiskinan", f"{std_poverty:.2f}%")
        with col4:
            total_features = len(df_processed.select_dtypes(include=np.number).columns)
            st.metric("Jumlah Fitur", total_features)
        
        create_info_box("""
        <p><strong>Interpretasi Statistik:</strong> Standar deviasi menunjukkan variasi data. 
        Std dev yang tinggi mengindikasikan kesenjangan yang besar antar provinsi dalam tingkat kemiskinan.</p>
        """, "insight")
        
        st.markdown("---")
        
        # Data Preview
        st.subheader("📋 Pratinjau Data")
        with st.expander("📖 Klik untuk melihat data mentah"):
            st.dataframe(df_processed.head(20), use_container_width=True)
            
            csv = df_processed.to_csv(index=False)
            st.download_button(
                label="📥 Download Data CSV",
                data=csv,
                file_name="data_kemiskinan_indonesia.csv",
                mime="text/csv"
            )
        
        # Descriptive Statistics
        with st.expander("📊 Lihat Statistik Deskriptif Lengkap"):
            st.dataframe(df_processed.describe(), use_container_width=True)
            
            create_info_box("""
            <p><strong>Cara membaca statistik deskriptif:</strong></p>
            <ul>
                <li><strong>Mean:</strong> Nilai rata-rata</li>
                <li><strong>Std:</strong> Standar deviasi (ukuran variasi)</li>
                <li><strong>Min/Max:</strong> Nilai minimum dan maksimum</li>
                <li><strong>25%, 50%, 75%:</strong> Quartiles (pembagian data menjadi 4 bagian)</li>
            </ul>
            """, "info")
        
        st.markdown("---")
        
        # Distribution Analysis
        st.subheader("📊 Analisis Distribusi Data")
        
        create_info_box("""
        <h4>🎯 Mengapa Analisis Distribusi Penting?</h4>
        <p>Memahami distribusi data membantu kita mengetahui:</p>
        <ul>
            <li>Apakah data terdistribusi normal atau skewed?</li>
            <li>Apakah ada outliers yang signifikan?</li>
            <li>Bagaimana data terkonsentrasi di sekitar nilai tertentu?</li>
        </ul>
        """, "info")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_distribution_plot(df_provinsi, 'Persentase Kemiskinan (P0)')
            st.plotly_chart(fig, use_container_width=True)
            
            create_info_box("""
            <p><strong>Interpretasi:</strong> Histogram menunjukkan sebaran frekuensi tingkat kemiskinan. 
            Jika data menumpuk di satu sisi, itu menunjukkan skewness (kemiringan).</p>
            """, "insight")
        
        with col2:
            fig = create_distribution_plot(df_provinsi, 'Indeks Pembangunan Manusia')
            st.plotly_chart(fig, use_container_width=True)
            
            create_info_box("""
            <p><strong>Interpretasi:</strong> Distribusi IPM yang lebih merata menunjukkan 
            pembangunan manusia yang relatif konsisten antar provinsi.</p>
            """, "insight")
        
        st.markdown("---")
        
        # Correlation Heatmap
        st.subheader("🔥 Analisis Korelasi Antar Variabel")
        
        create_info_box("""
        <h4>🎓 Memahami Korelasi</h4>
        <p><strong>Korelasi</strong> mengukur kekuatan dan arah hubungan linear antara dua variabel:</p>
        <ul>
            <li><strong>+1.0:</strong> Korelasi positif sempurna (naik bersamaan)</li>
            <li><strong>-1.0:</strong> Korelasi negatif sempurna (berbanding terbalik)</li>
            <li><strong>0.0:</strong> Tidak ada hubungan linear</li>
            <li><strong>|r| > 0.7:</strong> Korelasi kuat</li>
            <li><strong>0.3 < |r| < 0.7:</strong> Korelasi sedang</li>
            <li><strong>|r| < 0.3:</strong> Korelasi lemah</li>
        </ul>
        <p><strong>⚠️ Penting:</strong> Korelasi ≠ Kausalitas! Dua variabel bisa berkorelasi tanpa 
        saling menyebabkan.</p>
        """, "info")
        
        fig = create_interactive_correlation_heatmap(df_processed)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Key Findings from Correlation
        st.markdown("#### 🔍 Temuan Kunci dari Analisis Korelasi:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            create_info_box("""
            <h4>💡 Korelasi Negatif Kuat</h4>
            <p><strong>IPM vs Kemiskinan (-0.71):</strong></p>
            <p>Provinsi dengan IPM tinggi cenderung memiliki kemiskinan rendah. Ini menunjukkan bahwa 
            investasi dalam pendidikan, kesehatan, dan ekonomi adalah strategi efektif pengentasan kemiskinan.</p>
            <p><strong>Pengeluaran Per Kapita vs Kemiskinan (-0.64):</strong></p>
            <p>Daya beli yang tinggi berkorelasi dengan kemiskinan rendah, mengkonfirmasi pentingnya 
            peningkatan pendapatan masyarakat.</p>
            """, "insight")
        
        with col2:
            create_info_box("""
            <h4>⚠️ Temuan Mengejutkan</h4>
            <p><strong>PDRB vs Kemiskinan (-0.24):</strong></p>
            <p>Korelasi yang sangat lemah! Ini menunjukkan bahwa pertumbuhan ekonomi (PDRB tinggi) 
            tidak otomatis menurunkan kemiskinan. Fenomena "growth without development".</p>
            <p><strong>TPAK vs Kemiskinan (+0.46):</strong></p>
            <p>Paradoks! Partisipasi kerja tinggi justru di daerah miskin. Ini karena tekanan ekonomi 
            memaksa semua orang bekerja, bukan indikator ekonomi yang baik.</p>
            """, "insight")
        
        st.markdown("---")
        
        # Scatter Plots Analysis
        st.subheader("📊 Analisis Hubungan Bivariate")
        
        create_info_box("""
        <h4>📚 Scatter Plot untuk Analisis Hubungan</h4>
        <p>Scatter plot memvisualisasikan hubungan antara dua variabel kontinu. Garis tren (trendline) 
        menunjukkan arah umum hubungan:</p>
        <ul>
            <li><strong>Garis menurun:</strong> Hubungan negatif (satu naik, yang lain turun)</li>
            <li><strong>Garis mendatar:</strong> Tidak ada hubungan</li>
            <li><strong>Garis menaik:</strong> Hubungan positif (naik bersamaan)</li>
        </ul>
        <p>Poin-poin yang tersebar menunjukkan variasi; poin yang menjauh dari garis adalah outliers.</p>
        """, "info")
        
        features = [
            ('Indeks Pembangunan Manusia', 'IPM adalah indikator komprehensif pembangunan'),
            ('Pengeluaran Per Kapita', 'Pengeluaran mencerminkan daya beli dan kemampuan ekonomi'),
            ('Rata-Rata Lama Sekolah', 'Pendidikan adalah investasi jangka panjang keluar dari kemiskinan'),
            ('Umur Harapan Hidup', 'Kesehatan yang baik mendukung produktivitas ekonomi'),
            ('Akses Sanitasi Layak', 'Sanitasi yang buruk memicu biaya kesehatan dan produktivitas rendah')
        ]
        
        for i in range(0, len(features), 2):
            cols = st.columns(2)
            for idx, col in enumerate(cols):
                if i + idx < len(features):
                    feature, explanation = features[i + idx]
                    with col:
                        if feature in df_provinsi.columns:
                            fig = create_scatter_plotly(df_provinsi, feature)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            with st.expander(f"💡 Interpretasi: {feature}"):
                                st.write(explanation)
        
        st.markdown("---")
        
        # Top/Bottom Analysis
        st.subheader("🏆 Analisis Ranking Provinsi")
        
        create_info_box("""
        <h4>🎯 Mengapa Ranking Penting?</h4>
        <p>Membandingkan provinsi terbaik dan terburuk membantu:</p>
        <ul>
            <li>Mengidentifikasi best practices dari provinsi berkinerja baik</li>
            <li>Menentukan prioritas intervensi untuk provinsi berkinerja buruk</li>
            <li>Memahami kesenjangan regional dalam pembangunan</li>
        </ul>
        """, "info")
        
        selected_metric = st.selectbox(
            "Pilih Metrik untuk Analisis Ranking:",
            ['Persentase Kemiskinan (P0)', 'Indeks Pembangunan Manusia', 
             'Pengeluaran Per Kapita', 'Rata-Rata Lama Sekolah']
        )
        
        if selected_metric in df_provinsi.columns:
            fig = create_bar_chart_top_bottom(df_provinsi, selected_metric)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Key Takeaways
        st.subheader("🎯 Kesimpulan Analisis Eksplorasi")
        
        create_info_box("""
        <h3>📌 Temuan Utama:</h3>
        <ol>
            <li><strong>IPM adalah Prediktor Terkuat:</strong> Dengan korelasi -0.71, IPM menunjukkan 
            hubungan paling kuat dengan kemiskinan. Fokus pada pembangunan manusia komprehensif adalah kunci.</li>
            
            <li><strong>Pertumbuhan Ekonomi ≠ Penurunan Kemiskinan:</strong> PDRB hanya berkorelasi -0.24 
            dengan kemiskinan. Yang penting adalah distribusi dan inklusivitas pertumbuhan, bukan hanya size ekonomi.</li>
            
            <li><strong>Kesenjangan Regional Signifikan:</strong> Standar deviasi yang tinggi menunjukkan 
            disparitas besar antar provinsi dalam berbagai indikator pembangunan.</li>
            
            <li><strong>Pendekatan Multi-dimensi Diperlukan:</strong> Kemiskinan dipengaruhi oleh banyak 
            faktor yang saling terkait: pendidikan, kesehatan, infrastruktur, dan ekonomi.</li>
        </ol>
        
        <h3>💡 Rekomendasi Kebijakan:</h3>
        <ul>
            <li>Prioritaskan investasi dalam pendidikan dan kesehatan</li>
            <li>Pastikan pertumbuhan ekonomi yang inklusif dan merata</li>
            <li>Perkuat infrastruktur dasar (sanitasi, air bersih)</li>
            <li>Fokus pada quality of employment, bukan hanya quantity</li>
            <li>Tailor intervensi sesuai karakteristik spesifik setiap provinsi</li>
        </ul>
        """, "conclusion")
    
    else:
        st.error("❌ **Gagal memuat data.** Periksa kembali file dan path-nya.")

def create_folium_map(df_provinsi, geojson_data):
    """Create interactive folium map with enhanced features"""
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

    m = folium.Map(
        location=[-2.5, 118.0], 
        zoom_start=5, 
        tiles='OpenStreetMap',
        control_scale=True
    )

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
    """Halaman Visualisasi Peta dengan Penjelasan Geografis"""
    st.header("🗺️ Peta Sebaran Kemiskinan di Indonesia")
    st.markdown("### *Analisis Spasial Kemiskinan Berdasarkan Wilayah*")
    st.markdown("---")
    
    # Educational Introduction
    create_info_box("""
    <h3>🌍 Mengapa Analisis Geografis Penting?</h3>
    <p>Peta interaktif ini memvisualisasikan distribusi spasial kemiskinan di Indonesia. Analisis 
    geografis membantu kita:</p>
    <ul>
        <li><strong>Identifikasi Pola Spasial:</strong> Melihat clustering kemiskinan di wilayah tertentu</li>
        <li><strong>Regional Disparities:</strong> Memahami kesenjangan antar wilayah Indonesia Barat dan Timur</li>
        <li><strong>Targeted Intervention:</strong> Menentukan prioritas geografis untuk program pengentasan</li>
        <li><strong>Proximity Effects:</strong> Mengidentifikasi spillover effects antar provinsi tetangga</li>
    </ul>
    <p><strong>💡 Cara Menggunakan Peta:</strong> Hover mouse di atas provinsi untuk info cepat, 
    klik untuk detail lengkap semua indikator.</p>
    """, "info")

    with st.spinner("Memuat peta..."):
        df_processed, df_provinsi = load_data(DATA_PATH)
        geojson_data = load_geojson(GEOJSON_URL)

    if df_provinsi is not None and geojson_data is not None:
        required_cols = ['Provinsi', 'Persentase Kemiskinan (P0)']
        if not all(col in df_provinsi.columns for col in required_cols):
            st.error(f"❌ **Kolom yang diperlukan tidak ditemukan:** {required_cols}")
            return

        # Display map
        m = create_folium_map(df_provinsi, geojson_data)
        st_folium(m, width=None, height=600, use_container_width=True)
        
        # Map Legend Explanation
        create_info_box("""
        <h4>🎨 Membaca Peta Choropleth</h4>
        <p><strong>Gradasi Warna:</strong></p>
        <ul>
            <li><strong>Kuning Muda:</strong> Tingkat kemiskinan rendah (< 5%)</li>
            <li><strong>Oranye:</strong> Tingkat kemiskinan sedang (5-10%)</li>
            <li><strong>Merah Tua:</strong> Tingkat kemiskinan tinggi (> 10%)</li>
        </ul>
        <p>Peta ini menggunakan teknik <strong>Choropleth Mapping</strong>, di mana intensitas warna 
        mewakili nilai data di setiap wilayah geografis.</p>
        """, "info")
        
        st.markdown("---")

        # Summary statistics
        st.subheader("📊 Analisis Komparatif Antar Provinsi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔴 Top 5 Provinsi - Kemiskinan Tertinggi")
            
            create_info_box("""
            <p><strong>Area Prioritas:</strong> Provinsi-provinsi ini membutuhkan intervensi mendesak. 
            Tingkat kemiskinan tinggi sering berkorelasi dengan:</p>
            <ul>
                <li>Akses pendidikan terbatas</li>
                <li>Infrastruktur kesehatan kurang memadai</li>
                <li>Isolasi geografis</li>
                <li>Ketergantungan pada sektor ekonomi tertentu</li>
            </ul>
            """, "insight")
            
            top_5 = df_provinsi.nlargest(5, 'Persentase Kemiskinan (P0)')[
                ['Provinsi', 'Persentase Kemiskinan (P0)', 'Indeks Pembangunan Manusia']
            ]
            st.dataframe(top_5.reset_index(drop=True), use_container_width=True)
        
        with col2:
            st.markdown("#### 🟢 Top 5 Provinsi - Kemiskinan Terendah")
            
            create_info_box("""
            <p><strong>Best Practices:</strong> Provinsi-provinsi ini dapat menjadi model pembelajaran. 
            Faktor kesuksesan umumnya meliputi:</p>
            <ul>
                <li>Investasi tinggi dalam pendidikan</li>
                <li>Diversifikasi ekonomi</li>
                <li>Infrastruktur yang baik</li>
                <li>Tata kelola pemerintahan efektif</li>
            </ul>
            """, "insight")
            
            bottom_5 = df_provinsi.nsmallest(5, 'Persentase Kemiskinan (P0)')[
                ['Provinsi', 'Persentase Kemiskinan (P0)', 'Indeks Pembangunan Manusia']
            ]
            st.dataframe(bottom_5.reset_index(drop=True), use_container_width=True)
        
        st.markdown("---")
        
        # Regional Analysis
        st.subheader("🌏 Analisis Regional: Disparitas Geografis")
        
        # Create regional comparison visualization
        if 'Persentase Kemiskinan (P0)' in df_provinsi.columns:
            # Define regions (simplified)
            barat = df_provinsi[df_provinsi['Provinsi'].isin([
                'DKI JAKARTA', 'JAWA BARAT', 'JAWA TENGAH', 'JAWA TIMUR', 
                'BANTEN', 'DI YOGYAKARTA', 'SUMATERA UTARA', 'SUMATERA BARAT'
            ])]
            
            timur = df_provinsi[df_provinsi['Provinsi'].isin([
                'PAPUA', 'PAPUA BARAT', 'MALUKU', 'MALUKU UTARA', 
                'NUSA TENGGARA TIMUR', 'NUSA TENGGARA BARAT'
            ])]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_barat = barat['Persentase Kemiskinan (P0)'].mean() if len(barat) > 0 else 0
                st.metric("📍 Indonesia Barat", f"{avg_barat:.2f}%", 
                         delta=None, delta_color="normal")
            
            with col2:
                avg_timur = timur['Persentase Kemiskinan (P0)'].mean() if len(timur) > 0 else 0
                st.metric("📍 Indonesia Timur", f"{avg_timur:.2f}%", 
                         delta=None, delta_color="inverse")
            
            with col3:
                gap = avg_timur - avg_barat
                st.metric("📊 Kesenjangan", f"{gap:.2f}%", 
                         delta=None, delta_color="off")
            
            create_info_box("""
            <h4>🔍 Interpretasi Disparitas Regional</h4>
            <p>Indonesia menghadapi kesenjangan signifikan antara wilayah Barat dan Timur. 
            Faktor-faktor penyebab meliputi:</p>
            <ol>
                <li><strong>Sejarah Pembangunan:</strong> Fokus pembangunan historis di Jawa dan Sumatera</li>
                <li><strong>Geografis:</strong> Wilayah Timur lebih terisolasi dengan biaya logistik tinggi</li>
                <li><strong>Infrastruktur:</strong> Ketimpangan akses infrastruktur dasar</li>
                <li><strong>Investasi:</strong> Konsentrasi investasi swasta di wilayah Barat</li>
            </ol>
            <p><strong>Implikasi Kebijakan:</strong> Diperlukan affirmative action dan positive discrimination 
            untuk wilayah Indonesia Timur guna mengurangi kesenjangan struktural ini.</p>
            """, "insight")
        
        st.markdown("---")
        
        # Interactive Comparison Tool
        st.subheader("🔬 Alat Perbandingan Provinsi")
        
        create_info_box("""
        <p>Gunakan tool ini untuk membandingkan indikator pembangunan antara dua provinsi pilihan Anda.</p>
        """, "info")
        
        col1, col2 = st.columns(2)
        
        with col1:
            prov1 = st.selectbox("Pilih Provinsi 1:", df_provinsi['Provinsi'].tolist(), key='prov1')
        
        with col2:
            prov2 = st.selectbox("Pilih Provinsi 2:", df_provinsi['Provinsi'].tolist(), key='prov2')
        
        if prov1 and prov2 and prov1 != prov2:
            comparison_metrics = [
                'Persentase Kemiskinan (P0)',
                'Indeks Pembangunan Manusia',
                'Rata-Rata Lama Sekolah',
                'Pengeluaran Per Kapita',
                'Umur Harapan Hidup'
            ]
            
            data1 = df_provinsi[df_provinsi['Provinsi'] == prov1][comparison_metrics].iloc[0]
            data2 = df_provinsi[df_provinsi['Provinsi'] == prov2][comparison_metrics].iloc[0]
            
            comparison_df = pd.DataFrame({
                prov1: data1.values,
                prov2: data2.values,
                'Selisih': data2.values - data1.values
            }, index=comparison_metrics)
            
            st.dataframe(comparison_df.style.format("{:.2f}"), use_container_width=True)
            
            # Visualization
            fig = go.Figure()
            fig.add_trace(go.Bar(name=prov1, x=comparison_metrics, y=data1.values, marker_color='#42a5f5'))
            fig.add_trace(go.Bar(name=prov2, x=comparison_metrics, y=data2.values, marker_color='#ef5350'))
            
            fig.update_layout(
                title=f'Perbandingan {prov1} vs {prov2}',
                barmode='group',
                height=400,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Conclusions
        st.subheader("🎯 Kesimpulan Analisis Geografis")
        
        create_info_box("""
        <h3>📌 Insight Utama dari Peta:</h3>
        <ol>
            <li><strong>Clustering Geografis:</strong> Kemiskinan menunjukkan pola clustering - provinsi 
            dengan kemiskinan tinggi cenderung berdekatan secara geografis, mengindikasikan faktor regional.</li>
            
            <li><strong>Disparitas Barat-Timur:</strong> Indonesia Timur menghadapi tantangan kemiskinan 
            lebih besar dibanding Barat, mencerminkan kesenjangan struktural historis.</li>
            
            <li><strong>Urban-Rural Divide:</strong> Provinsi dengan pusat urban besar (Jakarta, Jawa) 
            umumnya memiliki tingkat kemiskinan lebih rendah.</li>
            
            <li><strong>Natural Resources Paradox:</strong> Beberapa provinsi kaya sumber daya alam 
            (Papua, Kalimantan) justru memiliki kemiskinan tinggi - resource curse phenomenon.</li>
        </ol>
        
        <h3>💡 Rekomendasi Berbasis Geografis:</h3>
        <ul>
            <li><strong>Indonesia Timur:</strong> Massive infrastructure investment, connectivity improvement, 
            dan special economic zones</li>
            <li><strong>Provinsi Terisolasi:</strong> Digital connectivity untuk akses pendidikan dan 
            kesehatan online</li>
            <li><strong>Resource-Rich Provinces:</strong> Strengthen revenue sharing dan local content policies</li>
            <li><strong>Spillover Strategy:</strong> Leverage kesuksesan provinsi maju untuk mengangkat 
            provinsi tetangga melalui regional cooperation</li>
        </ul>
        """, "conclusion")
    
    else:
        st.error("❌ **Gagal memuat data peta.** Periksa koneksi internet dan ketersediaan file.")

def run_insights_page():
    """Halaman Insight dan Rekomendasi Kebijakan"""
    st.header("💡 Insight & Rekomendasi Kebijakan")
    st.markdown("### *Dari Data ke Actionable Insights*")
    st.markdown("---")
    
    df_processed, df_provinsi = load_data(DATA_PATH)
    
    if df_processed is not None and df_provinsi is not None:
        # Key Insights Section
        st.subheader("🔍 Temuan Kunci dari Analisis")
        
        create_info_box("""
        <h3>1️⃣ IPM sebagai Golden Metric</h3>
        <p><strong>Temuan:</strong> IPM berkorelasi -0.71 dengan kemiskinan, paling kuat dibanding 
        indikator lainnya.</p>
        <p><strong>Mengapa?</strong> IPM adalah composite index yang mengukur:</p>
        <ul>
            <li>Kesehatan (umur harapan hidup)</li>
            <li>Pendidikan (lama sekolah dan harapan lama sekolah)</li>
            <li>Standar hidup (pengeluaran per kapita)</li>
        </ul>
        <p><strong>Implikasi:</strong> Peningkatan simultan di ketiga dimensi ini menghasilkan dampak 
        terbesar dalam pengentasan kemiskinan.</p>
        """, "info")
        
        create_info_box("""
        <h3>2️⃣ Growth vs Development Paradox</h3>
        <p><strong>Temuan:</strong> PDRB hanya berkorelasi -0.24 dengan kemiskinan.</p>
        <p><strong>Fenomena:</strong> "Jobless growth" atau "growth without development" - ekonomi tumbuh 
        tapi kemiskinan tidak turun signifikan.</p>
        <p><strong>Penyebab:</strong></p>
        <ul>
            <li>Pertumbuhan terkonsentrasi di sektor padat modal (migas, pertambangan)</li>
            <li>Ketimpangan distribusi pendapatan tinggi</li>
            <li>Trickle-down effect tidak berjalan optimal</li>
        </ul>
        <p><strong>Solusi:</strong> Fokus pada inclusive growth - pertumbuhan yang menyerap tenaga kerja 
        dan mendistribusikan manfaat secara merata.</p>
        """, "insight")
        
        create_info_box("""
        <h3>3️⃣ Employment Quality over Quantity</h3>
        <p><strong>Temuan:</strong> TPAK berkorelasi +0.46 dengan kemiskinan (paradoks!).</p>
        <p><strong>Interpretasi:</strong> Tingkat partisipasi kerja tinggi di daerah miskin bukan karena 
        ekonomi bagus, tapi karena:</p>
        <ul>
            <li>Semua anggota keluarga terpaksa bekerja untuk survival</li>
            <li>Banyak unpaid family workers</li>
            <li>Underemployment tinggi (bekerja di bawah kapasitas)</li>
        </ul>
        <p><strong>Lesson:</strong> Yang penting bukan seberapa banyak orang bekerja, tapi seberapa 
        layak (decent) pekerjaan mereka.</p>
        """, "insight")
        
        st.markdown("---")
        
        # Policy Recommendations
        st.subheader("🎯 Rekomendasi Kebijakan Berbasis Data")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎓 Pendidikan", 
            "🏥 Kesehatan", 
            "💼 Ekonomi", 
            "🏗️ Infrastruktur"
        ])
        
        with tab1:
            create_info_box("""
            <h3>📚 Strategi Pendidikan untuk Pengentasan Kemiskinan</h3>
            
            <h4>🎯 Prioritas Jangka Pendek (1-3 tahun):</h4>
            <ul>
                <li><strong>Beasiswa Targeted:</strong> Program beasiswa khusus untuk anak dari keluarga miskin 
                dengan focus pada STEM dan vocational skills</li>
                <li><strong>School Feeding Programs:</strong> Makan siang gratis di sekolah untuk meningkatkan 
                enrollment dan mengurangi dropout</li>
                <li><strong>Digital Literacy:</strong> Akselerasi digital skills untuk mengakses ekonomi digital</li>
            </ul>
            
            <h4>🎯 Prioritas Jangka Menengah (3-5 tahun):</h4>
            <ul>
                <li><strong>Wajib Belajar 12 Tahun:</strong> Universal secondary education dengan subsidi penuh 
                untuk keluarga miskin</li>
                <li><strong>Teacher Quality:</strong> Improve kompensasi dan training guru, terutama di daerah 
                terpencil</li>
                <li><strong>Vocational Training:</strong> Expand SMK dengan kurikulum berbasis industri</li>
            </ul>
            
            <h4>📊 Target Indikator:</h4>
            <ul>
                <li>Rata-rata lama sekolah naik 1 tahun dalam 5 tahun</li>
                <li>Enrollment rate SMA mencapai 95%</li>
                <li>Literacy rate 100% untuk usia 15-24</li>
            </ul>
            """, "info")
        
        with tab2:
            create_info_box("""
            <h3>🏥 Strategi Kesehatan untuk Pembangunan Manusia</h3>
            
            <h4>🎯 Prioritas Jangka Pendek:</h4>
            <ul>
                <li><strong>Universal Healthcare:</strong> Perluas cakupan BPJS dengan subsidi 100% untuk 
                keluarga miskin</li>
                <li><strong>Preventive Care:</strong> Program imunisasi massal dan maternal health</li>
                <li><strong>Nutrition Programs:</strong> Suplementasi gizi untuk ibu hamil dan balita</li>
            </ul>
            
            <h4>🎯 Prioritas Jangka Menengah:</h4>
            <ul>
                <li><strong>Health Infrastructure:</strong> Bangun Puskesmas dan Posyandu di setiap desa 
                tertinggal</li>
                <li><strong>Telemedicine:</strong> Leverage teknologi untuk akses kesehatan di daerah remote</li>
                <li><strong>Mental Health:</strong> Integrasikan layanan kesehatan mental dalam primary care</li>
            </ul>
            
            <h4>📊 Target Indikator:</h4>
            <ul>
                <li>Umur harapan hidup naik 2 tahun dalam 5 tahun</li>
                <li>Infant mortality rate turun 20%</li>
                <li>Universal health coverage 100%</li>
            </ul>
            """, "info")
        
        with tab3:
            create_info_box("""
            <h3>💼 Strategi Ekonomi Inklusif</h3>
            
            <h4>🎯 Strategi Makro:</h4>
            <ul>
                <li><strong>Labor-Intensive Growth:</strong> Prioritas sektor padat karya (manufaktur, 
                konstruksi, pariwisata)</li>
                <li><strong>SME Development:</strong> Kemudahan akses kredit mikro dan pendampingan UMKM</li>
                <li><strong>Fair Wages:</strong> Enforce upah minimum dan expand social protection</li>
            </ul>
            
            <h4>🎯 Strategi Mikro:</h4>
            <ul>
                <li><strong>Conditional Cash Transfers:</strong> Bantuan tunai bersyarat (pendidikan, kesehatan)</li>
                <li><strong>Skill Training:</strong> Pelatihan vokasi aligned dengan kebutuhan industri</li>
                <li><strong>Entrepreneurship Support:</strong> Inkubator bisnis dan mentoring untuk usaha mikro</li>
            </ul>
            
            <h4>📊 Target Indikator:</h4>
            <ul>
                <li>Pengeluaran per kapita naik 15% dalam 5 tahun</li>
                <li>Gini ratio turun dari 0.38 ke 0.35</li>
                <li>Decent work jobs naik 25%</li>
            </ul>
            """, "info")
        
        with tab4:
            create_info_box("""
            <h3>🏗️ Strategi Infrastruktur sebagai Enabler</h3>
            
            <h4>🎯 Infrastruktur Dasar:</h4>
            <ul>
                <li><strong>Sanitasi & Air Bersih:</strong> Prioritas tertinggi - korelasi kuat dengan kemiskinan</li>
                <li><strong>Listrik:</strong> Universal electrification untuk akses ekonomi digital</li>
                <li><strong>Jalan & Transportasi:</strong> Connect isolated areas untuk akses pasar</li>
            </ul>
            
            <h4>🎯 Infrastruktur Digital:</h4>
            <ul>
                <li><strong>Broadband Coverage:</strong> Internet cepat di semua kecamatan</li>
                <li><strong>Digital Government:</strong> E-government untuk efisiensi layanan publik</li>
                <li><strong>E-Commerce Enabler:</strong> Logistics infrastructure untuk UMKM online</li>
            </ul>
            
            <h4>📊 Target Indikator:</h4>
            <ul>
                <li>Akses sanitasi layak 95% dalam 5 tahun</li>
                <li>Akses air bersih 98% dalam 5 tahun</li>
                <li>Broadband coverage 80% populasi</li>
            </ul>
            """, "info")
        
        st.markdown("---")
        
        # Implementation Strategy
        st.subheader("🚀 Strategi Implementasi")
        
        create_info_box("""
        <h3>📋 Framework Implementasi: Theory of Change</h3>
        
        <h4>1️⃣ INPUTS (Resources)</h4>
        <ul>
            <li>Budget allocation: Minimum 20% APBN untuk pendidikan & kesehatan</li>
            <li>Human resources: Trained workers dan policy makers</li>
            <li>Technology: Digital platforms untuk monitoring</li>
            <li>Partnerships: Kolaborasi pemerintah-swasta-civil society</li>

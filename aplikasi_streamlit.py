import io
import os
import base64
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import fungsi pembantu dari utils
from utils import run_simulation, c_briere, T_min_b, T_max_b, p_hv, p_vh

# ==========================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="SEIR-LM Dengue Simulation Dashboard",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Definisikan path data default (sesuai kode Dash Anda)
bandung_path = "POWER_Point_Daily_Bandung.csv"
jakarta_path = "POWER_Point_Daily_JakartaPusat.csv"

# ==========================================
# 2. TATA LETAK APLIKASI (HEADER & SIDEBAR)
# ==========================================

# Header Utama
st.title("🦟 SEIR-LM Dengue Simulation Dashboard")
st.markdown("*Simulator Epidemiologi Demam Berdarah Dengue (DBD) berbasis Dinamika Iklim Cuaca (Suhu & Curah Hujan)*")
st.divider()

# --- SIDEBAR KONTROL ---
st.sidebar.header("⚙️ Pengaturan Simulasi")

# 1. Pilihan Kota
city_options = {
    "Kota Bandung (Baseline Kalibrasi)": "bandung",
    "Kota Jakarta Pusat (Baseline Kalibrasi)": "jakarta",
    "Kota Kustom (Parameter Bebas)": "custom"
}
selected_city_label = st.sidebar.selectbox(
    "Pilih Kota Simulasi:", 
    list(city_options.keys())
)
city_type = city_options[selected_city_label]

# 2. Jenis Sumber Data Iklim
climate_options = {
    "Data Historis Default": "default",
    "Unggah File CSV Baru": "upload",
    "Simulasi Cuaca Manual": "manual"
}
selected_climate_label = st.sidebar.radio(
    "Sumber Data Iklim:", 
    list(climate_options.keys())
)
climate_type = climate_options[selected_climate_label]

# Placeholder untuk dataframe iklim
climate_df = None
upload_contents = None

# 3. Area Unggah File CSV (Conditional)
if climate_type == "upload":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Unggah CSV Iklim Anda")
    uploaded_file = st.sidebar.file_uploader("Pilih File CSV", type=['csv'])
    
    # Template CSV untuk didownload
    template_csv = "Date,T2M,PRECTOTCORR\n2026-01-01,25.5,10.0\n2026-01-02,26.0,0.0"
    st.sidebar.download_button(
        label="📥 Unduh Template CSV",
        data=template_csv,
        file_name="template_iklim.csv",
        mime="text/csv"
    )
    
    if uploaded_file is not None:
        try:
            df_test = pd.read_csv(uploaded_file)
            required_cols = ["Date", "T2M", "PRECTOTCORR"]
            missing = [col for col in required_cols if col not in df_test.columns]
            
            if missing:
                st.sidebar.error(f"Error: Kolom {missing} tidak ditemukan.")
            else:
                st.sidebar.success(f"File berhasil diunggah ({len(df_test)} baris).")
                # Format untuk proses selanjutnya
                climate_df = df_test.copy()
                climate_df['Date'] = pd.to_datetime(climate_df['Date'])
                climate_df.set_index('Date', inplace=True)
                climate_df = climate_df[['T2M', 'PRECTOTCORR']].interpolate().fillna(0)
        except Exception as e:
            st.sidebar.error(f"Error membaca file: {str(e)}")

# 4. Kontrol Simulasi Cuaca Manual (Conditional)
manual_temp, manual_rain, manual_duration = 25.0, 5.0, 180
if climate_type == "manual":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Kontrol Iklim Manual")
    manual_temp = st.sidebar.slider("Suhu Udara Rata-Rata (°C):", 15.0, 38.0, 25.0, 0.5)
    manual_rain = st.sidebar.slider("Curah Hujan Harian (mm):", 0, 50, 5, 1)
    manual_duration = st.sidebar.slider("Durasi Simulasi (Hari):", 30, 365, 180, 15)

# 5. Kontrol Intervensi Kebijakan (Larvasida & Fogging)
##st.sidebar.markdown("---")
##st.sidebar.subheader("🛡️ Intervensi Pengendalian Vektor")
##u1 = st.sidebar.slider("Efektivitas Larvasida (u1):", 0.0, 1.0, 0.0, 0.05, format="%.2f")
##u2 = st.sidebar.slider("Efektivitas Fogging Dewasa (u2):", 0.0, 1.0, 0.0, 0.05, format="%.2f")

# 6. Parameter Kustom Manusia & Genangan (Conditional)
custom_nh, custom_kmin, custom_theta, custom_calib = 25000, 8000.0, 300.0, 2.0
if city_type == "custom":
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Parameter Kustom SEIR-LM")
    custom_nh = st.sidebar.number_input("Total Populasi Manusia (Nh):", value=25000, step=1000)
    custom_kmin = st.sidebar.number_input("Daya Tampung Dasar Genangan (K_min):", value=8000.0, step=500.0)
    custom_theta = st.sidebar.number_input("Koefisien Efek Hujan (theta):", value=300.0, step=10.0)
    custom_calib = st.sidebar.number_input("Faktor Kalibrasi Transmisi (beta):", value=2.0, step=0.1)


# ==========================================
# 3. PERSIAPAN DATA & EKSEKUSI SIMULASI
# ==========================================

# A. Pemrosesan Data Iklim
if climate_type == "default":
    path = bandung_path if city_type == "bandung" else jakarta_path
    if city_type == "custom":
        path = bandung_path
    
    try:
        climate_df = pd.read_csv(path, skiprows=12)
        climate_df.replace(-999.0, np.nan, inplace=True)
        climate_df['Date'] = pd.to_datetime(climate_df['YEAR'] * 1000 + climate_df['DOY'], format='%Y%j')
        climate_df.set_index('Date', inplace=True)
        climate_df = climate_df[['T2M', 'PRECTOTCORR']].interpolate(method='linear').fillna(0)
    except FileNotFoundError:
        st.error(f"⚠️ File data iklim default tidak ditemukan di path:\n`{path}`\n\nPastikan path di kode sumber sudah sesuai dengan lokasi file Anda, atau gunakan opsi **Simulasi Cuaca Manual** / **Unggah CSV**.")
        st.stop()

elif climate_type == "manual" or (climate_type == "upload" and climate_df is None):
    # Buat data sintetis berdasarkan durasi manual (atau jika upload gagal/belum ada)
    dates = pd.date_range(start="2026-01-01", periods=manual_duration, freq="D")
    climate_df = pd.DataFrame({
        "T2M": [manual_temp] * manual_duration,
        "PRECTOTCORR": [manual_rain] * manual_duration
    }, index=dates)

# B. Menjalankan Simulasi
custom_params = {
    'Nh': custom_nh,
    'K_min': custom_kmin,
    'theta': custom_theta,
    'calibration_factor': custom_calib
}

with st.spinner('Menjalankan Simulasi SEIR-LM...'):
    res_df, Nh = run_simulation(
        city_type=city_type,
        climate_df=climate_df,
        custom_params=custom_params,
        #u1=u1,
        #u2=u2
    )

# C. Menghitung Nilai KPI
peak_infected_idx = res_df['Infected_Human'].idxmax()
peak_infected_val = res_df['Infected_Human'].max()
peak_day_num = (peak_infected_idx - res_df.index[0]).days

avg_r0 = res_df['R0'].mean()
risk_level = "Tinggi" if avg_r0 > 1.2 else ("Sedang" if avg_r0 >= 1.0 else "Rendah")

# Kematian spesifik akibat DBD
delta = 0.00315 if city_type == 'bandung' else (0.00058 if city_type == 'jakarta' else 0.001)
estimated_deaths = np.sum(res_df['Infected_Human'] * delta)
max_carrying_cap = res_df['K_R'].max()


# ==========================================
# 4. TAMPILAN DASHBOARD UTAMA (METRIK & GRAFIK)
# ==========================================

# A. Baris KPI (Key Performance Indicators)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Kasus Terinfeksi Puncak", f"{int(peak_infected_val)} Orang")
    st.caption(f"Hari ke-{peak_day_num} ({peak_infected_idx.strftime('%d %b %Y')})")

with col2:
    st.metric("Rerata Reproduksi R0(t)", f"{avg_r0:.2f}")
    st.caption(f"Status Risiko: **{risk_level}**")

with col3:
    st.metric("Estimasi Korban DBD", f"{int(estimated_deaths)} Jiwa")
    st.caption("Disease-Induced Deaths")

with col4:
    st.metric("Genangan Maksimum (K)", f"{int(max_carrying_cap)} Nyamuk")
    st.caption("Mosquito Carrying Capacity")

st.divider()

# B. Tabs Pilihan Grafik & Analisis
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Dinamika Host & Vektor", 
    "🦠 Analisis Laju Transmisi R0(t)", 
    "🌦️ Dinamika Cuaca (Input)", 
    "📑 Laporan Analisis & Rekomendasi"
])

# --- TAB 1: Dinamika Host & Vektor ---
with tab1:
    fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                         subplot_titles=("Dinamika Epidemiologi Tuan Rumah (Manusia)", 
                                         "Dinamika Populasi Vektor (Nyamuk Aedes aegypti)"))
    
    # Row 1: Manusia
    fig1.add_trace(go.Scatter(x=res_df.index, y=res_df['Exposed_Human'], name='Exposed (Eh)', line=dict(color='orange', width=2)), row=1, col=1)
    fig1.add_trace(go.Scatter(x=res_df.index, y=res_df['Infected_Human'], name='Infected (Ih)', line=dict(color='red', width=2.5)), row=1, col=1)
    fig1.add_trace(go.Scatter(x=res_df.index, y=res_df['Recovered_Human'], name='Recovered (Rh)', line=dict(color='green', width=2)), row=1, col=1)
    fig1.update_yaxes(title_text="Jumlah Manusia", row=1, col=1)
    
    # Row 2: Nyamuk
    fig1.add_trace(go.Scatter(x=res_df.index, y=res_df['Aquatic'], name='Akuatik / Larva (A)', line=dict(color='teal', width=1.5)), row=2, col=1)
    fig1.add_trace(go.Scatter(x=res_df.index, y=res_df['Susceptible_Vector'], name='Nyamuk Rentan (Sv)', line=dict(color='gray', width=1.5)), row=2, col=1)
    fig1.add_trace(go.Scatter(x=res_df.index, y=res_df['Infected_Vector'], name='Nyamuk Infektif (Iv)', line=dict(color='maroon', width=2)), row=2, col=1)
    fig1.update_yaxes(title_text="Jumlah Nyamuk", row=2, col=1)
    
    fig1.update_layout(height=600, legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1))
    st.plotly_chart(fig1, use_container_width=True)

# --- TAB 2: Analisis Laju Transmisi R0(t) ---
with tab2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=res_df.index, y=res_df['R0'], name='Instantaneous R0(t)', line=dict(color='#8a2be2', width=2.5)))
    fig2.add_shape(
        type="line", x0=res_df.index[0], y0=1.0, x1=res_df.index[-1], y1=1.0,
        line=dict(color="red", width=2, dash="dash")
    )
    fig2.update_layout(
        title="Dinamika Nilai R0(t) Terhadap Ambang Batas Epidemi (R0=1)",
        xaxis_title="Waktu", yaxis_title="R0(t)", height=500
    )
    st.plotly_chart(fig2, use_container_width=True)

# --- TAB 3: Dinamika Cuaca (Input) ---
with tab3:
    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                         subplot_titles=("Dinamika Temperatur Udara (T2M)", "Curah Hujan & Kapasitas Lingkungan (K)"))
    
    fig3.add_trace(go.Scatter(x=res_df.index, y=res_df['T2M'], name='Temperatur (°C)', line=dict(color='#ff7f0e', width=2)), row=1, col=1)
    fig3.update_yaxes(title_text="Temperatur (°C)", row=1, col=1)
    
    fig3.add_trace(go.Bar(x=res_df.index, y=res_df['PRECTOTCORR'], name='Curah Hujan (mm)', marker_color="#3691d2", opacity=0.6), row=2, col=1)
    fig3.add_trace(go.Scatter(x=res_df.index, y=res_df['K_R'], name='Daya Tampung K(R)', line=dict(color='#2ca02c', width=1.5)), row=2, col=1)
    fig3.update_yaxes(title_text="Skala Curah Hujan / Kapasitas", row=2, col=1)
    
    fig3.update_layout(height=600, legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1))
    st.plotly_chart(fig3, use_container_width=True)

# --- TAB 4: Laporan Analisis & Rekomendasi ---
with tab4:
    # Perhitungan Waktu Terbaik
    status_bahaya = "WASPADA EKSTREM" if avg_r0 > 1.5 else ("SIAGA" if avg_r0 >= 1.0 else "AMAN / TERKENDALIKAN")
    risk_color = "red" if status_bahaya == "WASPADA EKSTREM" else ("orange" if status_bahaya == "SIAGA" else "green")
    
    aquatic_diff = np.diff(res_df['Aquatic'])
    best_u1_idx = int(np.argmax(aquatic_diff)) if len(aquatic_diff) > 0 else 0
    best_u1_date = res_df.index[best_u1_idx].strftime('%d %B %Y')
    
    best_u2_idx = max(0, int((res_df['Infected_Vector'].idxmax() - pd.Timedelta(days=7) - res_df.index[0]).days))
    best_u2_date = res_df.index[best_u2_idx].strftime('%d %B %Y')
    
    # Render Laporan
    st.subheader("Laporan Analisis Epidemiologi Model SEIR-LM")
    st.markdown(f"**Status Kesehatan Wilayah:** <span style='color:{risk_color}; font-size:1.1em; font-weight:bold;'>{status_bahaya}</span>", unsafe_allow_html=True)
    st.divider()
    
    st.markdown("#### 🔍 Temuan Utama Simulasi:")
    kalimat_r0 = "Wabah berpotensi menyebar dan menetap secara endemik karena R0 > 1." if avg_r0 >= 1.0 else "Wabah akan mereda secara alami karena rata-rata R0 < 1."
    st.markdown(f"""
    * Rata-rata angka reproduksi R0(t) adalah **{avg_r0:.2f}**. {kalimat_r0}
    * Kasus infeksi manusia maksimum terjadi pada hari ke-{peak_day_num} ({peak_infected_idx.strftime('%d %B %Y')}) sebanyak **{int(peak_infected_val)} orang**.
    * Kapasitas lingkungan maksimum mencapai **{int(max_carrying_cap)} genangan penampungan nyamuk** akibat pengaruh curah hujan.
    * Total estimasi kematian spesifik DBD selama periode simulasi adalah **{int(estimated_deaths)} jiwa**.
    """)
    
    st.markdown("#### 🛡️ Panduan Waktu Intervensi Vektor (Rekomendasi Kebijakan):")
    st.markdown(f"""
    * **Aplikasi Larvasida (u1):** Rekomendasi terbaik dilakukan sebelum atau pada tanggal **{best_u1_date}** (hari ke-{best_u1_idx}) ketika laju pertumbuhan populasi larva sedang berada pada tingkat tertinggi.
    * **Penyemprotan / Fogging Nyamuk Dewasa (u2):** Paling tepat diimplementasikan pada atau sebelum tanggal **{best_u2_date}** (hari ke-{best_u2_idx}), yaitu 7 hari sebelum puncak populasi nyamuk dewasa pembawa virus (Iv).
    """)
    
    st.info("""
    **Catatan Biofisis:**
    Model ini dipengaruhi secara mekanistik oleh iklim. Suhu rata-rata di atas 27.5°C mempercepat replikasi virus (Briere), sedangkan akumulasi curah hujan 14 hari sebelumnya memicu pembentukan genangan air K(R) yang memperluas sarang nyamuk.
    """)

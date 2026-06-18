import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.interpolate import interp1d

# ==========================================
# 1. PARAMETER BIOFISIK DAN FUNGSI INTERPOLASI
# ==========================================

# Konstanta Briere & Mortalitas Nyamuk
c_briere = 0.000202
T_min_b = 13.35    # Suhu minimal nyamuk aktif (Celcius)
T_max_b = 40.08    # Suhu maksimal nyamuk mati/tidak aktif (Celcius)
p_hv = 0.75        # Probabilitas keberhasilan transfer virus per gigitan
p_vh = 0.75        # Probabilitas transfer nyamuk -> manusia

def briere_function(T, c, T_min, T_max):
    """
    Menghitung komponen laju gigitan/aktivitas berdasarkan fungsi Briere.
    Jika suhu di luar batas biologis (T_min s/d T_max), aktivitas dihentikan (0).
    Dapat menerima input berupa scalar maupun NumPy array.
    """
    return np.where((T > T_min) & (T < T_max), 
                    c * T * (T - T_min) * np.sqrt(T_max - T), 
                    0.0)

def mortality_function(T):
    """
    Fungsi mortalitas nyamuk dewasa (Mu_v) berbasis suhu (Polinomial).
    Mortalitas terendah di suhu ideal 27.5 C, naik di suhu dingin/panas ekstrem.
    """
    T_opt = 27.5
    mu_min = 0.05
    a = 0.002
    mu = a * (T - T_opt)**2 + mu_min
    return np.minimum(mu, 1.0)

def hitung_beta_hv(T, calibration_factor):
    """Menghitung Laju Penularan Manusia ke Nyamuk (Beta_hv)."""
    b_T = briere_function(T, c_briere, T_min_b, T_max_b)
    return b_T * p_hv * calibration_factor

def hitung_beta_vh(T, calibration_factor):
    """Menghitung Laju Penularan Nyamuk ke Manusia (Beta_vh)."""
    b_T = briere_function(T, c_briere, T_min_b, T_max_b)
    return b_T * p_vh * calibration_factor

# ==========================================
# 2. PERSAMAAN DIFERENSIAL SISTEM SEIR-LM
# ==========================================

def seirlm_model(y, t, phi, sigma, mu_A, Lambda_h, mu_h, alpha, gamma, delta, omega, Nh, 
                  K_func, Beta_hv_func, Beta_vh_func, MU_v_func, u1=0.0, u2=0.0):
    """
    Sistem Persamaan Diferensial Biasa (ODE) SEIR-LM untuk DBD.
    y = [A, Sv, Iv, Sh, Eh, Ih, Rh]
    u1 = Kontrol Larvasida (0.0 s/d 1.0)
    u2 = Kontrol Fogging Nyamuk Dewasa (0.0 s/d 1.0)
    """
    A, Sv, Iv, Sh, Eh, Ih, Rh = y
    
    # Evaluasi parameter biofisis dinamis bergantung waktu t
    K_R_t = max(K_func(t), 1.0)
    beta_hv_t = Beta_hv_func(t)
    beta_vh_t = Beta_vh_func(t)
    mu_v_t = MU_v_func(t)
    
    # --- SUB-MODEL VEKTOR NYAMUK (LM) ---
    # Dinamika Fase Akuatik (A)
    dA_dt = phi * (1.0 - A / K_R_t) * (Sv + Iv) - (sigma + mu_A + u1) * A
    
    # Dinamika Nyamuk Dewasa Rentan (Sv)
    dSv_dt = sigma * A - beta_hv_t * Sv * (Ih / Nh) - (mu_v_t + u2) * Sv
    
    # Dinamika Nyamuk Dewasa Terinfeksi (Iv)
    dIv_dt = beta_hv_t * Sv * (Ih / Nh) - (mu_v_t + u2) * Iv
    
    # --- SUB-MODEL MANUSIA (SEIR) ---
    # Dinamika Manusia Rentan (Sh)
    dSh_dt = Lambda_h - beta_vh_t * Sh * (Iv / Nh) - mu_h * Sh + omega * Rh
    
    # Dinamika Manusia Terpapar/Masa Inkubasi (Eh)
    dEh_dt = beta_vh_t * Sh * (Iv / Nh) - (alpha + mu_h) * Eh
    
    # Dinamika Manusia Terinfeksi (Ih)
    dIh_dt = alpha * Eh - (gamma + mu_h + delta) * Ih
    
    # Dinamika Manusia Sembuh (Rh)
    dRh_dt = gamma * Ih - (mu_h + omega) * Rh
    
    return [dA_dt, dSv_dt, dIv_dt, dSh_dt, dEh_dt, dIh_dt, dRh_dt]

# ==========================================
# 3. KORELASI DAN ANALISIS LAJU TRANMISI R0
# ==========================================

def calc_R0_t(t_array, Sv_array, Nh, gamma, alpha, mu_h, delta, Beta_vh_func, Beta_hv_func, MU_v_func):
    """
    Menghitung angka reproduksi dasar instan R0(t) menggunakan pendekatan Next Generation Matrix.
    """
    beta_vh_t = Beta_vh_func(t_array)
    beta_hv_t = Beta_hv_func(t_array)
    mu_v_t = np.clip(MU_v_func(t_array), 0.05, 1.0)  # Batasi laju kematian nyamuk min 5% agar tidak pembagian nol
    
    # Formula R0^2 dari NGM
    R0_squared = (alpha / (alpha + mu_h)) * (beta_vh_t / mu_v_t + 1e-9) * ((beta_vh_t * Sv_array) / (Nh * (gamma + mu_h + delta)))
    
    return np.sqrt(np.maximum(R0_squared, 0.0))

# ==========================================
# 4. EKSEKUSI SIMULASI UTAMA (Bandung & Jakarta)
# ==========================================

def run_simulation(city_type, climate_df, custom_params=None, initial_state=None, u1=0.0, u2=0.0):
    """
    Mengambil data iklim dan menjalankan simulasi SEIR-LM.
    city_type: 'bandung', 'jakarta', atau 'custom'
    climate_df: pandas DataFrame dengan kolom Date, T2M, dan PRECTOTCORR
    custom_params: dict parameter jika menggunakan kota kustom
    """
    df = climate_df.copy()
    
    # Preprocessing akumulasi curah hujan 14 hari rolling sum (dengan batas atas 200 mm)
    if 'PRECTOTCORR_ROLL14' not in df.columns:
        df['PRECTOTCORR_ROLL14'] = df['PRECTOTCORR'].clip(upper=200).rolling(window=14).sum().shift(1)
        df.dropna(subset=['PRECTOTCORR_ROLL14'], inplace=True)
    
    # Mengatur parameter sesuai kota pilihan
    if city_type == 'bandung':
        Nh = 21847
        mu_h = 1.0 / (74.41 * 365.0)
        Lambda_h = (0.01416 * Nh) / 365.0
        alpha = 1.0 / 5.0
        gamma = 1.0 / 3.0
        delta = 0.00315
        omega = 1.0 / 120.0
        K_min = 10000.0
        theta = 500.0
        calibration_factor = 3.0
    elif city_type == 'jakarta':
        Nh = 28311
        mu_h = 1.0 / (74.18 * 365.0)
        Lambda_h = (0.01175 * Nh) / 365.0
        alpha = 1.0 / 5.0
        gamma = 1.0 / 3.0
        delta = 0.00058
        omega = 1.0 / 120.0
        K_min = 7000.0
        theta = 250.0
        calibration_factor = 1.24
    else: # Custom
        Nh = custom_params.get('Nh', 25000)
        mu_h = 1.0 / (custom_params.get('life_expectancy', 73.0) * 365.0)
        Lambda_h = (custom_params.get('birth_rate', 0.012) * Nh) / 365.0
        alpha = 1.0 / custom_params.get('incubation_days', 5.0)
        gamma = 1.0 / custom_params.get('recovery_days', 3.0)
        delta = custom_params.get('dengue_mortality', 0.001)
        omega = 1.0 / custom_params.get('immunity_months', 4.0) * 30.0
        K_min = custom_params.get('K_min', 8000.0)
        theta = custom_params.get('theta', 300.0)
        calibration_factor = custom_params.get('calibration_factor', 2.0)
        
    phi = 3.0
    sigma = 0.08
    mu_A = 0.15
    
    params = (phi, sigma, mu_A, Lambda_h, mu_h, alpha, gamma, delta, omega, Nh)
    
    # 1. Hitung parameter dinamis harian
    df['BETA_hv'] = df['T2M'].apply(lambda x: hitung_beta_hv(x, calibration_factor))
    df['BETA_vh'] = df['T2M'].apply(lambda x: hitung_beta_vh(x, calibration_factor))
    df['MU_v'] = df['T2M'].apply(mortality_function)
    df['K_R'] = K_min + (theta * df['PRECTOTCORR_ROLL14'])
    
    # 2. Definisikan fungsi interpolator waktu
    t_days = np.arange(len(df))
    K_func = interp1d(t_days, df['K_R'], kind='linear', fill_value='extrapolate')
    Beta_hv_func = interp1d(t_days, df['BETA_hv'], kind='linear', fill_value='extrapolate')
    Beta_vh_func = interp1d(t_days, df['BETA_vh'], kind='linear', fill_value='extrapolate')
    MU_v_func = interp1d(t_days, df['MU_v'], kind='linear', fill_value='extrapolate')
    
    # 3. Kondisi Awal
    if initial_state is None:
        Ih_0 = 100
        Eh_0 = 250
        Rh_0 = 0
        Sh_0 = Nh - Ih_0 - Eh_0 - Rh_0
        A_0 = max(K_func(0), 1.0)
        Sv_0 = A_0 * (sigma / max(MU_v_func(0), 0.01))
        Iv_0 = 3000
        y0 = [A_0, Sv_0, Iv_0, Sh_0, Eh_0, Ih_0, Rh_0]
    else:
        y0 = initial_state
        
    # 4. Selesaikan ODE
    solution = odeint(
        seirlm_model, y0, t_days, 
        args=(phi, sigma, mu_A, Lambda_h, mu_h, alpha, gamma, delta, omega, Nh, 
              K_func, Beta_hv_func, Beta_vh_func, MU_v_func, u1, u2)
    )
    
    # 5. Ekstrak Hasil
    A_out  = solution[:, 0]
    Sv_out = solution[:, 1]
    Iv_out = solution[:, 2]
    Sh_out = solution[:, 3]
    Eh_out = solution[:, 4]
    Ih_out = solution[:, 5]
    Rh_out = solution[:, 6]
    
    # 6. Hitung R0(t) harian
    R0_array = calc_R0_t(t_days, Sv_out, Nh, gamma, alpha, mu_h, delta, Beta_vh_func, Beta_hv_func, MU_v_func)
    
    # 7. Konstruksi Output DataFrame
    res_df = pd.DataFrame({
        'Date': df.index,
        'Aquatic': A_out,
        'Susceptible_Vector': Sv_out,
        'Infected_Vector': Iv_out,
        'Susceptible_Human': Sh_out,
        'Exposed_Human': Eh_out,
        'Infected_Human': Ih_out,
        'Recovered_Human': Rh_out,
        'R0': R0_array,
        'T2M': df['T2M'],
        'PRECTOTCORR': df['PRECTOTCORR'],
        'K_R': df['K_R']
    })
    res_df.set_index('Date', inplace=True)
    
    return res_df, Nh

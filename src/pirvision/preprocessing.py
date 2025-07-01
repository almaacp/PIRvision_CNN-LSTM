from sklearn.preprocessing import StandardScaler
import numpy as np
import pywt
from sklearn.metrics import mean_squared_error

def handle_missing_values(df):
    return df.dropna()  # Menghapus baris yang mengandung nilai NaN

def detect_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR  # Batas bawah
    upper = Q3 + 1.5 * IQR  # Batas atas
    return df[(df[col] >= lower) & (df[col] <= upper)]  # Menghapus baris yang mengandung outlier

def denoise_signal(signal_2d, wavelet='db4', level=1):
    denoised = []
    for sample in signal_2d:
        coeffs = pywt.wavedec(sample, wavelet, level=level) # Menghitung koefisien wavelet
        cA = coeffs[0]  # Ambil koefisien approximasi (low frequency)
        cDs = coeffs[1:]    # Ambil koefisien detail (high frequency)
        # Hitung threshold berdasarkan MAD (median absolute deviation)
        denoised_coeffs = [cA]  # Simpan koefisien approximasi tanpa perubahan
        for cD in cDs:
            sigma = np.median(np.abs(cD)) / 0.6745  # MAD untuk estimasi deviasi standar
            threshold = sigma * np.sqrt(2 * np.log(len(cD)))    # Thresholding menggunakan soft thresholding
            denoised_cD = pywt.threshold(cD, threshold, mode='soft')    # Terapkan soft thresholding
            denoised_coeffs.append(denoised_cD) # Simpan koefisien detail yang telah di-denoise
        cleaned = pywt.waverec(denoised_coeffs, wavelet)    # Rekonstruksi sinyal dari koefisien yang telah di-denoise
        denoised.append(cleaned[:len(sample)])  # Pastikan panjang sinyal tetap sama dengan aslinya
    return np.array(denoised)

# NRMSE mengukur seberapa besar kesalahan prediksi relatif terhadap rentang nilai sebenarnya
def normalized_root_mse(true, pred):
    return np.sqrt(mean_squared_error(true, pred)) / (np.max(true) - np.min(true))

def normalize_zscore(df):
    scaler = StandardScaler()   # Inisialisasi scaler untuk normalisasi Z-score
    df[df.columns] = scaler.fit_transform(df[df.columns])   # Normalisasi menggunakan Z-score
    return df
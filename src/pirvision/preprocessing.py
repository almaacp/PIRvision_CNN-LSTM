from sklearn.preprocessing import StandardScaler
import numpy as np
import pywt
from sklearn.metrics import mean_squared_error

def handle_missing_values(df):
    return df.dropna()

def detect_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

def denoise_signal(signal_2d, wavelet='db4', level=1):
    denoised = []
    for sample in signal_2d:
        coeffs = pywt.wavedec(sample, wavelet, level=level)
        cA = coeffs[0]  # Approximation coeffs (low frequency) tetap
        cDs = coeffs[1:]    # Ambil semua detail coeffs (high frequency)

        # Hitung threshold berdasarkan MAD (median absolute deviation)
        denoised_coeffs = [cA]
        for cD in cDs:
            sigma = np.median(np.abs(cD)) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(cD)))
            denoised_cD = pywt.threshold(cD, threshold, mode='soft')
            denoised_coeffs.append(denoised_cD)

        cleaned = pywt.waverec(denoised_coeffs, wavelet)
        denoised.append(cleaned[:len(sample)])  # truncate
    return np.array(denoised)

def normalized_root_mse(true, pred):
    return np.sqrt(mean_squared_error(true, pred)) / (np.max(true) - np.min(true))

def normalize_zscore(df):
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    return df
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_segment_data(paths):
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"[!] File not found: {p}") # Pastikan file tersebut ada sebelum memuat
    df_list = [pd.read_csv(p) for p in paths]   # Muat semua file CSV ke dalam daftar DataFrame
    return pd.concat(df_list, ignore_index=True)    # Gabungkan semua DataFrame ke dalam satu DataFrame

def stratified_split(X, y, test_size, val_size, seed):
    # Split testing set terlebih dahulu
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    # Kemudian split sisa data menjadi training dan validation set
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size / (1 - test_size), stratify=y_temp, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test

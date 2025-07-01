# 🔎 PIRvision Activity Classification

> Klasifikasi aktivitas dan keberadaan manusia berdasarkan sinyal sensor PIR menggunakan **deep learning** dan **balancing data**.
> Mengintegrasikan model **CNN-LSTM** sebagai pendekatan spasial-temporal untuk meningkatkan akurasi klasifikasi pada data time-series yang tidak seimbang.

---

## ❓ Latar Belakang

Penelitian ini menangani dua tantangan utama pada dataset **PIRvision**:

1. **Ketidakseimbangan kelas**, yang menyulitkan model dalam mengenali kelas minoritas.
2. **Kebutuhan analisis spasial dan temporal** dari sinyal sensor PIR, yang belum banyak dimanfaatkan secara bersamaan di penelitian sebelumnya.

Pendekatan yang digunakan:

* **Model hybrid CNN-LSTM** untuk menangkap pola spasial antar sensor dan pola temporal antar waktu.
* Evaluasi metode **balancing data** seperti **SMOTE** dan **Random Undersampling** pada data sensor time-series.

---

## 🔧 Instalasi

### 1. Clone Repositori

```bash
git clone https://github.com/username/pirvision-cnn-lstm.git
cd pirvision-cnn-lstm
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Package Secara Lokal

```bash
pip install -e .
```

---

## 📦 Import Modul Dimanapun

Setelah install lokal, maka dapat menggunakan package seperti:

```python
from pirvision.config import *
from pirvision.dataloader import load_and_segment_data, stratified_split
from pirvision.preprocessing import handle_missing_values, detect_outliers_iqr, denoise_signal, normalized_root_mse, normalize_zscore
from pirvision.balancing import apply_smote, apply_rus
from pirvision.models.cnn_lstm import build_cnn_lstm
from pirvision.models.lstm import build_lstm
from pirvision.models.knn import train_knn_with_validation
from pirvision.trainer import train_model
from pirvision.evaluator import evaluate_classification
from pirvision.utils import plot_confusion_matrix, plot_class_distribution
```
Bisa dipakai di:

* Python script (`.py`)
* Notebook
* VS Code Interactive Window

---

## 🚀 Menjalankan Pipeline dengan Mode Interaktif (GUI)

1. Buka file `src/pirvision/main.py` di **VS Code Interactive Window** (`#%%`) atau **Jupyter Notebook**
2. Jalankan semua sel
3. Pilih:
   * Model: `CNN-LSTM`, `LSTM`, `KNN`
   * Balancing: `No Balancing`, `SMOTE`, `RUS`
4. Klik `Run Model`

📊 Akan ditampilkan:
* Metrik evaluasi (Accuracy, Precision, Recall, F1)
* Confusion Matrix (heatmap)

---

## 🏗️ Struktur Pipeline

* Dataset digabung dari 2 file `.csv`
* Tahapan preprocessing:
  * Null handling
  * Outlier removal (IQR)
  * Segmentasi time-series
  * Denoising (wavelet)
  * Normalisasi (Z-score)
* Training + validation + testing split
* Balancing (SMOTE / RUS / None)
* Model: CNN-LSTM, LSTM, KNN
* Evaluasi akhir pada testing set

---

## 🏷️ Label Kelas

| Label | Deskripsi                 |
| ----- | ------------------------- |
| `0`   | Vacancy                   |
| `1`   | Stationary human presence |
| `2`   | Other activity/motion     |

> Label `3` yang ada di csv diubah ke `2` agar hanya terdapat 3 kelas klasifikasi utama.

---

## 🧪 Unit Testing

Menjalankan semua unit test:

```bash
pytest tests/
```

---

## ✨ Fitur Tambahan

* Informasi `Date` & `Time` tetap tersedia untuk analisis temporal
* Visualisasi distribusi kelas dan sinyal PIR
* Pipeline modular dan mudah diperluas
* Tidak menyimpan gambar ke disk, hanya menampilkan di runtime

---

## 📬 Kontak

**Alma Alya Cipta Putri**
📧 \[[almaalyaciptaputri@gmail.com](mailto:almaalyaciptaputri@gmail.com)]

---
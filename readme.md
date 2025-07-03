# Development of Hybrid CNN-LSTM Model on Imbalanced Data for Human Presence and Activity Detection Based on PIRvision Data

> Classification of human presence and activity based on PIR sensor signals using **deep learning** and **balanced data**. Integrating **CNN-LSTM** model as a spatial-temporal approach to improve classification accuracy on imbalanced time-series sensor data.

## ❓ Background

This research addresses two main challenges on the **PIRvision** dataset:
1. **Class imbalance**, which makes it difficult for models to recognize minority classes.
2. **The need for spatial and temporal analysis** of PIR sensor signals, which has not been utilized simultaneously in previous research.

Approach used:
* **Hybrid CNN-LSTM model** to capture spatial patterns between sensors and temporal patterns between times.
* Evaluation of **data balancing** methods such as **SMOTE** and **Random Undersampling** on time-series sensor data.

## 🔧 Installation

### 1. Clone Repository and Login to Repository

```bash
git clone https://github.com/almaacp/PIRvision_CNN-LSTM.git
cd PIRvision_CNN-LSTM
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Package Locally

```bash
pip install -e .
```

After local install, then can use the package:

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
Which can be accessed at:
* Python script (`.py`)
* Notebook
* VS Code Interactive Window

## 🚀 Running Pipeline with Interactive Mode (GUI)

1. Open the file `src/pirvision/main.py` in the **VS Code Interactive Window** (`#%%`) or **Jupyter Notebook**
2. Run all cells
3. Select:
   * Model: `CNN-LSTM`, `LSTM`, `KNN`
   * Balancing: `No Balancing`, `SMOTE`, `RUS`
4. Click `Run Model`

After running the model, it will display:
* Evaluation metrics (Accuracy, Precision, Recall, F1-Score)
* Confusion Matrix (heatmap)

## 🏗️ Pipeline Structure

* Dataset merged from 2 `.csv` files
* Preprocessing stages:
  * Null handling
  * Outlier removal (IQR)
  * Time-series segmentation
  * Denoising (wavelet)
  * Normalization (Z-score)
* Training + validation + testing split
* Balancing (SMOTE/RUS/None)
* Model: CNN-LSTM, LSTM, KNN
* Final evaluation on testing set

## 🏷️ Class Labels

| Label | Description |
| ----- | ------------------------- |
| `0` | Vacancy |
| `1` | Stationary human presence |
| `3` | Other activity/motion |

## ✨ Additional Features

* Fixed `Date` & `Time` information available for temporal analysis
* Visualization of class distribution and PIR signals
* Modular and easily extensible pipeline
* Does not save images to disk, only displays at runtime

Thank you for trying out this research! If you have any questions and feedback, please contact me at the contacts below:

📧 [almaalyaciptaputri@gmail.com](mailto:almaalyaciptaputri@gmail.com)

---

[Translate Bahasa Indonesia]

# Pengembangan Model Hybrid CNN-LSTM pada Imbalanced Data untuk Deteksi Keberadaan dan Aktivitas Manusia Berdasarkan Data PIRvision

> Klasifikasi aktivitas dan keberadaan manusia berdasarkan sinyal sensor PIR menggunakan **deep learning** dan **balancing data**. Mengintegrasikan model **CNN-LSTM** sebagai pendekatan spasial-temporal untuk meningkatkan akurasi klasifikasi pada data sensor time-series yang tidak seimbang.

## ❓ Latar Belakang

Penelitian ini menangani dua tantangan utama pada dataset **PIRvision**:
1. **Ketidakseimbangan kelas**, yang menyulitkan model dalam mengenali kelas minoritas.
2. **Kebutuhan analisis spasial dan temporal** dari sinyal sensor PIR, yang belum banyak dimanfaatkan secara bersamaan di penelitian sebelumnya.

Pendekatan yang digunakan:
* **Model hybrid CNN-LSTM** untuk menangkap pola spasial antar sensor dan pola temporal antar waktu.
* Evaluasi metode **balancing data** seperti **SMOTE** dan **Random Undersampling** pada data sensor time-series.

## 🔧 Instalasi

### 1. Clone Repositori dan Masuk ke Repositori

```bash
git clone https://github.com/almaacp/PIRvision_CNN-LSTM.git
cd PIRvision_CNN-LSTM
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Package Secara Lokal

```bash
pip install -e .
```

Setelah install lokal, maka dapat menggunakan package:

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
Yang dapat diakses di:
* Python script (`.py`)
* Notebook
* VS Code Interactive Window

## 🚀 Menjalankan Pipeline dengan Mode Interaktif (GUI)

1. Buka file `src/pirvision/main.py` di **VS Code Interactive Window** (`#%%`) atau **Jupyter Notebook**
2. Jalankan semua sel
3. Pilih:
   * Model: `CNN-LSTM`, `LSTM`, `KNN`
   * Balancing: `No Balancing`, `SMOTE`, `RUS`
4. Klik `Run Model`

Setelah menjalankan model, maka akan ditampilkan:
* Metrik evaluasi (Accuracy, Precision, Recall, F1-Score)
* Confusion Matrix (heatmap)

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

## 🏷️ Label Kelas

| Label | Deskripsi                 |
| ----- | ------------------------- |
| `0`   | Vacancy                   |
| `1`   | Stationary human presence |
| `3`   | Other activity/motion     |

## ✨ Fitur Tambahan

* Informasi `Date` & `Time` tetap tersedia untuk analisis temporal
* Visualisasi distribusi kelas dan sinyal PIR
* Pipeline modular dan mudah diperluas
* Tidak menyimpan gambar ke disk, hanya menampilkan di runtime

Terima kasih sudah mencoba penelitian ini! Apabila ada pertanyaan dan masukan, hubungi saya pada kontak di bawah ini:

📧 [almaalyaciptaputri@gmail.com](mailto:almaalyaciptaputri@gmail.com)

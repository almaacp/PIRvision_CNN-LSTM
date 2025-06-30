# PIRvision Activity Classification

Klasifikasi keberadaan dan aktivitas manusia berdasarkan sinyal sensor PIR dengan pipeline preprocessing, balancing data, dan pemodelan deep learning (CNN-LSTM).

---

## 🔧 Instalasi

1. Clone atau download repositori:
```bash
cd project_root  # masuk ke folder project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install package `pirvision` secara lokal:
```bash
pip install -e .
```

> ✅ Ini akan membuat kamu bisa mengimpor dari `pirvision` di Jupyter/VSCode tanpa error.

---

## 📂 Struktur Proyek

```
project_root/
├── data/                         # Dataset CSV
│   ├── pirvision_office_dataset1.csv
│   └── pirvision_office_dataset2.csv
├── src/
│   └── pirvision/               # Modular Python package
│       ├── config.py           # Konfigurasi global
│       ├── dataloader.py       # Load, segmentasi, split
│       ├── preprocessing.py    # Null, outlier, z-score
│       ├── balancing.py        # SMOTE, RUS
│       ├── utils.py            # Plotting
│       ├── evaluator.py        # Metrik
│       ├── trainer.py          # Keras model training
│       ├── main.py             # Pipeline interaktif
│       └── models/             # Model klasifikasi
│           ├── cnn_lstm.py
│           ├── lstm.py
│           └── knn.py
├── tests/                      # Unit tests
├── requirements.txt            # Dependensi
├── setup.py                    # Packaging
└── README.md
```

---

## 🚀 Cara Menjalankan Pipeline

### 🎛️ Mode Interaktif via GUI
1. Buka file `src/pirvision/main.py` dengan **VS Code Interactive Window** (`#%%`) atau **Jupyter Notebook**
2. Jalankan semua sel hingga muncul dropdown
3. Pilih:
   - Model: `CNN-LSTM`, `LSTM`, atau `KNN`
   - Balancing: `None`, `SMOTE`, `RUS`
4. Klik `Run Model`

✅ Akan muncul hasil akurasi dan confusion matrix.

---

## 📦 Import Modul di Mana Saja
Setelah `pip install -e .`, kamu bisa:
```python
from pirvision.models.lstm import build_lstm
from pirvision.trainer import train_model
```
Bisa dipakai di:
- file Python biasa (`run_model.py`)
- Jupyter Notebook
- Interactive Console

---

## 🧪 Testing
Untuk menjalankan semua unit test:
```bash
pytest tests/
```

---

## 🔍 Label Kelas
- `0` = Vacancy
- `1` = Stationary human presence
- `2` = Other activity/motion (awalnya label `3`, di-map ke `2`)

---

## ✨ Catatan Tambahan
- Dataset akan digabung otomatis dari dua file `.csv`
- Label tidak perlu di-encode ulang
- Semua visualisasi langsung tampil (tidak disimpan ke disk)
- Distribusi kelas divisualisasikan sebelum training

---

## 📬 Kontak
Jika ada pertanyaan atau masukan:
**Nama Kamu** - [email@example.com]
#%% Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
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

#%% Load raw data dan analisis missing values
df = load_and_segment_data(DATA_PATHS)  # Muat DataFrame dari beberapa file CSV
display(df.head(10))  # Tampilkan beberapa baris pertama DataFrame

#%% Analisis missing values
print(f"Dataset shape before missing values handling: {df.shape}") # Tampilkan bentuk awal DataFrame
null_count = df.isnull().sum().sum()    # Hitung jumlah nilai null di seluruh DataFrame
print(f">> Total missing values: {null_count}")    # Tampilkan jumlah nilai null
# Jika ada nilai null, tangani dengan fungsi handle_missing_values
if null_count > 0:
    df = handle_missing_values(df)
    print(f"Dataset shape after missing values handling: {df.shape}")
# Jika tidak ada nilai null, tampilkan pesan tidak ada nilai null
else:
    print("Tidak ada missing values dalam dataset.")

#%% Pisahkan label, waktu, dan fitur numerik
# Ganti label 3 menjadi 2 untuk konsistensi
labels = df["Label"].replace({3: 2}).values
# Kolom non-numerik yang akan dipisahkan
non_numeric_cols = ["Date", "Time"]
# Cek apakah kolom non-numerik ada
non_numeric_df = df[non_numeric_cols] if all(col in df.columns for col in non_numeric_cols) else None
# Ambil kolom numerik yang dimulai dengan "PIR_" atau "Temperature_F"
numerical_cols = [c for c in df.columns if c.startswith("PIR_") or c == "Temperature_F"]
# DataFrame hanya dengan fitur numerik
features_df = df[numerical_cols]

#%% Outlier analysis
# Cek distribusi fitur numerik sebelum menghapus outlier
plt.figure()
sns.boxplot(data=features_df, x='Temperature_F')
plt.title("Before Outlier Removal")
plt.tight_layout()
plt.show()

# Hapus outlier menggunakan metode IQR untuk fitur 'Temperature_F'
features_df = detect_outliers_iqr(features_df, 'Temperature_F')

# Cek distribusi fitur numerik setelah menghapus outlier
plt.figure()
sns.boxplot(data=features_df, x='Temperature_F')
plt.title("After Outlier Removal (IQR)")
plt.tight_layout()
plt.show()

#%% Segmentasi
segment_size = WINDOW_SIZE  # Ukuran segmen sesuai dengan WINDOW_SIZE
total_segments = len(features_df) // segment_size   # Total segmen yang dapat dibuat
# Bentuk data menjadi 3D array (jumlah segmen, ukuran segmen, jumlah fitur)
X_raw = features_df.values[:total_segments * segment_size].reshape((total_segments, segment_size, -1))
y = labels[:total_segments * segment_size].reshape(-1, segment_size)[:, 0]  # Ambil label pertama dari setiap segmen
# Simpan waktu awal tiap segmen
if non_numeric_df is not None:
    time_data = non_numeric_df.iloc[:total_segments * segment_size:segment_size].reset_index(drop=True)

#%% Denoising
X_denoised = X_raw.copy()   # Salin data mentah ke array baru untuk denoising
for i, col in enumerate(numerical_cols):
    if col.startswith("PIR_"):  # Hanya lakukan denoising untuk fitur PIR
        X_denoised[:, :, i] = denoise_signal(X_raw[:, :, i])
    else:   # Biarkan fitur selain PIR tetap
        X_denoised[:, :, i] = X_raw[:, :, i]

# Normalized Root Mean Squared Error (NRMSE) untuk fitur PIR
for feature_id, col in enumerate(numerical_cols):
    if col.startswith("PIR_"):  # Hanya hitung NRMSE untuk fitur PIR
        nrmse = normalized_root_mse(X_raw[:, :, feature_id].flatten(), X_denoised[:, :, feature_id].flatten())
        print(f"NRMSE (feature {col}): {nrmse:.4f}")
    else:
        continue    # Lewatkan fitur selain PIR

# Widget dropdown
pir_features = [(i, col) for i, col in enumerate(numerical_cols) if col.startswith("PIR_")]
feature_dropdown = widgets.Dropdown(
    options=[(name, idx) for idx, name in pir_features],
    description='PIR Feature:',
    layout=widgets.Layout(width='300px')
)

# Output area
plot_out = widgets.Output()

# Fungsi update plot
sample_id = 1
def update_plot(change):
    with plot_out:
        clear_output(wait=True)
        feature_id = change['new']
        feature_name = numerical_cols[feature_id]
        
        plt.figure(figsize=(8, 4))
        plt.plot(X_raw[sample_id, :, feature_id], label="Original", linestyle='--')
        plt.plot(X_denoised[sample_id, :, feature_id], label="Denoised", alpha=0.8)
        plt.title(f"Before vs After Denoising for {feature_name}")
        plt.xlabel("Time Step")
        plt.ylabel("Sensor Value")
        plt.legend()
        plt.tight_layout()
        plt.show()

# Trigger saat pertama kali & saat dropdown berubah
feature_dropdown.observe(update_plot, names='value')
update_plot({'new': feature_dropdown.value})

# Tampilkan
display(widgets.VBox([
    widgets.HTML("<div style='text-align:center'><h3>Denoising Viewer for PIR Features</h3></div>"),
    feature_dropdown,
    plot_out
]))

# Visualisasi sebelum dan sesudah denoising
# sample_id, feature_id = 1, 1
# plt.figure(figsize=(8, 4))
# plt.plot(X_raw[sample_id, :, feature_id], label="Original", linestyle='--')
# plt.plot(X_denoised[sample_id, :, feature_id], label="Denoised", alpha=0.8)
# plt.title(f"Before vs After Denoising for Feature PIR_1")
# plt.xlabel("Time Step")
# plt.ylabel("Sensor Value")
# plt.legend()
# plt.tight_layout()
# plt.show()

#%% Normalisasi
X_reshaped = X_denoised.reshape(-1, X_denoised.shape[2])
X_df = pd.DataFrame(X_reshaped, columns=numerical_cols)
X_df_before_norm = X_df.copy()
X_scaled_df = normalize_zscore(X_df)
X = X_scaled_df.values.reshape(X_denoised.shape)

# Visualisasi distribusi nilai sesudah normalisasi
plt.figure(figsize=(15, 5))
sns.boxplot(data=X_df_before_norm)
plt.title("Distribusi Nilai Sebelum Normalisasi Z-Score")
plt.ylabel("Nilai Asli")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualisasi distribusi nilai sesudah normalisasi
plt.figure(figsize=(15, 5))
sns.boxplot(data=X_scaled_df)
plt.title("Distribusi Nilai Setelah Normalisasi Z-Score")
plt.ylabel("Z-Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% Distribusi kelas
plot_class_distribution(y)
for idx, count in enumerate(np.bincount(y)):
    print(f"Class {idx}: {count} data")

#%% Splitting
X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y, TEST_SIZE, VAL_SIZE, RANDOM_SEED)
# Tampilkan bentuk data
print(f">> Training set shape: {X_train.shape}")
print(f">> Validation set shape: {X_val.shape}")
print(f">> Test set shape: {X_test.shape}")

#%% GUI function
def run_selected_model(model_name, balancing):
    
    print(f"\nRunning: {model_name} + {balancing}...") 
    
    Xb, yb = X_train, y_train

    if balancing == 'SMOTE':
        print(f"✓ SMOTE applied")
        Xb, yb = apply_smote(Xb.reshape(len(Xb), -1), yb)
        Xb = Xb.reshape(-1, WINDOW_SIZE, X.shape[2])
    elif balancing == 'RUS':
        print(f"✓ RUS applied")
        Xb, yb = apply_rus(Xb.reshape(len(Xb), -1), yb)
        Xb = Xb.reshape(-1, WINDOW_SIZE, X.shape[2])
    else:
        print(f"✓ No Balancing applied")

    if model_name == 'CNN-LSTM':
        print(f"✓ CNN-LSTM applied")
        model = build_cnn_lstm((WINDOW_SIZE, X.shape[2]), 3)
        train_model(model, Xb, yb, X_val, y_val)
        y_pred = model.predict(X_test).argmax(axis=1)

    elif model_name == 'LSTM':
        print(f"✓ LSTM applied")
        model = build_lstm((WINDOW_SIZE, X.shape[2]), 3)
        train_model(model, Xb, yb, X_val, y_val)
        y_pred = model.predict(X_test).argmax(axis=1)

    else:
        print(f"✓ KNN applied")
        Xb_flat = Xb.reshape(len(Xb), -1)
        Xval_flat = X_val.reshape(len(X_val), -1)
        Xtest_flat = X_test.reshape(len(X_test), -1)
        model, best_k, best_acc = train_knn_with_validation(Xb_flat, yb, Xval_flat, y_val)
        print(f"KNN Best k = {best_k}, Validation Accuracy = {best_acc:.4f}")
        y_pred = model.predict(Xtest_flat)

    report_dict, cm = evaluate_classification(y_test, y_pred)
    report_df = pd.DataFrame(report_dict).transpose()
    return report_df, cm, f"{model_name} + {balancing}"

#%% Widget Setup
# model_options = ['CNN-LSTM', 'LSTM', 'KNN']
# model_checkboxes = [widgets.Checkbox(value=False, indent=False, description=opt,
#                                      layout=widgets.Layout(width='200px', margin='0 0 0 26px')) for opt in model_options]
# model_box = widgets.VBox([
#     widgets.HTML("<h3 style='margin: 0 0 0 0'>🧠 Pilih Model Klasifikasi:</h3>"),
#     *model_checkboxes
# ], layout=widgets.Layout(margin='0 0 0 20px'))

# balancing_options = ['No Balancing', 'SMOTE', 'RUS']
# balancing_checkboxes = [widgets.Checkbox(value=False, indent=False, description=opt,
#                                          layout=widgets.Layout(width='200px', margin='0 0 0 26px')) for opt in balancing_options]
# balancing_box = widgets.VBox([
#     widgets.HTML("<h3 style='margin: 0 0 0 0'>⚖️ Pilih Metode Balancing:</h3>"),
#     *balancing_checkboxes
# ], layout=widgets.Layout(margin='0 0 0 20px'))

# run_button = widgets.Button(
#     description='RUN MODEL + BALANCING METHOD',
#     button_style='primary',
#     layout=widgets.Layout(width='300px', margin='20px 0 0 0', align_self='center')
# )

# out_process = widgets.Output()
# out_result = widgets.Output()
# evaluations = []

# # RUN logic
# def on_run_button_clicked(b):
#     with out_process:
#         clear_output()
#         selected_models = [cb.description for cb in model_checkboxes if cb.value]
#         selected_balancing = [cb.description for cb in balancing_checkboxes if cb.value]
#         if not selected_models or not selected_balancing:
#             display(widgets.HTML("<span style='color:red'>Silakan pilih minimal 1 model dan 1 balancing method.</span>"))
#             return
#         evaluations.clear()
#         for model in selected_models:
#             for balancing in selected_balancing:
#                 print(f"\nRunning: {model} + {balancing}...") 
#                 report_df, cm, label = run_selected_model(model, balancing)
#                 evaluations.append((report_df, cm, label))
#     with out_result:
#         clear_output()
#         for report_df, cm, label in evaluations:
#             display(widgets.HTML(f"<h3>Evaluation Report: <u>{label}</u></h3>"))
#             display(report_df.round(4))
#             print()
#             plot_confusion_matrix(cm, classes=["vacancy", "stationary", "motion"], title=f"Confusion Matrix - {label}")
#             display(widgets.HTML("<hr>"))

# run_button.on_click(on_run_button_clicked)

# # Display
# display(widgets.VBox([
#     widgets.HTML("<h1 style='text-align: center; margin: 10px 0 0 0'>Eksperimen PIRvision Classification</h1>"),
#     widgets.HTML("<p style='text-align: center; margin: 0 0 0 0'>Silakan pilih model klasifikasi dan metode balancing yang ingin diuji.</p>"),
#     widgets.HTML("<hr>"),
#     widgets.HBox([model_box, balancing_box], layout=widgets.Layout(justify_content='flex-start', gap='40px')),
#     run_button,
#     widgets.HTML("<hr>"),
#     widgets.HTML("<h2>🛠️ Proses Training:</h2>"),
#     out_process,
#     widgets.HTML("<h2>📈 Hasil Evaluasi:</h2>"),
#     out_result
# ]))

model_dropdown = widgets.Dropdown(options=['CNN-LSTM', 'LSTM', 'KNN'], description='Model:')
balancing_dropdown = widgets.Dropdown(options=['No Balancing', 'SMOTE', 'RUS'], description='Balancing:')
run_button = widgets.Button(description='Run Model')
# out = widgets.Output()

# def on_run_button_clicked(b):
#     with out:
#         out.clear_output()
#         run_selected_model(model_dropdown.value, balancing_dropdown.value)

# run_button = widgets.Button(
#     description='RUN MODEL + BALANCING METHOD',
#     button_style='primary',
#     layout=widgets.Layout(width='300px', margin='20px 0 0 0', align_self='center')
# )

run_button = widgets.Button(
    description='Run Model & Balancing Method',
    button_style='primary',
    layout=widgets.Layout(width='250px', margin='30px 0 10px 0', align_self='center')
)

out_process = widgets.Output()
out_result = widgets.Output()
evaluations = []

# RUN logic
def on_run_button_clicked(b):
    with out_process:
        clear_output()
        evaluations.clear()
        report_df, cm, label = run_selected_model(model_dropdown.value, balancing_dropdown.value)
        evaluations.append((report_df, cm, label))
    with out_result:
        clear_output()
        for report_df, cm, label in evaluations:
            print(f"Evaluation Report {label}")
            display(report_df.round(4))
            print()
            plot_confusion_matrix(cm, classes=["vacancy", "stationary", "motion"], title=f"Confusion Matrix - {label}")
            display(widgets.HTML("<hr>"))

run_button.on_click(on_run_button_clicked)

display(widgets.VBox([
    widgets.HTML("<h1 style='text-align: center; margin: 10px 0 0 0'>Eksperimen PIRvision Classification</h1>"),
    widgets.HTML("<p style='text-align: center; margin: 0 0 10px 0'>Silakan pilih model klasifikasi dan metode balancing yang ingin diuji.</p>"),
    model_dropdown, 
    balancing_dropdown, 
    run_button, 
    widgets.HTML("<hr>"),
    widgets.HTML("<h2>🛠️ Proses Training:</h2>"),
    out_process,
    widgets.HTML("<h2>📈 Hasil Evaluasi:</h2>"),
    out_result
]))

# %%

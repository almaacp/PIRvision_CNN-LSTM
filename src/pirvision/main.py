#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display

from pirvision.config import *
from pirvision.dataloader import load_and_segment_data, stratified_split
from pirvision.preprocessing import handle_missing_values, detect_outliers_iqr, normalize_zscore
from pirvision.balancing import apply_smote, apply_rus
from pirvision.models.cnn_lstm import build_cnn_lstm
from pirvision.models.lstm import build_lstm
from pirvision.models.knn import train_knn_with_validation
from pirvision.trainer import train_model
from pirvision.evaluator import evaluate_classification
from pirvision.utils import plot_confusion_matrix, plot_class_distribution

#%% Load raw data
print(">> Loading dataset...")
df = load_and_segment_data(DATA_PATHS)
print(f">> Dataset shape before null handling: {df.shape}")
null_count = df.isnull().sum().sum()
print(f">> Total null values: {null_count}")
if null_count > 0:
    df = handle_missing_values(df)
    print(f">> Dataset shape after null handling: {df.shape}")
else:
    print(">> No missing values found.")

#%% Pisahkan label dan fitur
labels = df["Label"].replace({3: 2}).values  # langsung ubah label 3→2
features_df = df.drop(columns=["Label"])     # hanya fitur saja

#%% Outlier analysis hanya pada fitur
plt.figure()
sns.boxplot(data=features_df, x='Temperature_F')
plt.title("Before Outlier Removal")
plt.tight_layout()
plt.show()

features_df = detect_outliers_iqr(features_df, 'Temperature_F')

plt.figure()
sns.boxplot(data=features_df, x='Temperature_F')
plt.title("After Outlier Removal (IQR)")
plt.tight_layout()
plt.show()

#%% Normalize & segmentasi
pir_cols = [col for col in features_df.columns if col.startswith("PIR_")]
features = features_df[pir_cols + ['Temperature_F']]
features = normalize_zscore(features)

segment_size = WINDOW_SIZE
total_segments = len(features) // segment_size
X = features.values[:total_segments * segment_size].reshape((total_segments, segment_size, -1))
y = labels[:total_segments * segment_size].reshape(-1, segment_size)[:, 0]

#%% Visualisasi & distribusi kelas
plot_class_distribution(y)
for idx, count in enumerate(np.bincount(y)):
    print(f"Class {idx}: {count} data")

#%% Splitting
X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y, TEST_SIZE, VAL_SIZE, RANDOM_SEED)

#%% GUI function
def run_selected_model(model_name, balancing):
    print(f"\n>> Running: {model_name} + {balancing}")

    Xb, yb = X_train, y_train

    if balancing == 'SMOTE':
        print(f">> SMOTE applied")
        Xb, yb = apply_smote(Xb.reshape(len(Xb), -1), yb)
        Xb = Xb.reshape(-1, WINDOW_SIZE, X.shape[2])
    elif balancing == 'RUS':
        print(f">> RUS applied")
        Xb, yb = apply_rus(Xb.reshape(len(Xb), -1), yb)
        Xb = Xb.reshape(-1, WINDOW_SIZE, X.shape[2])
    else:
        print(f">> No Balancing applied")

    if model_name == 'CNN-LSTM':
        print(f">> CNN-LSTM applied")
        model = build_cnn_lstm((WINDOW_SIZE, X.shape[2]), 3)
        train_model(model, Xb, yb, X_val, y_val)
        y_pred = model.predict(X_test).argmax(axis=1)
    elif model_name == 'LSTM':
        print(f">> LSTM applied")
        model = build_lstm((WINDOW_SIZE, X.shape[2]), 3)
        train_model(model, Xb, yb, X_val, y_val)
        y_pred = model.predict(X_test).argmax(axis=1)
    else:
        print(f">> KNN applied")
        Xb_flat = Xb.reshape(len(Xb), -1)
        Xval_flat = X_val.reshape(len(X_val), -1)
        Xtest_flat = X_test.reshape(len(X_test), -1)
        model, best_k, best_acc = train_knn_with_validation(Xb_flat, yb, Xval_flat, y_val)
        print(f">> KNN Best k={best_k} | Validation Accuracy={best_acc:.4f}")
        y_pred = model.predict(Xtest_flat)

    acc, prec, rec, f1, cm = evaluate_classification(y_test, y_pred)
    print()
    print(f"Accuracy={acc:.4f}      Precision={prec:.4f}")
    print(f"Recall={rec:.4f}        F1-Score={f1:.4f}")
    plot_confusion_matrix(cm, classes=["Vacancy", "Stationary", "Motion"], title=f"{model_name} + {balancing}")

#%% Interactive widgets
model_dropdown = widgets.Dropdown(options=['CNN-LSTM', 'LSTM', 'KNN'], description='Model:')
balancing_dropdown = widgets.Dropdown(options=['No Balancing', 'SMOTE', 'RUS'], description='Balancing:')
run_button = widgets.Button(description='Run Model')
out = widgets.Output()

def on_run_button_clicked(b):
    with out:
        out.clear_output()
        run_selected_model(model_dropdown.value, balancing_dropdown.value)

run_button.on_click(on_run_button_clicked)
display(widgets.VBox([model_dropdown, balancing_dropdown, run_button, out]))

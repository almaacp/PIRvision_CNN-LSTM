from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def evaluate_classification(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)    # Menghitung akurasi
    # Menghitung precision, recall, dan F1-score dengan rata-rata makro
    prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)   # Menghitung confusion matrix
    return acc, prec, recall, f1, cm

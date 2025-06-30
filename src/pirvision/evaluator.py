def evaluate_classification(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    acc = accuracy_score(y_true, y_pred)
    prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    return acc, prec, recall, f1, cm

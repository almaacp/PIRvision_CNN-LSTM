from sklearn.metrics import classification_report, confusion_matrix

def evaluate_classification(y_true, y_pred):
    class_names = ["vacancy", "stationary", "motion"]
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    return report, cm

def normalized_root_mse(true, pred):
    from sklearn.metrics import mean_squared_error
    import numpy as np
    return np.sqrt(mean_squared_error(true, pred)) / (np.max(true) - np.min(true))

def plot_confusion_matrix(cm, classes, title):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

def plot_class_distribution(labels):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    sns.countplot(x=pd.Series(labels))
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
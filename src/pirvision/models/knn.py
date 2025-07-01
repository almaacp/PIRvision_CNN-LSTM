from sklearn.neighbors import KNeighborsClassifier

def train_knn_with_validation(X_train, y_train, X_val, y_val, k_values=range(1, 16)):
    best_acc = -1
    best_model = None
    best_k = None

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        model.fit(X_train, y_train)
        acc = model.score(X_val, y_val)
        print(f"K={k}: Accuracy={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_k = k

    return best_model, best_k, best_acc
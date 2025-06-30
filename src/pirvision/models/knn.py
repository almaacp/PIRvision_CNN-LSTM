def train_knn(X_train, y_train, k=5):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=k)
    return model.fit(X_train, y_train)

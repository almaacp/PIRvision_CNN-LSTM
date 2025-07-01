from sklearn.neighbors import KNeighborsClassifier

def train_knn_with_validation(X_train, y_train, X_val, y_val, k_values=range(1, 16)):
    best_acc = -1   # Inisialisasi akurasi terbaik
    best_model = None   # Inisialisasi model terbaik
    best_k = None   # Inisialisasi nilai k terbaik
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k, metric='euclidean') # Inisialisasi KNN dengan k yang berbeda
        model.fit(X_train, y_train) # Melatih model KNN dengan data training
        acc = model.score(X_val, y_val) # Menghitung akurasi pada data validasi
        print(f"K={k}: Accuracy={acc:.4f}") # Menampilkan akurasi untuk setiap nilai k
        # Simpan model dan nilai k sebagai yang terbaik apabila akurasi lebih baik dari sebelumnya
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_k = k
    return best_model, best_k, best_acc
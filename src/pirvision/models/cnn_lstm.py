from tensorflow.keras import layers, models

def build_cnn_lstm(input_shape, num_classes):
    # Inisialisasi model Sequential
    model = models.Sequential()
    # Tambahkan layer Conv1D dengan 32 filter, kernel size 3, dan fungsi aktivasi ReLU
    model.add(layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    # Tambahkan layer MaxPooling1D dengan ukuran pool 2
    model.add(layers.MaxPooling1D(2))
    # Tambahkan layer LSTM dengan 64 unit
    model.add(layers.LSTM(64))
    # Tambahkan layer Dropout untuk regularisasi
    model.add(layers.Dropout(0.5))
    # Tambahkan layer Dense output dengan fungsi aktivasi softmax untuk klasifikasi multi-kelas
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

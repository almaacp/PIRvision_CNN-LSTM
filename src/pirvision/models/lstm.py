from tensorflow.keras import layers, models

def build_lstm(input_shape, num_classes):
    # Inisialisasi model Sequential
    model = models.Sequential()
    # Tambahkan layer LSTM dengan 64 unit
    model.add(layers.LSTM(64, input_shape=input_shape))
    # Tambahkan layer Dropout untuk regularisasi
    model.add(layers.Dropout(0.5))
    # Tambahkan layer Dense output dengan fungsi aktivasi softmax untuk klasifikasi multi-kelas
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model
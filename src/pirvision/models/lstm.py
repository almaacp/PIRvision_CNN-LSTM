def build_lstm(input_shape, num_classes):
    from tensorflow.keras import layers, models
    model = models.Sequential()
    model.add(layers.LSTM(64, input_shape=input_shape))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model
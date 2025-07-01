def train_model(model, X_train, y_train, X_val, y_val, epochs=20):
    # Kompilasi model dengan optimizer Adam dan loss function sparse categorical crossentropy
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])   
    # Fit model dengan data training dan validation
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=32)
    return history
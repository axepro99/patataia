
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from data_loader import get_image_generators

def create_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid') # Binaria: dirección up/down
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("--- Entrenamiento base con Candle Stick Patterns ---")
    batch_size = 32
    img_size = (224, 224)
    candle_train_gen, candle_val_gen, stock_train_gen, stock_val_gen = get_image_generators(batch_size, img_size)

    input_shape = (img_size[0], img_size[1], 3)
    model = create_cnn_model(input_shape)
    model.summary()

    # Entrenamiento base
    history = model.fit(
        candle_train_gen,
        epochs=20,
        validation_data=candle_val_gen
    )

    print("--- Fine-tuning con Stock Chart Patterns ---")
    # Opcional: Unfreeze layers, cambiar learning rate, etc.
    model.fit(
        stock_train_gen,
        epochs=10,
        validation_data=stock_val_gen
    )

    model.save('candle_stock_direction_cnn.h5')
    print("Modelo guardado como 'candle_stock_direction_cnn.h5'")


if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()
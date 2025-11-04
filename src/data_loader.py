
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_image_generators(batch_size=32, img_size=(224, 224)):
    """
    Crea generadores de datos para imágenes de Candle Stick Patterns y Stock Chart Patterns.
    Las carpetas deben tener subcarpetas por clase: 'up' y 'down'.
    Args:
        batch_size (int): Tamaño de lote.
        img_size (tuple): Tamaño de las imágenes.
    Returns:
        train_gen, val_gen: Generadores de Keras para entrenamiento y validación.
    """
    # Candle Stick Patterns (entrenamiento base)
    candle_dir = os.path.join(os.getcwd(), 'candle-stick-patterns', 'image-data')
    # Stock Chart Patterns (fine-tuning)
    stock_dir = os.path.join(os.getcwd(), 'stock-chart-patterns', 'Patterns')

    # Data augmentation para entrenamiento
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    # Generador para Candle Stick Patterns
    candle_train_gen = train_datagen.flow_from_directory(
        candle_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True
    )
    candle_val_gen = train_datagen.flow_from_directory(
        candle_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=True
    )

    # Generador para Stock Chart Patterns (fine-tuning)
    stock_train_gen = train_datagen.flow_from_directory(
        stock_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True
    )
    stock_val_gen = train_datagen.flow_from_directory(
        stock_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=True
    )

    return candle_train_gen, candle_val_gen, stock_train_gen, stock_val_gen
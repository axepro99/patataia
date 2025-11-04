import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from src.preprocess import preprocess_data

# --- Parámetros de Predicción ---
MODEL_PATH = 'crypto_predictor.h5'
SEQUENCE_LENGTH = 60
SYMBOL = 'BTCUSDT'

def make_prediction():
    """
    Carga el modelo entrenado y realiza una predicción de ejemplo.
    """
    # --- 1. Cargar el Modelo ---
    try:
        model = load_model(MODEL_PATH)
        print("Modelo cargado exitosamente.")
    except (IOError, ImportError) as e:
        print(f"Error al cargar el modelo desde {MODEL_PATH}.")
        print("Asegúrate de haber entrenado el modelo primero ejecutando 'src/train.py'.")
        print(f"Error: {e}")
        return

    # --- 2. Cargar y Preparar los Datos de Entrada ---
    # Para una predicción real, aquí descargarías los datos más recientes.
    # Por simplicidad, usaremos los últimos datos del archivo CSV que ya tenemos.
    try:
        df_full = pd.read_csv(f'data_{SYMBOL.lower()}_1m_1y.csv', parse_dates=['Open time'])
    except FileNotFoundError:
        print("Archivo de datos no encontrado. No se puede realizar la predicción.")
        return

    # Usamos las últimas (SEQUENCE_LENGTH * 2) filas para asegurar que tenemos suficientes datos
    # para el preprocesamiento y la creación de la última secuencia.
    last_data_chunk = df_full.tail(SEQUENCE_LENGTH * 2)

    # Preprocesamos este fragmento de datos para obtener el scaler y la última secuencia
    _, _, scaler, target_col_index = preprocess_data(last_data_chunk, SEQUENCE_LENGTH)
    
    # Re-escalamos todos los datos para obtener la última secuencia
    scaled_data = scaler.transform(last_data_chunk.select_dtypes(include=np.number))
    
    # La última secuencia disponible para la predicción
    last_sequence = np.array([scaled_data[-SEQUENCE_LENGTH:]])
    
    print(f"\nForma de la secuencia de entrada para la predicción: {last_sequence.shape}")

    # --- 3. Realizar la Predicción ---
    predicted_price_scaled = model.predict(last_sequence)

    # --- 4. Invertir la Normalización ---
    # La predicción está en la escala 0-1. Necesitamos devolverla a su valor original.
    # Creamos un array temporal con la misma forma que los datos originales (número de features)
    # para poder usar el `inverse_transform` del scaler.
    dummy_array = np.zeros((1, scaled_data.shape[1]))
    dummy_array[:, target_col_index] = predicted_price_scaled
    
    # Invertir la transformación
    predicted_price = scaler.inverse_transform(dummy_array)[0, target_col_index]

    # --- 5. Mostrar el Resultado ---
    last_real_price = df_full['Close'].iloc[-1]
    print(f"\nÚltimo precio de cierre real ({SYMBOL}): {last_real_price:.4f}")
    print(f"Predicción del siguiente precio de cierre ({SYMBOL}): {predicted_price:.4f}")

    if predicted_price > last_real_price:
        print("Tendencia predicha: Alcista 📈")
    else:
        print("Tendencia predicha: Bajista 📉")


if __name__ == '__main__':
    make_prediction()
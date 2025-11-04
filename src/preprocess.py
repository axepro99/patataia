import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ta import add_all_ta_features

def preprocess_data(df, sequence_length=60):
    """
    Preprocesa los datos para el modelo LSTM.

    Args:
        df (pandas.DataFrame): DataFrame con los datos de las velas.
        sequence_length (int): Longitud de la secuencia de entrada para el modelo.

    Returns:
        tuple: Contiene (X, y, scaler) donde X son las secuencias de entrada,
               y son los valores a predecir, y scaler es el objeto para
               la normalización.
    """
    print("Iniciando preprocesamiento de datos...")
    
    # --- 1. Feature Engineering: Añadir indicadores técnicos ---
    # Usamos la librería 'ta' para añadir varios indicadores populares
    df = add_all_ta_features(
        df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
    )
    print(f"Columnas después de añadir indicadores técnicos: {df.columns.tolist()}")

    # --- 2. Seleccionar la columna objetivo y normalizar los datos ---
    # Vamos a predecir el precio de cierre ('Close')
    target_col = 'Close'
    
    # Crear una copia para evitar SettingWithCopyWarning
    data_to_process = df.copy()

    # Normalizar todas las características numéricas entre 0 y 1
    # Esto es crucial para que las redes neuronales funcionen bien
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_to_process.select_dtypes(include=np.number))

    # --- 3. Crear las secuencias para el modelo LSTM ---
    # El modelo usará 'sequence_length' velas para predecir la siguiente.
    X, y = [], []
    
    # El índice de la columna 'Close'
    target_col_index = data_to_process.columns.get_loc(target_col)

    for i in range(sequence_length, len(scaled_data)):
        # X: Secuencia de 'sequence_length' velas
        X.append(scaled_data[i - sequence_length:i])
        # y: El precio de cierre de la siguiente vela
        y.append(scaled_data[i, target_col_index])

    X, y = np.array(X), np.array(y)
    
    print(f"Forma de los datos de entrada (X): {X.shape}")
    print(f"Forma de los datos de salida (y): {y.shape}")
    
    return X, y, scaler, df.columns.get_loc(target_col)

if __name__ == '__main__':
    # --- Ejemplo de uso ---
    try:
        df = pd.read_csv('data_btcusdt_1m_1y.csv', parse_dates=['Open time'])
    except FileNotFoundError:
        print("Archivo de datos no encontrado. Ejecuta 'src/data_loader.py' primero.")
        exit()

    X, y, scaler, target_col_index = preprocess_data(df, sequence_length=60)
    
    print("\nPreprocesamiento completado.")
    print(f"Índice de la columna objetivo ('Close'): {target_col_index}")

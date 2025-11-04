from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_lstm_model(input_shape):
    """
    Crea un modelo secuencial de tipo LSTM (Long Short-Term Memory),
    ideal para datos de series temporales como los precios de acciones.

    Args:
        input_shape (tuple): La forma de los datos de entrada (timesteps, features).
                             Por ejemplo: (60, 1) para 60 días de precios de cierre.

    Returns:
        tensorflow.keras.Model: El modelo LSTM compilado.
    """
    model = Sequential()

    # Capa 1: LSTM
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    # Capa 2: LSTM
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    # Capa de salida
    model.add(Dense(units=1)) # Salida de 1 valor (la predicción del siguiente precio)

    # Compilación del modelo
    model.compile(optimizer='adam', loss='mean_squared_error')

    print("Modelo LSTM creado con éxito.")
    model.summary()

    return model

if __name__ == '__main__':
    # Ejemplo de cómo crear el modelo
    # Suponemos que usaremos 60 time steps (ej: 60 días) y 1 feature (ej: precio de cierre)
    example_input_shape = (60, 1)
    lstm_model = create_lstm_model(example_input_shape)
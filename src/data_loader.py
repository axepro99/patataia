import pandas as pd
import os

def load_stock_data(data_path='../data'):
    """
    Carga los datos de los patrones de gráficos de bolsa desde un archivo CSV.

    Args:
        data_path (str): La ruta a la carpeta 'data' donde se encuentran los CSV.

    Returns:
        pandas.DataFrame: Un DataFrame con los datos cargados.
    """
    # Asume que el dataset de patrones de gráficos tiene un CSV principal.
    # DEBERÁS AJUSTAR 'nombre_del_fichero.csv' al nombre real del archivo.
    csv_path = os.path.join(data_path, 'stock-chart-patterns', 'nombre_del_fichero.csv') # ¡AJUSTA ESTE NOMBRE!

    if not os.path.exists(csv_path):
        print(f"Error: No se encontró el fichero en {csv_path}")
        print("Asegúrate de haber descargado el dataset y de que el nombre del fichero es correcto.")
        return None

    print(f"Cargando datos desde {csv_path}...")
    df = pd.read_csv(csv_path)
    print("Datos cargados con éxito.")
    return df

if __name__ == '__main__':
    # Ejemplo de cómo usar la función
    stock_patterns_df = load_stock_data()
    if stock_patterns_df is not None:
        print("\nPrimeras 5 filas del dataset:")
        print(stock_patterns_df.head())
        print("\nInformación del DataFrame:")
        stock_patterns_df.info()
# Proyecto de Predicción del Mercado de Valores

Este proyecto tiene como objetivo desarrollar un modelo de Machine Learning para predecir los movimientos del mercado de valores a corto plazo. Se utilizarán técnicas de análisis de patrones de velas y gráficos de bolsa.

## Datasets

Este modelo se entrenará utilizando los siguientes datasets de Kaggle:

1.  [Candle Stick Patterns](https://www.kaggle.com/datasets/mineshjethva/candle-stick-patterns)
2.  [Stock Chart Patterns](https://www.kaggle.com/datasets/mustaphaelbakai/stock-chart-patterns)

## Estructura del Proyecto

-   `/data`: (Carpeta local, no en Git) Aquí debes descargar y descomprimir los datasets.
-   `/notebooks`: Contiene Jupyter Notebooks para el análisis exploratorio de datos (EDA).
-   `/src`: Contiene el código fuente principal.
    -   `data_loader.py`: Scripts para cargar y preprocesar los datos.
    -   `model.py`: Definición del modelo de Machine Learning (ej. LSTM, RandomForest, etc.).
    -   `train.py`: Script para entrenar el modelo.
-   `requirements.txt`: Dependencias de Python para el proyecto.

## Cómo Empezar

1.  **Clona el repositorio:**
    ```bash
    git clone https://github.com/axepro99/patataia.git
    cd patataia
    ```

2.  **Crea un entorno virtual e instala las dependencias:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Descarga los datos:**
    -   Ve a los enlaces de Kaggle mencionados arriba.
    -   Descarga los datasets.
    -   Crea una carpeta `data/` en la raíz del proyecto.
    -   Descomprime el contenido de los datasets dentro de la carpeta `data/`.

4.  **Explora los datos:**
    -   Inicia Jupyter Lab o Notebook:
      ```bash
      jupyter lab
      ```
    -   Abre el notebook que se encuentra en la carpeta `notebooks/` para empezar el análisis.
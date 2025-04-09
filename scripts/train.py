import numpy as np
import os
import pandas as pd
from scripts.model import create_model
from scripts.preprocessing import preprocess_image


def load_data(data_dir, labels_path):
    # Cargar etiquetas desde CSV
    labels_df = pd.read_csv(labels_path)

    # Imprimir las columnas disponibles para depuración
    print("Columnas disponibles en el CSV:", labels_df.columns.tolist())

    # Verificar si las columnas necesarias existen
    required_columns = ["nudos", "entrenudos", "distancia", "largo", "grosor"]
    altura_columns = []

    # Buscar columnas de altura
    for col in labels_df.columns:
        if "altura" in col.lower():
            altura_columns.append(col)

    print(f"Columnas de altura encontradas: {altura_columns}")

    # Verificar que todas las columnas requeridas existen
    missing_columns = [col for col in required_columns if col not in labels_df.columns]
    if missing_columns:
        raise ValueError(f"Columnas faltantes en el CSV: {missing_columns}")

    if len(altura_columns) < 2:
        raise ValueError(
            f"Se necesitan al menos dos columnas de altura (con/sin curvatura)"
        )

    X, y = [], []
    for fname in os.listdir(data_dir):
        if fname.endswith(".jpg"):
            # Buscar etiqueta correspondiente en el CSV
            label_row = labels_df[labels_df["filename"] == fname]

            if not label_row.empty:
                img = preprocess_image(os.path.join(data_dir, fname))
                X.append(img)

                # Extraer valores de etiquetas
                y_values = [
                    label_row["nudos"].values[0],
                    label_row["entrenudos"].values[0],
                    label_row["distancia"].values[0],
                    label_row["largo"].values[0],
                    label_row["grosor"].values[0],
                    label_row[altura_columns[0]].values[0],  # Primera columna de altura
                    label_row[altura_columns[1]].values[0],  # Segunda columna de altura
                ]
                y.append(y_values)
                print(f"Cargada imagen: {fname} con etiquetas: {y_values}")

    return np.array(X), np.array(y)


def train():
    # Asegurar que existen los directorios necesarios
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models/trained", exist_ok=True)

    X, y = load_data("data/processed", "data/labels/labels.csv")
    model = create_model()
    model.fit(X, y, epochs=10, validation_split=0.2)
    model.save("models/trained/cana_model.h5")


if __name__ == "__main__":
    train()

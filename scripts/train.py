import numpy as np
import os
import pandas as pd
from model import create_model
from preprocessing import preprocess_image


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

    dataset = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".jpg"):
            # Buscar etiqueta correspondiente en el CSV
            label_row = labels_df[labels_df["filename"] == fname]

            if not label_row.empty:
                img, _ = preprocess_image(os.path.join(data_dir, fname))
                if img is not None:
                    # Extraer valores de etiquetas
                    y_values = [
                        label_row["nudos"].values[0],
                        label_row["entrenudos"].values[0],
                        label_row["distancia"].values[0],
                        label_row["largo"].values[0],
                        label_row["grosor"].values[0],
                        label_row[altura_columns[0]].values[0],
                        label_row[altura_columns[1]].values[0],
                    ]
                    if len(y_values) == 7:
                        dataset.append((img, y_values))
                        print(f"✅ Cargada imagen: {fname} con etiquetas: {y_values}")
                    else:
                        print(f"❌ Etiquetas con longitud incorrecta para imagen: {fname}")
                else:
                    print(f"⚠️ Imagen inválida (None) al procesar: {fname}")
            else:
                print(f"❌ No se encontraron etiquetas para la imagen: {fname}")

    # Filtrado de imágenes inválidas (por forma o tipo)
    filtered_dataset = []
    for i, data in enumerate(dataset):
        img, label = data
        if img is None:
            print(f"❌ Imagen en índice {i} es None.")
        elif not isinstance(img, np.ndarray):
            print(f"❌ Tipo inválido en índice {i}: {type(img)}")
        elif img.shape != (224, 224, 3):
            print(f"⚠️ Imagen con forma inesperada en índice {i}: {img.shape}")
        else:
            filtered_dataset.append((img, label))

    # Separar imágenes y etiquetas listas para entrenamiento
    X = np.array([data[0] for data in filtered_dataset])
    y = np.array([data[1] for data in filtered_dataset])
    return X, y


def train():
    # Asegurar que existen los directorios necesarios
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models/trained", exist_ok=True)

    X, y = load_data("data/processed", "data/labels/auto_labels.csv")

    print(f"🔢 Total de muestras válidas: {len(X)}")

    model = create_model()
    model.fit(X, y, epochs=10, validation_split=0.2)
    model.save("models/trained/cana_model.h5")
    print("✅ Modelo entrenado y guardado correctamente.")


if __name__ == "__main__":
    train()

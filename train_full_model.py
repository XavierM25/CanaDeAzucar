import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.enhanced_model import create_enhanced_model
from scripts.preprocessing import preprocess_image


def train_model_with_all_images():
    """Entrena un modelo potente con todas las imágenes disponibles"""
    # Configuración xd
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    labels_file = "data/labels/auto_labels.csv"
    output_dir = "models/trained"

    os.makedirs(output_dir, exist_ok=True)

    # Verificar si existen etiquetas
    if not os.path.exists(labels_file):
        print("❌ No se encontraron etiquetas en", labels_file)
        print("Primero ejecuta: python create_dataset.py")
        return

    # Cargar etiquetas
    labels_df = pd.read_csv(labels_file)
    print(f"✅ Se cargaron etiquetas para {len(labels_df)} imágenes")

    # Procesar imágenes si no están ya procesadas
    if not os.path.exists(processed_dir) or len(os.listdir(processed_dir)) < len(
        labels_df
    ):
        print("Procesando imágenes...")
        os.makedirs(processed_dir, exist_ok=True)
        from scripts.batch_process import batch_process_images

        batch_process_images(raw_dir, processed_dir, "output/processed_visualizations")

    # Cargar imágenes y etiquetas
    X, y = [], []

    for _, row in labels_df.iterrows():
        filename = row["filename"]

        # Primero intentar desde processed, luego desde raw
        processed_path = os.path.join(processed_dir, filename)
        raw_path = os.path.join(raw_dir, filename)

        image_path = processed_path if os.path.exists(processed_path) else raw_path

        if os.path.exists(image_path):
            try:
                # Cargar y procesar imagen
                if image_path.startswith(processed_dir):
                    # Imagen ya procesada, solo cargar
                    img = cv2.imread(image_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    img = img / 255.0
                else:
                    # Imagen original, procesar
                    img = preprocess_image(image_path)

                X.append(img)

                # Obtener etiquetas
                y_values = [
                    row["nudos"],
                    row["entrenudos"],
                    row["distancia"],
                    row["largo"],
                    row["grosor"],
                    row["altura_con_corbatra"],
                    row["altura_sin_corbatra"],
                ]
                y.append(y_values)
            except Exception as e:
                print(f"❌ Error procesando {filename}: {e}")

    if not X:
        print("❌ No se pudieron cargar imágenes")
        return

    # Convertir a arrays numpy
    X = np.array(X)
    y = np.array(y)
    print(f"✅ Dataset cargado: {len(X)} imágenes")

    # Crear modelo mejorado
    model = create_enhanced_model()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.00001),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, "best_model.h5"),
            save_best_only=True,
            monitor="val_loss",
        ),
    ]

    # Entrenar modelo
    print("🔄 Iniciando entrenamiento...")
    history = model.fit(
        X,
        y,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1,
    )

    # Guardar modelo final
    model.save(os.path.join(output_dir, "cana_model.h5"))
    print(f"✅ Modelo guardado en {os.path.join(output_dir, 'cana_model.h5')}")

    # Visualizar curvas de entrenamiento
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["mae"], label="Train")
    plt.plot(history.history["val_mae"], label="Validation")
    plt.title("Mean Absolute Error")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"))
    print(
        f"✅ Gráficas de entrenamiento guardadas en {os.path.join(output_dir, 'training_curves.png')}"
    )

    return model


if __name__ == "__main__":
    # Importaciones necesarias que podrían faltar
    import cv2

    train_model_with_all_images()

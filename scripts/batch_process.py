import os
import cv2
import numpy as np
from scripts.process_image import preprocess_image, extract_sugarcane_features
import json
import matplotlib.pyplot as plt


def batch_process_images(input_dir, output_dir, visualization_dir):
    """
    Procesar todas las imágenes en un directorio usando el algoritmo avanzado
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)

    results = []

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            vis_path = os.path.join(
                visualization_dir, f"{os.path.splitext(filename)[0]}_processed.png"
            )
            json_path = os.path.join(
                visualization_dir, f"{os.path.splitext(filename)[0]}_features.json"
            )

            print(f"Procesando: {filename}")

            try:
                # Aplicar el algoritmo avanzado de detección de bordes y extracción de objetos
                processed_img, pixels_per_cm = preprocess_image(input_path)

                # Extraer características
                features = extract_sugarcane_features(input_path, pixels_per_cm)
                features["filename"] = filename
                results.append(features)

                # Guardar imagen procesada
                processed_bgr = cv2.cvtColor(
                    (processed_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
                )
                cv2.imwrite(output_path, processed_bgr)

                # Guardar características en JSON
                with open(json_path, "w") as f:
                    json.dump(features, f, indent=4)

                # Crear visualización
                plt.figure(figsize=(12, 8))

                # Imagen original
                original = cv2.imread(input_path)
                original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                plt.subplot(121)
                plt.imshow(original)
                plt.title("Imagen Original")

                # Imagen procesada
                plt.subplot(122)
                plt.imshow(processed_img)
                plt.title(
                    "Imagen Procesada\n"
                    + f"Nudos: {features['nudos']}, "
                    + f"Entrenudos: {features['entrenudos']}\n"
                    + f"Grosor: {features['grosor_cm']:.2f} cm, "
                    + f"Altura: {features['altura_sin_curvatura_cm']:.2f} cm"
                )

                plt.tight_layout()
                plt.savefig(vis_path)
                plt.close()

                print(f"✅ Procesada: {filename}")

            except Exception as e:
                print(f"❌ Error procesando {filename}: {e}")

    # Guardar resumen de resultados
    if results:
        summary_path = os.path.join(visualization_dir, "resultados_procesamiento.csv")
        import pandas as pd

        df = pd.DataFrame(results)
        df.to_csv(summary_path, index=False)
        print(f"\nResumen guardado en: {summary_path}")

    return len(results)


if __name__ == "__main__":
    num_processed = batch_process_images(
        input_dir="data/raw",
        output_dir="data/processed",
        visualization_dir="output/processed_visualizations",
    )
    print(f"\nSe procesaron {num_processed} imágenes exitosamente.")

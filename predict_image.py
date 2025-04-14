import cv2
import numpy as np
import tensorflow as tf
from scripts.preprocessing import preprocess_image  # Usa el mismo preprocesamiento
import sys

def load_model(model_path="models/trained/cana_model.h5"):
    return tf.keras.models.load_model(model_path)

def predict_features(image_path, model):
    img = preprocess_image(image_path)
    
    if img is None or not isinstance(img, np.ndarray):
        print(f"❌ Error: No se pudo procesar la imagen: {image_path}")
        return None

    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0]
    return prediction


def main(image_path):
    model = tf.keras.models.load_model("models/trained/cana_model.h5")
    prediction = predict_features(image_path, model)

    if prediction is None:
        print("❌ Predicción fallida. Revisa la imagen o el preprocesamiento.")
        return  # Evita que el resto del código intente usar `prediction`

    labels = [
        "Nudos",
        "Entrenudos",
        "Distancia",
        "Largo",
        "Grosor",
        "Altura con corbata",
        "Altura sin corbata",
    ]

    print(f"📷 Resultados para {image_path}:")
    for label, value in zip(labels, prediction):
        print(f"{label}: {value:.2f}")



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Proporcione la ruta a la imagen. Ejemplo:")
        print("   python predict_image.py data/new/my_cana.jpg")
    else:
        image_path = sys.argv[1]
        main(image_path)

import tensorflow as tf
from tensorflow.keras import layers, models


def create_model(input_shape=(224, 224, 3), num_outputs=7):
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(
                num_outputs
            ),  # Salida para: nudos, entrenudos, distancia, largo, grosor, altura, corbata
        ]
    )
    # Usar el nombre completo para la función de pérdida en lugar de 'mse'
    model.compile(
        optimizer="adam", loss=tf.keras.losses.MeanSquaredError(), metrics=["mae"]
    )
    return model

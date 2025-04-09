import tensorflow as tf
from tensorflow.keras import layers, models


def create_enhanced_model(input_shape=(224, 224, 3), num_outputs=7):
    """
    Modelo más potente usando transfer learning con MobileNetV2
    """
    # Cargar MobileNetV2 preentrenado
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )

    # Congelar capas del modelo base
    base_model.trainable = False

    # Construir modelo completo
    model = tf.keras.Sequential(
        [
            # Preprocesamiento
            layers.InputLayer(input_shape=input_shape),
            layers.Lambda(lambda x: x * 2.0 - 1.0),  # Escalar a [-1,1] para MobileNetV2
            # Modelo base
            base_model,
            # Capas de clasificación/regresión
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_outputs),  # Sin activación para regresión
        ]
    )

    # Compilar modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=["mae"],
    )

    return model

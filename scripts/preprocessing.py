import cv2
import os
import numpy as np
from scripts.process_image import (
    extract_sugarcane_features,
    preprocess_image as advanced_preprocess,
)


# Reemplazar la función simple con la avanzada
def preprocess_image(image_path, target_size=(224, 224)):
    """Redirige a la implementación avanzada de preprocesamiento"""
    return advanced_preprocess(image_path, target_size)

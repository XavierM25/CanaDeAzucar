import os
import cv2
from preprocessing import preprocess_image

def batch_preprocess_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(input_dir, filename)
            try:
                processed_image = preprocess_image(image_path)
                # Convertir de float32 [0, 1] a uint8 [0, 255]
                image_bgr = (processed_image * 255).astype("uint8")
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR))
                print(f"✅ Procesada: {filename}")
            except Exception as e:
                print(f"❌ Error procesando {filename}: {e}")

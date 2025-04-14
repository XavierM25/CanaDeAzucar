import cv2
import numpy as np
from imutils import perspective
from imutils import contours as contour_utils
from PIL import Image, ImageFilter
from skimage.feature import peak_local_max
from scipy import ndimage
import os


def midpoint(ptA, ptB):
    """Calcula el punto medio entre dos puntos"""
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def preprocess_image(image_path, target_size=(299, 299), reference_width_cm=164.0):
    """
    Preprocesamiento avanzado para detectar específicamente caña de azúcar en papel blanco
    """
    # Cargar imagen
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")

    original = image.copy()

    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"debug_gray_{os.path.basename(image_path)}", gray)

    # Aplicar umbralización adaptativa para mejor separación
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )
    cv2.imwrite(f"debug_thresh_{os.path.basename(image_path)}", thresh)

    # Aplicar operaciones morfológicas para limpiar ruido
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    cv2.imwrite(f"debug_morph_{os.path.basename(image_path)}", closing)

    # Detectar contornos
    cnts = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    if not cnts:
        print(f"⚠️ No se detectaron contornos en {image_path}")
        cv2.imwrite(f"debug_nocontour_{os.path.basename(image_path)}", closing)
        resized = cv2.resize(image, target_size)
        return (
            cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) / 255.0,
            1.0,
        )  # Valor por defecto

    # Filtrar contornos por área (descartar muy pequeños)
    cnts = [c for c in cnts if cv2.contourArea(c) > 1000]

    if not cnts:
        print(f"⚠️ No se detectaron contornos grandes en {image_path}")
        resized = cv2.resize(image, target_size)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) / 255.0, 1.0

    # PASO 1: Encontrar el papel blanco (marco de referencia)
    # En vez de usar el contorno más grande, buscar un contorno que se parezca a un rectángulo
    paper_contour = None
    paper_area = 0
    height, width = image.shape[:2]
    image_area = height * width

    for c in cnts:
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # Si el contorno se aproxima a un cuadrilátero y es grande
        if len(approx) == 4 and area > paper_area and area < image_area * 0.95:
            paper_contour = c
            paper_area = area

    # Si no encontramos el papel, usar el contorno más grande como referencia
    if paper_contour is None:
        paper_contour = max(cnts, key=cv2.contourArea)

    # PASO 2: Calcular escala usando las dimensiones del papel
    # Calcular dimensiones del papel en píxeles
    paper_rect = cv2.minAreaRect(paper_contour)
    paper_box = cv2.boxPoints(paper_rect)
    paper_box = np.array(paper_box, dtype="int")
    paper_box = perspective.order_points(paper_box)
    
    paper_debug = image.copy()
    cv2.drawContours(paper_debug, [paper_contour], -1, (0, 255, 0), 3)
    cv2.imwrite(f"debug_paper_{os.path.basename(image_path)}", paper_debug)
    
    # Calcular dimensiones del papel
    (tl, tr, br, bl) = paper_box
    widthA = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    widthB = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    paper_width_px = max(int(widthA), int(widthB))

    # Convertir píxeles a centímetros
    pixels_per_cm = paper_width_px / reference_width_cm

    # PASO 3: Detectar la caña de azúcar
    # Crear una máscara del papel
    paper_mask = np.zeros_like(gray)
    cv2.drawContours(paper_mask, [paper_contour], -1, 255, -1)
    cv2.imwrite(f"debug_paper_mask_{os.path.basename(image_path)}", paper_mask)

    # Aplicar la máscara a la imagen umbralizada para encontrar solo objetos dentro del papel
    masked_thresh = cv2.bitwise_and(thresh, thresh, mask=paper_mask)
    
    # Aplicar operaciones morfológicas más específicas para la caña
    kernel_long = np.ones(
        (11, 3), np.uint8
    )  # Kernel largo para resaltar estructuras verticales
    dilated = cv2.dilate(masked_thresh, kernel_long, iterations=1)
    eroded = cv2.erode(dilated, kernel_long, iterations=1)
    
    cv2.imwrite(f"debug_masked_thresh_{os.path.basename(image_path)}", masked_thresh)
    cv2.imwrite(f"debug_eroded_{os.path.basename(image_path)}", eroded)
    
    # Encontrar contornos de objetos dentro del papel
    inner_cnts = cv2.findContours(
        eroded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    inner_cnts = inner_cnts[0] if len(inner_cnts) == 2 else inner_cnts[1]

    if not inner_cnts:
        # Si no encontramos nada, usar una región central del papel
        x, y, w, h = cv2.boundingRect(paper_contour)
        cane_roi = image[y + h // 4 : y + 3 * h // 4, x + w // 4 : x + 3 * w // 4]
        print(f"⚠️ No se detectaron contornos internos en {image_path}")
    else:
        # Filtrar contornos por relación de aspecto y área para encontrar la caña
        print(f"✅ Se detectaron {len(inner_cnts)} contornos internos")
        sugarcane_contour = None
        best_score = 0

        for i, c in enumerate(inner_cnts):
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = h / w if w > 0 else 0

            print(f"  🔍 Contorno #{i}: area={area:.2f}, aspect_ratio={aspect_ratio:.2f}")

            if area > 500 and aspect_ratio > 1.5:  # <-- más estricto y más realista
                score = area * aspect_ratio
                if score > best_score:
                    sugarcane_contour = c
                    best_score = score
            debug_inner = image.copy()
            cv2.drawContours(debug_inner, inner_cnts, -1, (0, 0, 255), 1)
            cv2.imwrite(f"debug_inner_{os.path.basename(image_path)}", debug_inner)

        # Si no encontramos nada que parezca una caña, usar el contorno más grande dentro del papel
        if sugarcane_contour is None and inner_cnts:
            sugarcane_contour = max(inner_cnts, key=cv2.contourArea)
            sugarcane_debug = image.copy()
            cv2.drawContours(sugarcane_debug, [sugarcane_contour], -1, (255, 0, 0), 2)
            cv2.imwrite(f"debug_sugarcane_contour_{os.path.basename(image_path)}", sugarcane_debug)
            print(f"🌿 Caña detectada en {image_path} - Área: {cv2.contourArea(sugarcane_contour):.2f}")
            
        # Recortar la caña
        if sugarcane_contour is not None:
            x, y, w, h = cv2.boundingRect(sugarcane_contour)
            # Añadir un margen
            margin = int(20 * pixels_per_cm / 10)  # Margen proporcional a la escala
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(image.shape[1] - x, w + 2 * margin)
            h = min(image.shape[0] - y, h + 2 * margin)
            cane_roi = image[y : y + h, x : x + w]
        else:
            # Usar una región central del papel como última opción
            x, y, w, h = cv2.boundingRect(paper_contour)
            cane_roi = image[y + h // 4 : y + 3 * h // 4, x + w // 4 : x + 3 * w // 4]

    # Verificar si ROI es válido
    if cane_roi.size == 0 or cane_roi.shape[0] == 0 or cane_roi.shape[1] == 0:
        print(f"⚠️   inválido en {image_path}, usando imagen original")
        cane_roi = image

    # Mejorar la calidad de la imagen recortada
    pil_image = Image.fromarray(cv2.cvtColor(cane_roi, cv2.COLOR_BGR2RGB))
    enhanced = pil_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    enhanced = np.array(enhanced)

    # Redimensionar para el modelo
    resized = cv2.resize(enhanced, target_size)

    # Convertir a RGB y normalizar
    # Asegura que el array sea uint8 antes de convertir
    resized_uint8 = (resized * 255).astype(np.uint8) if resized.max() <= 1.0 else resized.astype(np.uint8)

    # No necesitas convertirlo a BGR otra vez si ya está en RGB (PIL)
    normalized = resized_uint8 / 255.0
    if normalized.ndim != 3 or normalized.shape[2] != 3:
        print("⚠️ Imagen procesada no tiene 3 canales")

    return normalized, pixels_per_cm


def extract_sugarcane_features(image_path, pixels_per_cm=None):
    """
    Extrae características de la caña de azúcar con mejor detección de nudos
    """
    # Cargar imagen
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")

    # Si no se proporciona la relación píxeles/cm, calcularla
    if pixels_per_cm is None:
        preprocessed, pixels_per_cm = preprocess_image(image_path)

    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Umbralización adaptativa para segmentar mejor la caña
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2
    )

    # Operaciones morfológicas
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Encontrar el papel blanco primero (mayor contorno rectangular)
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    if not cnts:
        return {
            "nudos": 0,
            "entrenudos": 0,
            "distancia_entre_nudos_cm": 0.0,
            "largo_nudo_cm": 0.0,
            "grosor_cm": 0.0,
            "altura_con_curvatura_cm": 0.0,
            "altura_sin_curvatura_cm": 0.0,
            "es_cana_de_azucar": False,
        }

    # Encontrar el papel (contorno rectangular más grande)
    paper_contour = None
    max_area = 0

    for c in cnts:
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        if len(approx) == 4 and area > max_area:
            paper_contour = c
            max_area = area

    # Si no encontramos un contorno rectangular, usar el más grande
    if paper_contour is None:
        paper_contour = max(cnts, key=cv2.contourArea)

    # Crear una máscara para el área dentro del papel
    paper_mask = np.zeros_like(gray)
    cv2.drawContours(paper_mask, [paper_contour], -1, 255, -1)

    # Aplicar la máscara para buscar solo dentro del papel
    masked_thresh = cv2.bitwise_and(thresh, thresh, mask=paper_mask)

    # Aplicar operaciones morfológicas específicas para caña
    kernel_vertical = np.ones((11, 3), np.uint8)
    dilated = cv2.dilate(masked_thresh, kernel_vertical, iterations=1)

    # Detectar contornos dentro del papel
    inner_cnts = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    inner_cnts = inner_cnts[0] if len(inner_cnts) == 2 else inner_cnts[1]

    if not inner_cnts:
        return {
            "nudos": 0,
            "entrenudos": 0,
            "distancia_entre_nudos_cm": 0.0,
            "largo_nudo_cm": 0.0,
            "grosor_cm": 0.0,
            "altura_con_curvatura_cm": 0.0,
            "altura_sin_curvatura_cm": 0.0,
            "es_cana_de_azucar": False,
        }

    # Encontrar la caña de azúcar (objeto largo y vertical dentro del papel)
    cane_contour = None
    best_score = 0

    for c in inner_cnts:
        area = cv2.contourArea(c)
        if area < 500:  # Ignorar contornos muy pequeños
            continue

        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = h / w if w > 0 else 0

        # Puntaje basado en relación de aspecto y área
        # Preferimos objetos altos y delgados (cañas)
        if aspect_ratio > 1.5:  # Más alto que ancho
            score = area * aspect_ratio
            if score > best_score:
                cane_contour = c
                best_score = score

    # Si no encontramos un objeto que parezca caña, usar el contorno más grande
    if cane_contour is None and inner_cnts:
        filtered_cnts = [c for c in inner_cnts if cv2.contourArea(c) > 1000]
        if filtered_cnts:
            cane_contour = max(filtered_cnts, key=cv2.contourArea)

    if cane_contour is None:
        return {
            "nudos": 0,
            "entrenudos": 0,
            "distancia_entre_nudos_cm": 0.0,
            "largo_nudo_cm": 0.0,
            "grosor_cm": 0.0,
            "altura_con_curvatura_cm": 0.0,
            "altura_sin_curvatura_cm": 0.0,
            "es_cana_de_azucar": False,
        }

    # Obtener dimensiones de la caña
    rect = cv2.minAreaRect(cane_contour)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)

    # Calcular dimensiones
    (tl, tr, br, bl) = box

    # Ancho (grosor) y alto (largo) en píxeles
    width_px = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    height_px = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))

    # Convertir a centímetros
    grosor_cm = width_px / pixels_per_cm
    largo_cm = height_px / pixels_per_cm

    # Crear una máscara solo de la caña para análisis de nudos
    cane_mask = np.zeros_like(gray)
    cv2.drawContours(cane_mask, [cane_contour], -1, 255, -1)

    # Aplicar la máscara a la imagen umbralizada
    cane_thresh = cv2.bitwise_and(thresh, thresh, mask=cane_mask)

    # Detección de nudos usando transformación de distancia
    dist_transform = cv2.distanceTransform(cane_thresh, cv2.DIST_L2, 5)

    # Normalizar para visualización
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)

    # Encontrar máximos locales (posibles nudos)
    # Ajustar min_distance para mejor detección de nudos
    local_max = peak_local_max(
        dist_transform,
        min_distance=int(20 * pixels_per_cm / 10),  # Distancia proporcional a escala
        threshold_abs=0.2,  # Umbral absoluto para filtrar ruido
        labels=cane_thresh,
    )

    # Si detectamos demasiados "nudos", probablemente es ruido - filtrar
    if len(local_max) > 20:
        # Ordenar por valor de distancia y quedarse con los más prominentes
        values = [dist_transform[y, x] for y, x in local_max]
    sorted_indices = np.argsort(values)[::-1]  # Ordenar de mayor a menor
    local_max = local_max[sorted_indices[:10]]  # Quedarse con los 10 más prominentes

    # Estimación del número de nudos y entrenudos
    num_nudos = len(local_max)

    # Verificar si es una caña basado en características morfológicas
    is_cane = True

    # Si parece una caña, estimar el número de entrenudos
    num_entrenudos = max(0, num_nudos - 1) if is_cane else 0

    # Calcular distancia entre nudos
    if num_entrenudos > 0:
        # Ordenar nudos por altura (coordenada y)
        sorted_nodes = sorted(local_max, key=lambda p: p[0])

        # Calcular distancias entre nudos consecutivos
        distances = []
        for i in range(1, len(sorted_nodes)):
            y1, x1 = sorted_nodes[i - 1]
            y2, x2 = sorted_nodes[i]
            dist_px = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            distances.append(dist_px / pixels_per_cm)

        # Distancia promedio entre nudos
        distancia_nudos_cm = np.mean(distances) if distances else largo_cm / num_nudos
    else:
        distancia_nudos_cm = 0.0

    # Altura con curvatura (siguiendo el contorno)
    perimetro_px = cv2.arcLength(cane_contour, True)
    altura_con_curvatura_cm = perimetro_px / (2 * pixels_per_cm)

    # Altura sin curvatura (línea recta de extremo a extremo)
    x, y, w, h = cv2.boundingRect(cane_contour)
    altura_sin_curvatura_cm = h / pixels_per_cm

    # Resultados
    features = {
        "nudos": int(num_nudos),
        "entrenudos": int(num_entrenudos),
        "distancia_entre_nudos_cm": float(distancia_nudos_cm),
        "largo_nudo_cm": float(largo_cm / num_nudos if num_nudos > 0 else 0.0),
        "grosor_cm": float(grosor_cm),
        "altura_con_curvatura_cm": float(altura_con_curvatura_cm),
        "altura_sin_curvatura_cm": float(altura_sin_curvatura_cm),
        "es_cana_de_azucar": bool(is_cane),
    }

    return features

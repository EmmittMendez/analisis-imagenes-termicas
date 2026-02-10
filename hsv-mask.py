import cv2
import numpy as np
import os

# Carpetas
input_folder = '../imagenes-prueba'
output_folder = '../imagenes-prueba-mask'
os.makedirs(output_folder, exist_ok=True)

# Rango de azul en HSV
lower_blue = np.array([107, 50, 50])
upper_blue = np.array([130, 255, 255])

imagenes = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
total = len(imagenes)
last_percent = -1

# Procesar cada imagen en la carpeta
for idx, filename in enumerate(imagenes, 1):
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    if img is None:
        continue

    # Convertir a HSV y crear máscara para el fondo azul
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_fondo = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_sin_fondo = cv2.bitwise_not(mask_fondo)
    img_sin_fondo = cv2.bitwise_and(img, img, mask=mask_sin_fondo)

    # Guardar resultado
    out_path = os.path.join(output_folder, filename)
    cv2.imwrite(out_path, img_sin_fondo)

    # Mostrar progreso cada 5%
    percent = int((idx / total) * 100)
    if percent % 5 == 0 and percent != last_percent:
        print(f"Progreso: {percent}% ({idx} de {total} imágenes)")
        last_percent = percent

print('¡Procesamiento terminado!')
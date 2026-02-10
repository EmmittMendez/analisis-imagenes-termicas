import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern

# Carpetas
# input_folder = '../mask-imagenes-prueba'
input_folder = '../imagenes-prueba'
output_folder = '../imagenes-prueba-lbp'
os.makedirs(output_folder, exist_ok=True)

# Parámetros de LBP
radius = 1  # Radio del círculo
n_points = 8 * radius  # Número de puntos vecinos
method = 'uniform'  # Puede ser 'default', 'ror', 'uniform', 'var'

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        lbp = local_binary_pattern(img, n_points, radius, method)
        # Normalizar a 0-255 para guardar como imagen
        lbp_img = np.uint8(255 * lbp / lbp.max())
        out_name = f"{os.path.splitext(filename)[0]}_lbp.png"
        out_path = os.path.join(output_folder, out_name)
        cv2.imwrite(out_path, lbp_img)

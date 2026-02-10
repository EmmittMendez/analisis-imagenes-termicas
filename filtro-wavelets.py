import cv2
import numpy as np
import os
import pywt

# Carpetas
input_folder = '../imagenes-prueba'
output_folder = '../imagenes-wavelets'
os.makedirs(output_folder, exist_ok=True)

# Parámetros de la transformada wavelet
wavelet = 'db1'  # Daubechies 1 (Haar)
level = 1        # Nivel de descomposición

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # Normalizar a float
        img = np.float32(img) / 255.0
        # Descomposición wavelet
        coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
        cA, (cH, cV, cD) = coeffs
        # Escalar los coeficientes para visualización
        cA_img = np.uint8(255 * (cA - cA.min()) / (cA.max() - cA.min()))
        out_name = f"{os.path.splitext(filename)[0]}_wavelet_{wavelet}_A.png"
        out_path = os.path.join(output_folder, out_name)
        cv2.imwrite(out_path, cA_img)
        # Puedes guardar cH, cV, cD si lo deseas

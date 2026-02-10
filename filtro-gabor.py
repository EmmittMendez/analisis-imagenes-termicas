import cv2
import numpy as np
import os

# Carpetas
input_folder = '../imagenes-prueba'
output_folder = '../imagenes2-gabor'
os.makedirs(output_folder, exist_ok=True)

# Parámetros del banco de filtros de Gabor
ksize = 31
gamma = 0.5
psi = 0

# Múltiples orientaciones (en radianes)
orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 0°, 45°, 90°, 135°

# Múltiples escalas (wavelength)
lambdas = [4, 8, 16]  # Frecuencias: alta, media, baja

# Crear banco de filtros
gabor_filters = []
for theta in orientations:
    for lambd in lambdas:
        sigma = 0.56 * lambd  # Relación óptima
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, 
                                   lambd, gamma, psi, ktype=cv2.CV_32F)
        gabor_filters.append((kernel, theta, lambd))

# Procesar imágenes
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        # Aplicar todos los filtros y combinar respuestas
        responses = []
        for kernel, theta, lambd in gabor_filters:
            filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
            responses.append(filtered)
        
        # Opción 1: Magnitud máxima en cada píxel
        # gabor_result = np.max(np.array(responses), axis=0)
        # Calcular energía combinada
        gabor_result = np.sqrt(np.sum(np.array(responses)**2, axis=0))

        # Opción 2: Media de todas las respuestas
        # gabor_result = np.mean(np.array(responses), axis=0)
        
        # Normalizar a rango 0-255
        gabor_result = cv2.normalize(gabor_result, None, 0, 255, 
                                     cv2.NORM_MINMAX, cv2.CV_8U)
        
        out_path = os.path.join(output_folder, filename)
        cv2.imwrite(out_path, gabor_result)

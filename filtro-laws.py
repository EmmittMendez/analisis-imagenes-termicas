import cv2
import numpy as np
import os

# Carpetas
input_folder = '../imagenes-prueba'
output_folder = '../imagenes-laws'
os.makedirs(output_folder, exist_ok=True)

# Definir los kernels de Laws (5x5)
L5 = np.array([1, 4, 6, 4, 1])      # Level
E5 = np.array([-1, -2, 0, 2, 1])    # Edge
S5 = np.array([-1, 0, 2, 0, -1])    # Spot
R5 = np.array([1, -4, 6, -4, 1])    # Ripple
W5 = np.array([-1, 2, 0, -2, 1])    # Wave

# Lista de combinaciones de kernels (puedes agregar más si lo deseas)
kernels = [
    ('L5E5', np.outer(L5, E5)),
    ('E5L5', np.outer(E5, L5)),
    ('E5E5', np.outer(E5, E5)),
    ('S5S5', np.outer(S5, S5)),
    ('R5R5', np.outer(R5, R5)),
    ('W5W5', np.outer(W5, W5)),
]

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        for name, kernel in kernels:
            filtered = cv2.filter2D(img, -1, kernel)
            out_name = f"{os.path.splitext(filename)[0]}_laws_{name}.png"
            out_path = os.path.join(output_folder, out_name)
            cv2.imwrite(out_path, filtered)

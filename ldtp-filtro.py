import numpy as np
import cv2

def compute_frei_chen_masks():
    """
    Devuelve los 8 máscaras direccionales de Frei-Chen (3x3) y la máscara de segunda derivada Gaussiana.
    """
    sqrt2 = np.sqrt(2)
    # Frei-Chen 8 direcciones
    masks = [
        np.array([[1, sqrt2, 1], [0, 0, 0], [-1, -sqrt2, -1]]) / (2 * sqrt2),  # 0°
        np.array([[0, 1, sqrt2], [-1, 0, 1], [-sqrt2, -1, 0]]) / (2 * sqrt2),  # 45°
        np.array([[-1, 0, 1], [-sqrt2, 0, sqrt2], [-1, 0, 1]]) / (2 * sqrt2),  # 90°
        np.array([[-sqrt2, -1, 0], [1, 0, -1], [0, 1, sqrt2]]) / (2 * sqrt2),  # 135°
        np.array([[1, 0, -1], [sqrt2, 0, -sqrt2], [1, 0, -1]]) / (2 * sqrt2),  # 180°
        np.array([[0, -1, -sqrt2], [1, 0, -1], [sqrt2, 1, 0]]) / (2 * sqrt2),  # 225°
        np.array([[1, 0, -1], [sqrt2, 0, -sqrt2], [1, 0, -1]]) / (2 * sqrt2),  # 270°
        np.array([[sqrt2, 1, 0], [-1, 0, 1], [0, -1, -sqrt2]]) / (2 * sqrt2),  # 315°
    ]
    # Segunda derivada Gaussiana (Laplaciano)
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return masks, laplacian

def compute_ldtp(image, threshold=5, neighborhood=3, return_maps=False):
    """
    Calcula el descriptor LDTP para una imagen en escala de grises.
    :param image: np.ndarray, imagen en escala de grises.
    :param threshold: Umbral para la cuantización ternaria.
    :param neighborhood: Tamaño de vecindario (solo 3 soportado en esta versión).
    :param return_maps: Si True, retorna los mapas LDTPU y LDTPL.
    :return: histograma LDTP concatenado (LDTPU + LDTPL), (opcional) mapas LDTPU y LDTPL.
    """
    assert neighborhood == 3, "Solo neighborhood=3 soportado en esta versión."
    masks, laplacian = compute_frei_chen_masks()
    h, w = image.shape
    # Respuestas direccionales (8 mapas)
    responses = [cv2.filter2D(image.astype(np.float32), -1, k) for k in masks]
    responses = np.stack(responses, axis=-1)  # (h, w, 8)
    # Respuesta central (Laplaciano)
    center_response = cv2.filter2D(image.astype(np.float32), -1, laplacian)
    # Para cada píxel, asigna las 8 respuestas a los vecinos y la central al centro
    # Extrae 3x3 vecindarios
    pad = neighborhood // 2
    padded = np.pad(image, pad, mode='reflect')
    ldtpu = np.zeros((h, w, 9), dtype=np.uint8)
    ldtpl = np.zeros((h, w, 9), dtype=np.uint8)
    # Mapeo de vecinos: [0,1,2,3,4,5,6,7] alrededor, 8 centro
    # Orden: [arriba, arriba-der, der, abajo-der, abajo, abajo-izq, izq, arriba-izq, centro]
    # Para cada píxel, compara respuesta periférica con central
    for i in range(8):
        # Desplazamiento de vecino
        dy = [-1, -1, 0, 1, 1, 1, 0, -1]
        dx = [0, 1, 1, 1, 0, -1, -1, -1]
        resp = responses[:, :, i]
        # Mueve la respuesta periférica al lugar correspondiente
        shifted = np.roll(resp, shift=dy[i], axis=0)
        shifted = np.roll(shifted, shift=dx[i], axis=1)
        # Compara con central
        diff = shifted - center_response
        ldtpu[:, :, i] = (diff >= threshold).astype(np.uint8)
        ldtpl[:, :, i] = (diff <= -threshold).astype(np.uint8)
    # Centro: siempre 0 (no se codifica)
    ldtpu[:, :, 8] = 0
    ldtpl[:, :, 8] = 0
    # Codifica patrones binarios a enteros
    def patterns_to_codes(patterns):
        codes = np.zeros((h, w), dtype=np.uint16)
        for i in range(8):
            codes |= (patterns[:, :, i] << i)
        return codes
    codes_u = patterns_to_codes(ldtpu)
    codes_l = patterns_to_codes(ldtpl)
    # Histograma
    hist_u, _ = np.histogram(codes_u, bins=2**8, range=(0, 2**8))
    hist_l, _ = np.histogram(codes_l, bins=2**8, range=(0, 2**8))
    hist = np.concatenate([hist_u, hist_l])
    hist = hist.astype(np.float32) / hist.sum()  # Normaliza
    if return_maps:
        return hist, codes_u, codes_l
    return hist

# Ejemplo de uso
if __name__ == "__main__":
    # Carga imagen de ejemplo
    img = cv2.imread("ejemplo.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("No se encontró 'ejemplo.jpg'")
    ldtp_hist, ldtpu_map, ldtpl_map = compute_ldtp(img, threshold=5, return_maps=True)
    print("LDTP histogram shape:", ldtp_hist.shape)
    print("LDTP histogram (primeros 10 valores):", ldtp_hist[:10])
    # Visualiza mapas de patrones (opcional)
    cv2.imwrite("ldtp_upper.png", (ldtpu_map / ldtpu_map.max() * 255).astype(np.uint8))
    cv2.imwrite("ldtp_lower.png", (ldtpl_map / ldtpl_map.max() * 255).astype(np.uint8))
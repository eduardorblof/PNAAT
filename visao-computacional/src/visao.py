import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_images(images, titles, cmap=None, figsize=(15,5)):
    """Função auxiliar para exibir imagens lado a lado"""
    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i+1)
        if len(img.shape) == 2:  # Se for escala de cinza
            plt.imshow(img, cmap=cmap if cmap else 'gray')
        else:  # Se for colorida (BGR → RGB)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
    plt.show()

import urllib.request

url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg?"

resp = urllib.request.urlopen(url)

image = np.asarray(bytearray(resp.read()), dtype="uint8")

img = cv2.imdecode(image, cv2.IMREAD_COLOR)
 

# Exibindo para testar

from matplotlib import pyplot as plt
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

blur = cv2.blur(img, (5,5))
 
# 2. Filtro Gaussiano

gaussian = cv2.GaussianBlur(img, (5,5), 0)
 
# 3. Conversão para escala de cinza

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
# 4. Filtro Sobel

sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
sobel = cv2.magnitude(sobelx, sobely)
 
# 5. Detector de Bordas Canny

canny = cv2.Canny(gray, 100, 200)
 
show_images([blur, gaussian, sobel, canny],
            ["Média (Blur)", "Gaussiano", "Sobel", "Canny"])


# --- Segmentação de Imagens ---
 
# 1.1. Thresholding Simples
# Define um valor fixo de limiar (127). Pixels acima dele viram branco, abaixo viram preto.
ret, thresh_bin = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
 
# 1.2. Thresholding Adaptativo
# Calcula limiares localmente em diferentes regiões da imagem,
# útil quando a iluminação não é uniforme.
thresh_adapt = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # usa média ponderada gaussiana
    cv2.THRESH_BINARY,
    11, 2
)
 
show_images(
    [gray, thresh_bin, thresh_adapt],
    ["Original (Gray)", "Threshold Binário", "Threshold Adaptativo"]
)
 
# 2. Segmentação por K-means
# Reorganiza a imagem em clusters de cores semelhantes.
 
# Pré-processamento: transforma a imagem em um array 2D de pixels
Z = img.reshape((-1, 3))
Z = np.float32(Z)
 
# Critério de parada do algoritmo e número de clusters (K)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3  # número de grupos de cor
_, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
 
# Reconstruindo a imagem segmentada
centers = np.uint8(centers)
segmented = centers[labels.flatten()].reshape(img.shape)
 
# Exibindo original e segmentada
show_images(
    [img, segmented],
    ["Imagem Original", "Segmentação K-means (K=3)"]
)

# --- 4. Extração de Features ---
 
# 1. Detecção de Cantos (Shi-Tomasi)
# Detecta pontos onde a variação de intensidade é alta em duas direções.
corners = cv2.goodFeaturesToTrack(gray,
                                  maxCorners=100,     # número máximo de cantos
                                  qualityLevel=0.01,  # qualidade mínima do ponto
                                  minDistance=10)     # distância mínima entre cantos
corners = np.int0(corners)
 
# Criamos uma cópia da imagem para desenhar os cantos detectados
img_corners = img.copy()
for c in corners:
    x, y = c.ravel()
    cv2.circle(img_corners, (x, y), 4, (0, 0, 255), -1)  # círculo vermelho
 
# 2. ORB (Oriented FAST and Rotated BRIEF)
# Detecta pontos-chave e cria descritores para comparação entre imagens.
orb = cv2.ORB_create()
kp, des = orb.detectAndCompute(gray, None)
 
# Desenhando os keypoints encontrados na imagem
img_orb = cv2.drawKeypoints(img, kp, None,
                            color=(0, 255, 0),   # verde
                            flags=0)
 
show_images([img_corners, img_orb],
            ["Cantos (Shi-Tomasi)", "Keypoints (ORB)"])





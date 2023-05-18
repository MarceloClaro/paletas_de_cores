import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def extrair_paleta(imagem, n_cores):
    pixels = imagem.reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels.astype(float), n_cores, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(imagem.shape)
    return centers, segmented_image

st.title('Analisador de Paleta de Cores de Imagens de Solo')
n_cores = st.sidebar.slider('Número de cores na paleta', min_value=1, max_value=20, value=5, step=1)
imagem_up = st.sidebar.file_uploader('Carregar Imagem de Solo', type=['png', 'jpg', 'jpeg'])

if imagem_up is not None:
    imagem = Image.open(imagem_up)
    imagem_cv = np.array(imagem)

    # Verificar se a imagem está em escala de cinza
    if len(imagem_cv.shape) != 3 or imagem_cv.shape[2] != 3:
        st.error("Por favor, carregue uma imagem colorida. A imagem carregada parece estar em escala de cinza.")
    else:
        st.image(imagem_cv, caption='Imagem Original', use_column_width=True)

        centers, segmented_image = extrair_paleta(imagem_cv, n_cores)

        # Convertendo a imagem segmentada de volta para uma imagem de 8 bits
        segmented_image = cv2.convertScaleAbs(segmented_image)

        st.image(segmented_image, caption='Imagem Segmentada', use_column_width=True)

        st.subheader('Paleta de Cores:')
        plt.figure(figsize=(5, 2))
        plt.imshow([centers.astype(int)], aspect='auto')
        plt.axis('off')
        st.pyplot(plt)

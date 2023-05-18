import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Função para extrair paleta
def extrair_paleta(imagem, n_cores):
    imagem = cv2.resize(imagem, (50, 50), interpolation=cv2.INTER_AREA)
    imagem = imagem.reshape((-1, 3))  # Converta para matriz 2D

    # Verifique se n_cores é maior que 0
    assert n_cores > 0, "n_cores deve ser maior que 0"

    # Converta para float
    pixels = np.float32(imagem)

    # Critérios de parada para o algoritmo kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Aplicar kmeans
    _, labels, centers = cv2.kmeans(pixels, n_cores, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Converta de volta para valores de 8 bits
    centers = np.uint8(centers)

    # Mapear os rótulos para os centros
    segmented_image = centers[labels.flatten()]

    # Reshape de volta para a imagem original
    segmented_image = segmented_image.reshape(imagem.shape)

    return centers, segmented_image

# Código principal Streamlit
st.title('Analisador de Paleta de Cores')

imagem_up = st.file_uploader('Por favor, faça upload de uma imagem')

if imagem_up is not None:
    imagem = Image.open(imagem_up)
    imagem_cv = np.array(imagem)[:, :, ::-1]  # Converta RGB para BGR

    n_cores = st.slider('Número de cores para extração', min_value=1, max_value=10, value=5)

    centers, segmented_image = extrair_paleta(imagem_cv, n_cores)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Imagem original
    ax[0].imshow(cv2.cvtColor(imagem_cv, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Imagem Original')
    ax[0].axis('off')

    # Imagem segmentada
    ax[1].imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Imagem Segmentada')
    ax[1].axis('off')

    st.pyplot(fig)

    st.markdown('## Paleta de cores')
    st.image(centers[:, ::-1], caption='Cores extraídas', width=50)

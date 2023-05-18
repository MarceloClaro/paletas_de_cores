import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import numpy as np


def extrair_paleta(imagem, n_cores):
    # Redimensionar a imagem para acelerar a extração de cores
    imagem_reduzida = cv2.resize(imagem, (50, 50), interpolation=cv2.INTER_AREA)

    # Redefinir a imagem para um array 2D de pixels
    pixels = imagem_reduzida.reshape(-1, 3)

    # Executar k-means para encontrar as cores mais dominantes
    kmeans = cv2.kmeans(pixels.astype(float), n_cores, None,
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
                        1, cv2.KMEANS_RANDOM_CENTERS)

    # Retornar as cores encontradas
    return kmeans[2].astype(int)


st.title('Análise da Paleta de Cores do Solo')

imagem_upload = st.file_uploader('Faça upload da imagem do solo', type=['png', 'jpg'])

if imagem_upload is not None:
    imagem = Image.open(imagem_upload)
    st.image(imagem, caption='Imagem do solo carregada.', use_column_width=True)

    # Converter a imagem para o espaço de cores RGB (OpenCV usa BGR por padrão)
    imagem_cv = cv2.cvtColor(np.array(imagem), cv2.COLOR_RGB2BGR)

    # Extrair a paleta de cores
    n_cores = st.sidebar.slider('Número de cores para extrair', min_value=2, max_value=16, value=5)
    paleta_cores = extrair_paleta(imagem_cv, n_cores)

    # Mostrar a paleta de cores
    st.sidebar.header('Paleta de cores')
    for i, cor in enumerate(paleta_cores):
        st.sidebar.markdown(f'### Cor {i+1}')
        st.sidebar.markdown(f'RGB: {tuple(cor)}')
        st.sidebar.markdown(f'#### &#9608;', unsafe_allow_html=True,
                            color=f'rgb({cor[0]},{cor[1]},{cor[2]})')

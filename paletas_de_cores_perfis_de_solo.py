# -*- coding: utf-8 -*-
"""Script para gerar uma paleta de cores a partir de uma imagem"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import cv2
import streamlit as st
from PIL import Image

class Canvas():
    """
    Definição do objeto canvas

    Parâmetros
    ----------
    src : array_like
        a imagem de origem que você deseja transformar em uma paleta de cores
    nb_clusters :
        número de cores que você deseja manter
    pixel_size: integer, optional, default 4000
        tamanho em px da maior dimensão do canvas de saída
    """

    def __init__(self, src, nb_color, pixel_size=4000):
        self.src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        self.nb_color = nb_color
        self.tar_width = pixel_size
        self.colormap = []

    # Função principal
    def generate(self):
        im_source = self.resize()
        clean_img = self.cleaning(im_source)
        width, height, depth = clean_img.shape
        clean_img = np.array(clean_img, dtype="uint8")/255
        quantified_image, colors = self.quantification(clean_img)
        segmented_image = self.recreate_image(colors, quantified_image, width, height)
        return segmented_image, colors

    # Redimensiona a imagem para corresponder à largura alvo e respeitar a proporção da imagem
    def resize(self):
        (height, width) = self.src.shape[:2]
        if height > width: # modo retrato
            dim = (int(width * self.tar_width / float(height)), self.tar_width)
        else:
            dim = (self.tar_width, int(height * self.tar_width / float(width)))
        return cv2.resize(self.src, dim, interpolation=cv2.INTER_AREA)

    # Redução de ruído, operações morfológicas, abrindo e fechando
    def cleaning(self, picture):
        clean_pic = cv2.fastNlMeansDenoisingColored(picture,None,10,10,7,21)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        clean_pic = cv2.morphologyEx(clean_pic, cv2.MORPH_OPEN, kernel, cv2.BORDER_REPLICATE)
        clean_pic = cv2.morphologyEx(clean_pic, cv2.MORPH_CLOSE, kernel, cv2.BORDER_REPLICATE)
        return clean_pic

    # Retorna a imagem K-means
    def quantification(self, picture):
        width, height, dimension = tuple(picture.shape)
        image_array = np.reshape(picture, (width * height, dimension))
        image_array_sample = shuffle(image_array, random_state=0)[:1000]
        kmeans = KMeans(n_clusters=self.nb_color, random_state=42).fit(image_array_sample)
        labels = kmeans.predict(image_array)
        return labels, kmeans.cluster_centers_

    # Cria a imagem a partir de uma lista de cores, rótulos e tamanho da imagem
    def recreate_image(self, codebook, labels, width, height):
        d = codebook.shape[1]
        image = np.zeros((width, height, d))
        label_idx = 0
        for i in range(width):
            for j in range(height):
                image[i][j] = codebook[labels[label_idx]]
                label_idx += 1
        return image

# Parte Streamlit
st.title('Gerador de Paleta de Cores')
uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption='Imagem Carregada', use_column_width=True)

    nb_color = st.slider('Escolha o número de cores', min_value=2, max_value=80, value=5, step=1)

    if st.button('Gerar'):
        pixel_size = st.slider('Escolha o tamanho do pixel', min_value=500, max_value=4000, value=4000, step=100)
        canvas = Canvas(image, nb_color, pixel_size)
        segmented_image, colors = canvas.generate()

        # Convertendo a imagem segmentada para RGB antes de mostrar
        segmented_image_rgb = cv2.cvtColor(np.uint8(segmented_image * 255), cv2.COLOR_BGR2RGB)

        # Mostrando a imagem segmentada
        st.image(segmented_image_rgb, caption='Imagem Segmentada', use_column_width=True)

        # Mostra a paleta de cores da imagem segmentada com suas numerações correspondentes
        plt.figure(figsize=(5, 2), dpi=100)
        plt.axis('off')
        plt.title('Paleta de Cores')
        for sp in plt.gca().spines.values():
            sp.set_visible(False)
        plt.xticks([])
        plt.yticks([])
        for idx, color in enumerate(colors):
            plt.bar(idx+1, 1, color=color, label=idx+1)
        plt.legend(loc='upper left')
        st.pyplot(plt.gcf())

        # Converte a imagem segmentada para bytes
        segmented_image_bytes = cv2.imencode('.jpg', segmented_image_rgb)[1].tobytes()

        # Oferece a opção de baixar a imagem segmentada
        st.download_button('Baixar Imagem Segmentada', data=segmented_image_bytes, file_name='segmentada.jpg', mime='image/jpeg')

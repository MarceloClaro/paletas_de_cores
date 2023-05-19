# -*- coding: utf-8 -*-
"""Script para gerar uma paleta de cores a partir de uma imagem"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import cv2
import ntpath
import streamlit as st
import base64

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
        
        self.src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)  # Converte de BGR para RGB
        self.nb_color = nb_color
        self.tar_width = pixel_size
        self.colormap = []

    def generate(self):
        """Função principal"""
        im_source = self.resize()
        clean_img = self.cleaning(im_source)
        width, height, depth = clean_img.shape
        clean_img = np.array(clean_img, dtype="uint8")/255
        quantified_image, colors = self.quantification(clean_img)
        canvas = np.ones(quantified_image.shape[:3], dtype="uint8")*255

        for ind, color in enumerate(colors):
            self.colormap.append([int(c*255) for c in color])
            mask = cv2.inRange(quantified_image, color, color)
            cnts = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            for contour in cnts:
                _, _, width_ctr, height_ctr = cv2.boundingRect(contour)
                if width_ctr > 10 and height_ctr > 10 and cv2.contourArea(contour, True) < -100:
                    cv2.drawContours(canvas, [contour], -1, 0, 1)
                    #Add label
                    txt_x, txt_y = contour[0][0]
                    cv2.putText(canvas, '{:d}'.format(ind + 1),
                                (txt_x, txt_y + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)

        return canvas, colors, quantified_image

    def resize(self):
        """Redimensiona a imagem para corresponder à largura alvo e respeitar a proporção da imagem"""
        (height, width) = self.src.shape[:2]
        if height > width: # modo retrato
            dim = (int(width * self.tar_width / float(height)), self.tar_width)
        else:
            dim = (self.tar_width, int(height * self.tar_width / float(width)))
        return cv2.resize(self.src, dim, interpolation=cv2.INTER_AREA)

    def cleaning(self, picture):
        """Redução de ruído, operações morfológicas, abrindo e fechando"""
        clean_pic = cv2.fastNlMeansDenoisingColored(picture,None,10,10,7,21)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8

, 8))
        clean_pic = cv2.morphologyEx(clean_pic, cv2.MORPH_OPEN, kernel, cv2.BORDER_REPLICATE)
        clean_pic = cv2.morphologyEx( clean_pic, cv2.MORPH_CLOSE, kernel, cv2.BORDER_REPLICATE)
        return clean_pic

    def quantification(self, picture):
        """Retorna a imagem K-means"""
        width, height, dimension = tuple(picture.shape)
        image_array = np.reshape(picture, (width * height, dimension))
        image_array_sample = shuffle(image_array, random_state=0)[:1000]
        kmeans = KMeans(n_clusters=self.nb_color, random_state=42).fit(image_array_sample)
        labels = kmeans.predict(image_array)
        new_img = self.recreate_image(kmeans.cluster_centers_, labels, width, height)
        return new_img, kmeans.cluster_centers_

    def recreate_image(self, codebook, labels, width, height):
        """Cria a imagem a partir de uma lista de cores, rótulos e tamanho da imagem"""
        vfunc = lambda x: codebook[labels[x]]
        out = vfunc(np.arange(width*height))
        return np.resize(out, (width, height, codebook.shape[1]))

# Parte Streamlit
st.title('Gerador de Paleta de Cores')
uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converte de BGR para RGB
    st.image(image, caption='Imagem Carregada', use_column_width=True)

    nb_color = st.slider('Escolha o número de cores', min_value=2, max_value=80, value=5, step=1)

    if st.button('Gerar'):
        pixel_size = st.slider('Escolha o tamanho do pixel', min_value=500, max_value=4000, value=4000, step=100)
        canvas = Canvas(image, nb_color, pixel_size)
        result, colors, segmented_image = canvas.generate()

        # Mostrando a imagem resultante
        st.image(result, caption='Imagem Resultante', use_column_width=True)

        # Mostrando a imagem segmentada
        st.image(segmented_image, caption='Imagem Segmentada', use_column_width=True)

        # Fornecendo a opção de download para a imagem resultante
        is_success_result, im_buf_arr_result = cv2.imencode(".jpg", result)
        byte_im_result = im_buf_arr_result.tobytes()
        b64_result = base64.b64encode(byte_im_result).decode()
        link_result = f'<a href="data:file/jpg;base64,{b64_result}" download="result.jpg">Baixar imagem resultante em jpg</a>'
        st.markdown(link_result, unsafe_allow_html=True)

        # Fornecendo a opção de download para a imagem segmentada
        is_success_segmented, im_buf_arr_segmented = cv2.imencode(".jpg", segmented_image)
        byte_im_segmented = im_buf_arr_segmented.tobytes()
        b64_segmented = base64.b64encode(byte_im_segmented).decode()
        link_segmented = f'<a href="data:file/jpg;base64,{b64_segmented}" download="segmented.jpg">Baixar imagem segmentada em jpg</

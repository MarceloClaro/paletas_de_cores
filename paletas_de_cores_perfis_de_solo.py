# -*- coding: utf-8 -*-
"""Script para gerar uma paleta de cores a partir de uma imagem"""

# Importando as bibliotecas necessárias
import numpy as np  # Biblioteca para manipulação de arrays
from sklearn.cluster import KMeans  # Método de agrupamento para extração de cores
from sklearn.utils import shuffle  # Função para embaralhar os dados
import cv2  # Biblioteca para manipulação de imagens
import streamlit as st  # Biblioteca para criar aplicações web com Python

# Definição da classe Canvas
class Canvas():
    def __init__(self, src, nb_color, pixel_size=4000):
        # A imagem original, número de cores e tamanho do pixel são definidos no construtor
        self.src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)  # Converte a imagem de BGR para RGB
        self.nb_color = nb_color  # Número de cores a serem extraídas da imagem
        self.tar_width = pixel_size  # Tamanho do pixel na imagem redimensionada
        self.colormap = []  # Lista para guardar as cores extraídas

    # Método para gerar a paleta de cores
    def generate(self):
        im_source = self.resize()  # Redimensiona a imagem
        clean_img = self.cleaning(im_source)  # Limpa a imagem (ruído)
        width, height, depth = clean_img.shape  # Obtém as dimensões da imagem
        clean_img = np.array(clean_img, dtype="uint8") / 255  # Normaliza a imagem
        # Aplica a quantização (redução de cores) na imagem
        quantified_image, colors = self.quantification(clean_img)
        # Cria um novo canvas em branco
        canvas = np.ones(quantified_image.shape[:2], dtype="uint8") * 255

        # Para cada cor encontrada
        for ind, color in enumerate(colors):
            # Adiciona a cor à paleta
            self.colormap.append([int(c * 255) for c in color])
            # Cria uma máscara para a cor atual
            mask = cv2.inRange(quantified_image, color, color)
            # Encontra os contornos na máscara
            cnts = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            # Para cada contorno encontrado
            for contour in cnts:
                _, _, width_ctr, height_ctr = cv2.boundingRect(contour)
                # Se o contorno é suficientemente grande
                if width_ctr > 10 and height_ctr > 10 and cv2.contourArea(contour, True) < -100:
                    # Desenha o contorno no canvas
                    cv2.drawContours(canvas, [contour], -1, (0, 0, 0), 1)
                    # Desenha o índice da cor no canvas
                    txt_x, txt_y = contour[0][0]
                    cv2.putText(canvas, '{:d}'.format(ind + 1), (txt_x, txt_y + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return canvas, colors, quantified_image

    # Método para redimensionar a imagem
    def resize(self):
        im = self.src
        width = im.shape[1]
        height = im.shape[0]
        ratio = width / height
        target_width = self.tar_width
        target_height = int(target_width / ratio)
        return cv2.resize(im, (target_width, target_height))

    # Método para limpar ruídos na imagem
    def cleaning(self, pic):
        clean_pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        clean_pic = cv2.medianBlur(clean_pic, 7)
        _, clean_pic = cv2.threshold(clean_pic, 250, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((5, 5), np.uint8)
        img_erosion = cv2.erode(clean_pic, kernel, iterations=1)
        img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
        return img_dilation

    # Método para quantização da imagem
    def quantification(self, picture):
        width, height, depth = picture.shape
        flattened = np.reshape(picture, (width * height, depth))
        sample = shuffle(flattened)[:1000]
        kmeans = KMeans(n_clusters=self.nb_color).fit(sample)
        labels = kmeans.predict(flattened)
        new_img = self.recreate_image(kmeans.cluster_centers_, labels, width, height)
        return new_img, kmeans.cluster_centers_

    # Método para recriar a imagem
    def recreate_image(self, codebook, labels, width, height):
        vfunc = lambda x: codebook[labels[x]]
        out = vfunc(np.arange(width * height))
        return np.resize(out, (width, height, codebook.shape[1]))

# Início do script
st.title('Gerador de Paleta de Cores')
uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "png"])

if uploaded_file is not None:
    # Lê o arquivo de imagem carregado
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converte a imagem de BGR para RGB
    st.image(image, caption='Imagem Carregada', use_column_width=True)

    # Escolhe o número de cores para a paleta
    nb_color = st.slider('Escolha o número de cores', min_value=2, max_value=80, value=5, step=1)

    if st.button('Gerar'):
        pixel_size = st.slider('Escolha o tamanho do pixel', min_value=500, max_value=4000, value=4000, step=100)
        canvas = Canvas(image, nb_color, pixel_size)
        result, colors, segmented_image = canvas.generate()

        # Converter imagem segmentada para np.uint8
        segmented_image = (segmented_image * 255).astype(np.uint8)
        
        # Agora converta de BGR para RGB
        segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

        # Mostra a imagem resultante
        st.image(result, caption='Imagem Resultante', use_column_width=True)
        # Mostra a imagem segmentada
        st.image(segmented_image, caption='Imagem Segmentada', use_column_width=True)

        # Converte a imagem resultante para bytes
        result_bytes = cv2.imencode('.jpg', result)[1].tobytes()
        # Oferece a opção de baixar a imagem resultante
        st.download_button('Baixar Imagem Resultante', data=result_bytes, file_name='resultado.jpg', mime='image/jpeg')

        # Converte a imagem segmentada para bytes
        segmented_image_bytes = cv2.imencode('.jpg', segmented_image)[1].tobytes()
        # Oferece a opção de baixar a imagem segmentada
        st.download_button('Baixar Imagem Segmentada', data=segmented_image_bytes, file_name='segmentada.jpg', mime='image/jpeg')

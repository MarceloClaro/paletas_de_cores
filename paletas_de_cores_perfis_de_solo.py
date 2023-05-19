#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script para gerar uma paleta de cores a partir de uma imagem"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import cv2
import streamlit as st

class Canvas():
    """
    Definição do objeto Canvas

    Parâmetros
    ----------
    uploaded_file : objeto UploadedFile
        a imagem de origem que você quer transformar em uma paleta de cores
    nb_clusters :
        número de cores que você quer manter
    plot : boolean, opcional
        Se você quer ou não plotar os resultados
    save : boolean, opcional
        Se você quer ou não salvar os resultados
    pixel_size: inteiro, opcional, padrão 4000
        tamanho em pxl da maior dimensão do canvas de saída    
    """

    def __init__(self, uploaded_file, nb_color, plot=False, save=True, pixel_size=4000):
        
        self.namefile = uploaded_file.name.split(".")[0]
        self.src = cv2.cvtColor(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1), cv2.COLOR_BGR2RGB)
        self.nb_color = nb_color
        self.plot = plot
        self.save = save
        self.tar_width = pixel_size
        self.colormap = []

    def generate(self):
        im_source = self.resize()
        clean_img = self.cleaning(im_source)
        width, height, depth = clean_img.shape
        clean_img = np.array(clean_img, dtype="uint8")/255
        quantified_image, colors = self.quantification(clean_img)
        canvas = np.ones(quantified_image.shape[:3], dtype="uint8")*255

        for ind, color in enumerate(colors):
            self.colormap.append([int(c*255) for c in color])
            lower_color = np.array([int(c*255) for c in color])
            upper_color = np.array([int(c*255) for c in color])
            mask = cv2.inRange(quantified_image, lower_color, upper_color)
            cnts = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            for contour in cnts:
                _, _, width_ctr, height_ctr = cv2.boundingRect(contour)
                if width_ctr > 10 and height_ctr > 10 and cv2.contourArea(contour, True) < -100:
                    cv2.drawContours(canvas, [contour], -1, 0, 1)
                    txt_x, txt_y = contour[0][0]
                    cv2.putText(canvas, '{:d}'.format(ind + 1),
                                (txt_x, txt_y + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)

        if self.save:
            result_img = cv2.cvtColor((quantified_image*255).astype(np.uint8), cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{self.namefile}-result.png", result_img)
            cv2.imwrite(f"{self.namefile}-canvas.png", canvas)
        return result_img, colors, canvas

    def resize(self):
        (height, width) = self.src.shape[:2]
        if height > width:
            dim = (int(width * self.tar_width / float(height)), self.tar_width)
        else:
            dim = (self.tar_width, int(height * self.tar_width / float(width)))
        return cv2.resize(self.src, dim, interpolation=cv2.INTER_LINEAR)

    def cleaning(self, img):
        return cv2.bilateralFilter(img, 5, 50, 50)

    def quantification(self, img):
        # create a random subsample of your image to get the main colors
        # because the KMeans algorithm is costly in term of computation
        w, h, d = img.shape
        image_array = np.reshape(img, (w * h, d))
        image_array_sample = shuffle(image_array, random_state=0)[:1000]
        kmeans = KMeans(n_clusters=self.nb_color, random_state=0).fit(image_array_sample)

        # replace the color of the original image by the centroid color of the closest cluster
        labels = kmeans.predict(image_array)
        return np.reshape(labels, (w, h)), kmeans.cluster_centers_

def main():
    st.title('Paletas de Cores - Perfis de Solo')
    st.write("Selecione uma imagem para gerar a paleta de cores.")
    uploaded_file = st.file_uploader("Escolha uma imagem", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        nb_color = st.slider('Número de cores na paleta', min_value=2, max_value=80, value=5, step=1)
        canvas = Canvas(uploaded_file, nb_color)
        result, colors, canvas_image = canvas.generate()
        st.image(result, caption='Imagem Resultante', use_column_width=True)
        st.image(canvas_image, caption='Canvas Resultante', use_column_width=True)

if __name__ == "__main__":
    main()

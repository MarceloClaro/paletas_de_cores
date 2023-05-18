#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script para gerar uma imagem segmentada a partir de uma foto"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import cv2
import ntpath
import streamlit as st

class Canvas():
    def __init__(self, path_pic, nb_color, plot=False, save=True, pixel_size=4000):
        self.namefile = ntpath.basename(path_pic).split(".")[0]
        self.src = cv2.cvtColor(cv2.imread(path_pic), cv2.COLOR_BGR2RGB)
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
            mask = cv2.inRange(quantified_image, color, color)
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
            cv2.imwrite(f"{self.namefile}-result.png",
                        cv2.cvtColor(quantified_image.astype('float32')*255, cv2.COLOR_BGR2RGB))
            cv2.imwrite(f"{self.namefile}-canvas.png", canvas)
        return cv2.cvtColor(quantified_image.astype('float32')*255, cv2.COLOR_BGR2RGB), colors, canvas

    def resize(self):
        (height, width) = self.src.shape[:2]
        if height > width:
            dim = (int(width * self.tar_width / float(height)), self.tar_width)
        else:
            dim = (self.tar_width, int(height * self.tar_width / float(width)))
        return cv2.resize(self.src, dim, interpolation=cv2.INTER_AREA)

    def cleaning(self, picture):
        clean_pic = cv2.fastNlMeansDenoisingColored(picture,None,10,10,7,21)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        clean_pic = cv2.morphologyEx(clean_pic, cv2.MORPH_OPEN, kernel, cv2.BORDER_REPLICATE)
        clean_pic = cv2.morphologyEx( clean_pic, cv2.MORPH_CLOSE, kernel, cv2.BORDER_REPLICATE)
        return clean_pic

    def quantification(self, picture):
        width, height, dimension = tuple(picture.shape)
        image_array = np.reshape(picture, (width * height, dimension))
        image_array_sample = shuffle(image_array, random_state=0)[:1000]
        kmeans = KMeans(n_clusters=self.nb_color, random_state=42).fit(image_array_sample)
        labels = kmeans.predict(image_array)
        new_img = self.recreate_image(kmeans.cluster_centers_, labels, width, height)
        return new_img, kmeans.cluster_centers_

    def recreate_image(self, codebook, labels, width, height):
        vfunc = lambda x: codebook[labels[x]]
        out = vfunc(np.arange(width*height))
        return np.resize(out, (width, height, codebook.shape[1]))

    def display_colormap(self):
        fig, ax = plt.subplots(1, 1, figsize=(5, 2), constrained_layout=True)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        bars = ax.bar(range(len(self.colormap)), [2]*len(self.colormap), color=np.array(self.colormap)/255)
        for idx, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width()/2, 1, f'{idx+1}', ha='center', va='bottom')
        return fig

def main():
    path_pic = st.file_uploader("Escolha uma imagem", type=['png', 'jpg', 'jpeg'])
    if path_pic is not None:
        nb_color = st.slider('Número de cores na paleta', min_value=2, max_value=20, value=5, step=1)
        canvas = Canvas(path_pic, nb_color)
        result, colors, canvas_image = canvas.generate()
        st.image(result, caption='Imagem Resultante', use_column_width=True)
        colormap_fig = canvas.display_colormap()
        st.pyplot(colormap_fig)
        st.image(canvas_image, caption='Imagem segmentada', use_column_width=True)

if __name__ == "__main__":
    main()

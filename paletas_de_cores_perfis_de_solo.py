import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import cv2
import streamlit as st

class Canvas():
    """
    Definição do objeto canvas
    """

    def __init__(self, image, nb_color):
        self.src = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.nb_color = nb_color

    def generate(self):
        """Função principal"""
        quantified_image, colors = self.quantification(self.src)
        return quantified_image

    def quantification(self, picture):
        """Retorna a imagem K-mean"""
        width, height, dimension = tuple(picture.shape)
        image_array = np.reshape(picture, (width * height, dimension))
        image_array_sample = shuffle(image_array, random_state=0)[:1000]
        kmeans = KMeans(n_clusters=self.nb_color, random_state=42).fit(image_array_sample)
        labels = kmeans.predict(image_array)
        new_img = self.recreate_image(kmeans.cluster_centers_, labels, width, height)
        return new_img, kmeans.cluster_centers_

    def recreate_image(self, codebook, labels, width, height):
        """Cria a imagem a partir de uma lista de cores, rótulos e tamanho da imagem."""
        vfunc = lambda x: codebook[labels[x]]
        out = vfunc(np.arange(width*height))
        return np.resize(out, (width, height, codebook.shape[1]))

if __name__ == "__main__":
    st.title("Transforme a sua imagem em um canvas")

    uploaded_file = st.file_uploader("Faça upload da sua imagem", type=['png', 'jpg', 'jpeg'])
    nb_color = st.slider('Escolha o número de cores', min_value=1, max_value=255, value=5, step=1)

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        if st.button('Gerar'):
            canvas = Canvas(image, nb_color)
            result = canvas.generate()
            # Aqui você pode fazer o que quiser com a imagem resultante

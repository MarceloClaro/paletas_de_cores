import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import cv2
import streamlit as st
import ntpath


class Canvas():
    """
    Definição do objeto canvas
    """

    def __init__(self, image, nb_color):
        self.src = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.nb_color = nb_color

    def generate(self):
        """Função principal"""
        im_source = self.resize()
        clean_img = self.cleaning(im_source)
        quantified_image, colors = self.quantification(clean_img)
        return quantified_image, colors

    def resize(self):
        """Redimensiona a imagem para corresponder à largura-alvo e respeitar a proporção da imagem"""
        (height, width) = self.src.shape[:2]
        if height > width:  # modo retrato
            dim = (int(width * self.tar_width / float(height)), self.tar_width)
        else:
            dim = (self.tar_width, int(height * self.tar_width / float(width)))
        return cv2.resize(self.src, dim, interpolation=cv2.INTER_AREA)

    def cleaning(self, picture):
        """Redução de ruído, operações Morphomat, abertura e depois fechamento"""
        clean_pic = cv2.fastNlMeansDenoisingColored(picture, None, 10, 10, 7, 21)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        clean_pic = cv2.morphologyEx(clean_pic, cv2.MORPH_OPEN, kernel, cv2.BORDER_REPLICATE)
        clean_pic = cv2.morphologyEx(clean_pic, cv2.MORPH_CLOSE, kernel, cv2.BORDER_REPLICATE)
        return clean_pic

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
            result, colors = canvas.generate()
            # Aqui você pode fazer o que quiser com a imagem resultante e as cores

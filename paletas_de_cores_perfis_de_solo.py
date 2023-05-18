import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import cv2
import ntpath
import streamlit as st

class Canvas():
    """
    Definição do objeto canvas
    """
    def __init__(self, path_pic, nb_color, plot=False, save=True, pixel_size=4000):
        
        self.namefile = ntpath.basename(path_pic).split(".")[0]
        self.src = cv2.cvtColor(cv2.imread(path_pic), cv2.COLOR_BGR2RGB)
        self.nb_color = nb_color
        self.plot = plot
        self.save = save
        self.tar_width = pixel_size
        self.colormap = []

    def generate(self):
        """
        Função principal para gerar a imagem com cores reduzidas e suas bordas.
        """
        im_source = self.resize()
        clean_img = self.cleaning(im_source)
        quantified_image, colors = self.quantification(clean_img)
        edges = self.apply_canny(quantified_image)
        inverted_edges = self.invert_colors(edges)
        st.image(inverted_edges, caption='Imagem com cores invertidas.', use_column_width=True)

        if self.save:
            cv2.imwrite(f"./outputs/{self.namefile}-result.png",
                        cv2.cvtColor(quantified_image.astype('float32')*255, cv2.COLOR_BGR2RGB))
            cv2.imwrite(f"./outputs/{self.namefile}-edges.png", inverted_edges)

        return inverted_edges

    def resize(self):
        """
        Redimensiona a imagem para corresponder à largura alvo e respeitar a proporção da imagem.
        """
        (height, width) = self.src.shape[:2]
        if height > width: # modo retrato
            dim = (int(width * self.tar_width / float(height)), self.tar_width)
        else:
            dim = (self.tar_width, int(height * self.tar_width / float(width)))
        return cv2.resize(self.src, dim, interpolation=cv2.INTER_AREA)

    def cleaning(self, picture):
        """
        Redução de ruído, operações morfológicas, abertura e fechamento.
        """
        clean_pic = cv2.fastNlMeansDenoisingColored(picture,None,10,10,7,21)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        clean_pic = cv2.morphologyEx(clean_pic, cv2.MORPH_OPEN, kernel, cv2.BORDER_REPLICATE)
        clean_pic = cv2.morphologyEx(clean_pic, cv2.MORPH_CLOSE, kernel, cv2.BORDER_REPLICATE)
        return clean_pic

    def quantification(self, picture):
        """
        Retorna a imagem K-mean.
        """
        width, height, dimension = tuple(picture.shape)
        image_array = np.reshape(picture, (width * height, dimension))
        image_array_sample = shuffle(image_array, random_state=0)[:1000]
        kmeans = KMeans(n_clusters=self.nb_color, random_state=42).fit(image_array_sample)
        labels = kmeans.predict(image_array)
        new_img = self.recreate_image(kmeans.cluster_centers_, labels, width,height)
        return new_img, kmeans.cluster_centers_
    def recreate_image(self, codebook, labels, width, height):
    """
    Cria a imagem a partir de uma lista de cores, rótulos e tamanho da imagem.
    """
    vfunc = lambda x: codebook[labels[x]]
    out = vfunc(np.arange(width*height))
    return np.resize(out, (width, height, codebook.shape[1]))

def apply_canny(self, image):
    """
    Aplica o filtro de Canny à imagem para detectar bordas.
    """
    grayscale = cv2.cvtColor(image.astype('float32'), cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(grayscale, 30, 100)
    return edged

def invert_colors(self, image):
    """
    Inverte as cores da imagem.
    """
    return cv2.bitwise_not(image)


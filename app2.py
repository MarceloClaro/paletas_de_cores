import numpy as np
import cv2
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

# Classe Canvas para manipulação da imagem e quantificação de cores
class Canvas():
    def __init__(self, src, nb_color, pixel_size=4000):
        self.src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        self.nb_color = nb_color
        self.tar_width = pixel_size
        self.colormap = []
        self.color_percentages = []

    def generate(self):
        im_source = self.resize()
        clean_img = self.cleaning(im_source)
        clean_img = np.array(clean_img, dtype="uint8") / 255
        quantified_image, colors, color_percentages = self.quantification(clean_img)
        self.color_percentages = color_percentages
        canvas = np.ones(quantified_image.shape[:2], dtype="uint8") * 255
        individual_layers = []  # Lista para armazenar imagens das camadas coloridas numeradas

        # Itera sobre as cores encontradas e cria camadas numeradas
        for ind, color in enumerate(colors):
            self.colormap.append([int(c * 255) for c in color])
            mask = cv2.inRange(quantified_image, color, color)
            cnts = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            # Cria uma camada individual colorida com a cor dominante e numerada
            layer_colored = np.zeros(quantified_image.shape, dtype="uint8")
            color_bgr = tuple(int(c * 255) for c in color[::-1])  # Convertendo para BGR
            for contour in cnts:
                _, _, width_ctr, height_ctr = cv2.boundingRect(contour)
                if width_ctr > 10 and height_ctr > 10 and cv2.contourArea(contour, True) < -100:
                    cv2.drawContours(layer_colored, [contour], -1, color_bgr, -1)  # Preenche a camada com a cor dominante
                    txt_x, txt_y = contour[0][0]
                    cv2.putText(layer_colored, f'{ind + 1}', (txt_x, txt_y + 15),  # Numerando a camada
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            individual_layers.append(layer_colored)  # Armazena a camada numerada

        return canvas, colors, quantified_image, individual_layers

    def resize(self):
        (height, width) = self.src.shape[:2]
        if height > width:
            dim = (int(width * self.tar_width / float(height)), self.tar_width)
        else:
            dim = (self.tar_width, int(height * self.tar_width / float(width)))
        return cv2.resize(self.src, dim, interpolation=cv2.INTER_AREA)

    def cleaning(self, picture):
        clean_pic = cv2.fastNlMeansDenoisingColored(picture, None, 10, 10, 7, 21)
        kernel = np.ones((5, 5), np.uint8)
        img_erosion = cv2.erode(clean_pic, kernel, iterations=1)
        img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
        return img_dilation

    def quantification(self, picture):
        width, height, depth = picture.shape
        flattened = np.reshape(picture, (width * height, depth))
        sample = shuffle(flattened)[:1000]
        kmeans = KMeans(n_clusters=self.nb_color).fit(sample)
        labels = kmeans.predict(flattened)
        unique, counts = np.unique(labels, return_counts=True)
        color_percentages = counts / len(flattened) * 100
        new_img = self.recreate_image(kmeans.cluster_centers_, labels, width, height)
        return new_img, kmeans.cluster_centers_, color_percentages

    def recreate_image(self, codebook, labels, width, height):
        vfunc = lambda x: codebook[labels[x]]
        out = vfunc(np.arange(width * height))
        return np.resize(out, (width, height, codebook.shape[1]))

# Interface Streamlit
uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption='Imagem Carregada', use_column_width=True)

    nb_color = st.slider('Escolha o número de cores para pintar', 1, 80, 2)
    pixel_size = st.slider('Escolha o tamanho do pixel da pintura', 500, 8000, 4000)

    if st.button('Gerar'):
        canvas = Canvas(image, nb_color, pixel_size)
        result, colors, segmented_image, individual_layers = canvas.generate()
        segmented_image = (segmented_image * 255).astype(np.uint8)
        segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
        
        st.image(result, caption='Imagem Resultante', use_column_width=True)
        st.image(segmented_image, caption='Imagem Segmentada', use_column_width=True)

        # Exibir cada camada numerada e colorida para sobreposição
        st.subheader("Camadas Individuais Numeradas (Mapa Topográfico Colorido)")
        for i, layer in enumerate(individual_layers):
            st.image(layer, caption=f'Camada {i + 1} - Cor Dominante', use_column_width=True)
            layer_bytes = cv2.imencode('.jpg', layer)[1].tobytes()
            st.download_button(f"Baixar Camada {i + 1}", data=layer_bytes, file_name=f'layer_{i + 1}.jpg', mime='image/jpeg')

        # Download da imagem segmentada e da imagem resultante
        result_bytes = cv2.imencode('.jpg', result)[1].tobytes()
        st.download_button("Baixar imagem resultante", data=result_bytes, file_name='result.jpg', mime='image/jpeg')
        
        segmented_image_bytes = cv2.imencode('.jpg', segmented_image)[1].tobytes()
        st.download_button("Baixar imagem segmentada", data=segmented_image_bytes, file_name='segmented.jpg', mime='image/jpeg')

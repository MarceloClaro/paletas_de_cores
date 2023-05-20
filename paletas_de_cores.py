import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import cv2
import streamlit as st
from PIL import Image
import io
import base64

def rgb_to_cmyk(r, g, b):
    if (r == 0) and (g == 0) and (b == 0):
        return 0, 0, 0, 1
    c = 1 - r / 255
    m = 1 - g / 255
    y = 1 - b / 255

    min_cmy = min(c, m, y)
    c = (c - min_cmy) / (1 - min_cmy)
    m = (m - min_cmy) / (1 - min_cmy)
    y = (y - min_cmy) / (1 - min_cmy)
    k = min_cmy

    return c, m, y, k

def calculate_percentage(c, m, y, k):
    total_ink = c + m + y + k
    c_percent = (c / total_ink) * 100
    m_percent = (m / total_ink) * 100
    y_percent = (y / total_ink) * 100
    k_percent = (k / total_ink) * 100
    return c_percent, m_percent, y_percent, k_percent

class Canvas():
    def __init__(self, src, nb_color, pixel_size=4000):
        self.src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        self.nb_color = nb_color
        self.tar_width = pixel_size
        self.colormap = []

    def generate(self):
        im_source = self.resize()
        clean_img = self.cleaning(im_source)
        width, height, depth = clean_img.shape
        clean_img = np.array(clean_img, dtype="uint8") / 255
        quantified_image, colors = self.quantification(clean_img)
        canvas = np.ones(quantified_image.shape[:2], dtype="uint8") * 255

        for ind, color in enumerate(colors):
            self.colormap.append([int(c * 255) for c in color])
            mask = cv2.inRange(quantified_image, color, color)
            cnts = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            for contour in cnts:
                _, _, width_ctr, height_ctr = cv2.boundingRect(contour)
                if width_ctr > 10 and height_ctr > 10 and cv2.contourArea(contour, True) < -100:
                    cv2.drawContours(canvas, [contour], -1, (0, 0, 0), 1)
                    txt_x, txt_y = contour[0][0]
                    cv2.putText(canvas, '{:d}'.format(ind + 1), (txt_x, txt_y + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return canvas, colors, quantified_image

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
        new_img = self.recreate_image(kmeans.cluster_centers_, labels, width, height)
        return new_img, kmeans.cluster_centers_

    def recreate_image(self, codebook, labels, width, height):
        vfunc = lambda x: codebook[labels[x]]
        out = vfunc(np.arange(width * height))
        return np.resize(out, (width, height, codebook.shape[1]))

st.image("clube.png")
st.title('Gerador de Paleta de Cores para Pintura por Números')
st.subheader("Sketching and concept development")

uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "png"])
st.write("""
Este aplicativo é um programa criado pelo clube de artes plásticas para gerar uma paleta de cores numeradas a partir de uma imagem. Pode ser útil para artistas plásticos em suas pinturas de várias maneiras.
Primeiro, o aplicativo permite que o artista faça o upload de uma imagem de referência, que pode ser uma foto, uma ilustração ou qualquer imagem que deseje usar como base. Isso é útil para o artista visualizar uma cena ou um conceito que deseja pintar.
Em seguida, o aplicativo utiliza o algoritmo K-means para quantificar as cores presentes na imagem. O número de clusters (cores) é determinado pelo artista através de um controle deslizante. Isso permite ao artista controlar a quantidade de cores que deseja extrair da imagem.
Uma vez gerada a paleta de cores, o aplicativo exibe a imagem resultante, onde cada região da imagem original é substituída pela cor correspondente da paleta. Isso pode ajudar o artista a visualizar como sua pintura ficaria usando essas cores específicas.
Além disso, o aplicativo também exibe a imagem segmentada, onde cada região da imagem original é preenchida com uma cor sólida correspondente à cor dominante da região. Isso pode ajudar o artista a identificar áreas de destaque ou contrastes na imagem, facilitando o processo de esboço e desenvolvimento de conceitos.
O aplicativo também fornece a opção de baixar a imagem resultante (pintura numerada a ser pintada) e a imagem segmentada (prévia da tela pintada), permitindo que o artista as salve e as utilize como referência durante o processo de pintura.
Em resumo, este aplicativo pode ajudar um artista plástico fornecendo uma paleta de cores baseada em uma imagem de referência e auxiliando no esboço e desenvolvimento de conceitos através da imagem segmentada. Ele permite ao artista experimentar diferentes combinações de cores e visualizar como sua pintura pode ficar antes mesmo de começar a trabalhar na tela.
""")
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption='Imagem Carregada', use_column_width=True)

    nb_color = st.slider('Escolha o número de cores', min_value=2, max_value=80, value=5, step=1)

    if st.button('Gerar'):
        pixel_size = st.slider('Escolha o tamanho do pixel', min_value=500, max_value=8000, value=4000, step=100)

        pil_image = Image.open(io.BytesIO(file_bytes))
        if 'dpi' in pil_image.info:
            dpi = pil_image.info['dpi']
            st.write(f'Resolução da imagem: {dpi} DPI')

            cm_per_inch = pixel_size
            cm_per_pixel = cm_per_inch / dpi[0]
            st.write(f'Tamanho do pixel: {cm_per_pixel:.4f} centímetros')

        canvas = Canvas(image, nb_color, pixel_size)
        result, colors, segmented_image = canvas.generate()

        segmented_image = (segmented_image * 255).astype(np.uint8)
        segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

        st.image(result, caption='Imagem Resultante', use_column_width=True)
        st.image(segmented_image, caption='Imagem Segmentada', use_column_width=True)

        for i, color in enumerate(colors):
            color_block = np.ones((50, 50, 3), np.uint8) * color[::-1]
            st.image(color_block, width=50)

            r, g, b = color
            c, m, y, k = rgb_to_cmyk(r, g, b)
            c_percent, m_percent, y_percent, k_percent = calculate_percentage(c, m, y, k)

            # Cálculo das proporções das cores CMYK
            r, g, b = color
            c, m, y, k = rgb_to_cmyk(r, g, b)
            c_percent, m_percent, y_percent, k_percent = calculate_percentage(c, m, y, k)

            st.write(f"""
            A cor {i+1} tem os valores RGB ({int(r)}, {int(g)}, {int(b)}).
            Ela pode ser convertida para o modelo CMYK da seguinte forma:

            Ciano (C): {c_percent:.2f}%
            Magenta (M): {m_percent:.2f}%
            Amarelo (Y): {y_percent:.2f}%
            Preto (K): {k_percent:.2f}%

            Escolha a quantidade em mililitros (ml) para cada cor:
            Ciano (C): {st.number_input(f"Quantidade de Ciano (C) em ml", value=c_percent * 0.1, step=0.01):.2f} ml
            Magenta (M): {st.number_input(f"Quantidade de Magenta (M) em ml", value=m_percent * 0.1, step=0.01):.2f} ml
            Amarelo (Y): {st.number_input(f"Quantidade de Amarelo (Y) em ml", value=y_percent * 0.1, step=0.01):.2f} ml
            Preto (K): {st.number_input(f"Quantidade de Preto (K) em ml", value=k_percent * 0.1, step=0.01):.2f} ml
            """)


        result_bytes = cv2.imencode('.jpg', result)[1].tobytes()
        st.download_button(
            label="Baixar imagem resultante",
            data=result_bytes,
            file_name='result.jpg',
            mime='image/jpeg')

        segmented_image_bytes = cv2.imencode('.jpg', segmented_image)[1].tobytes()
        st.download_button(
            label="Baixar imagem segmentada",
            data=segmented_image_bytes,
            file_name='segmented.jpg',
            mime='image/jpeg')

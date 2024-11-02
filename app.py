import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import cv2
import streamlit as st
from PIL import Image
import io
import colorsys

# Function to convert RGB to CMYK
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

# Function to calculate ink amounts
def calculate_ml(c, m, y, k, total_ml):
    total_ink = c + m + y + k
    c_ml = (c / total_ink) * total_ml
    m_ml = (m / total_ink) * total_ml
    y_ml = (y / total_ink) * total_ml
    k_ml = (k / total_ink) * total_ml
    return c_ml, m_ml, y_ml, k_ml

# Function to generate color harmonies
def generate_color_harmony(color, harmony_type):
    r, g, b = [x / 255.0 for x in color]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    
    if harmony_type == "Análoga":
        h_adj = [h, (h + 0.05) % 1, (h - 0.05) % 1]
    elif harmony_type == "Complementar":
        h_adj = [h, (h + 0.5) % 1]
    elif harmony_type == "Tríade":
        h_adj = [h, (h + 1/3) % 1, (h + 2/3) % 1]
    elif harmony_type == "Tetrádica":
        h_adj = [h, (h + 0.25) % 1, (h + 0.5) % 1, (h + 0.75) % 1]
    else:
        h_adj = [h]
    
    harmonized_colors = [
        tuple(int(x * 255) for x in colorsys.hsv_to_rgb(h, s, v)) for h in h_adj
    ]
    return harmonized_colors

class Canvas():
    def __init__(self, src, nb_color, pixel_size=4000):
        self.src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        self.nb_color = nb_color
        self.tar_width = pixel_size
        self.colormap = []

    def generate(self):
        im_source = self.resize()
        clean_img = self.cleaning(im_source)
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

# Streamlit interface
st.image("clube.png")
st.title('Gerador de Paleta de Cores para Pintura por Números ')
st.subheader("Sketching and concept development")
st.write("Desenvolvido por Marcelo Claro")

uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption='Imagem Carregada', use_column_width=True)

    nb_color = st.slider('Escolha o número de cores para pintar', 1, 80, 2)
    total_ml = st.slider('Escolha o total em ml da tinta de cada cor', 1, 1000, 10)
    pixel_size = st.slider('Escolha o tamanho do pixel da pintura', 500, 8000, 4000)
    harmony_type = st.selectbox("Escolha a harmonia de cores", ["Análoga", "Complementar", "Tríade", "Tetrádica"])

    if st.button('Gerar'):
        canvas = Canvas(image, nb_color, pixel_size)
        result, colors, segmented_image = canvas.generate()
        segmented_image = (segmented_image * 255).astype(np.uint8)
        segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
        
        st.image(result, caption='Imagem Resultante', use_column_width=True)
        st.image(segmented_image, caption='Imagem Segmentada', use_column_width=True)

        # Display each color with corresponding number and harmony in the palette
        st.subheader("Paleta de Cores e Harmonias")
        for i, color in enumerate(colors):
            color_rgb = [int(c * 255) for c in color]
            color_block = np.ones((50, 50, 3), np.uint8) * color_rgb[::-1]  # BGR to RGB
            
            # Display color with label and RGB value
            st.write(f"**Número {i + 1}**")
            st.image(color_block, caption=f'RGB: {color_rgb}', width=50)
            
            # Display CMYK values
            r, g, b = color_rgb
            c, m, y, k = rgb_to_cmyk(r, g, b)
            c_ml, m_ml, y_ml, k_ml = calculate_ml(c, m, y, k, total_ml)
            st.write(f"CMYK: C: {c_ml:.2f} ml, M: {m_ml:.2f} ml, Y: {y_ml:.2f} ml, K: {k_ml:.2f} ml")

            # Accordion for color harmonies
            with st.expander(f"Mostrar Harmonia ({harmony_type})"):
                harmonized_colors = generate_color_harmony(color_rgb, harmony_type)
                for j, harmonized_color in enumerate(harmonized_colors):
                    harmony_block = np.ones((50, 50, 3), np.uint8) * harmonized_color[::-1]
                    st.image(harmony_block, caption=f'Harmonia {j + 1} - RGB: {harmonized_color}', width=50)
            
            # Separator for clarity between colors
            st.markdown("---")

        result_bytes = cv2.imencode('.jpg', result)[1].tobytes()
        st.download_button("Baixar imagem resultante", data=result_bytes, file_name='result.jpg', mime='image/jpeg')
        
        segmented_image_bytes = cv2.imencode('.jpg', segmented_image)[1].tobytes()
        st.download_button("Baixar imagem segmentada", data=segmented_image_bytes, file_name='segmented.jpg', mime='image/jpeg')

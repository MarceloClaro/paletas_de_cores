import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import cv2
import streamlit as st
from PIL import Image
import io
import colorsys
from scipy.spatial import distance

# Função para converter RGB em CMYK
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

# Função para calcular quantidade de tinta em ml para cada componente CMYK e Branco
def calculate_ml(c, m, y, k, total_ml, white_ratio=0.5):
    white_ml = total_ml * white_ratio
    remaining_ml = total_ml - white_ml
    total_ink = c + m + y + k
    c_ml = (c / total_ink) * remaining_ml
    m_ml = (m / total_ink) * remaining_ml
    y_ml = (y / total_ink) * remaining_ml
    k_ml = (k / total_ink) * remaining_ml
    return c_ml, m_ml, y_ml, k_ml, white_ml

# Função para gerar harmonias de cores
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

# Dicionário com significados dos arquétipos junguianos e valores RGB de exemplo
color_archetypes = {
    (255, 0, 0): 'Arquétipo do Herói - Energia, paixão e ação',          # Vermelho
    (0, 0, 255): 'Arquétipo do Sábio - Tranquilidade, confiança e sabedoria', # Azul
    (255, 255, 0): 'Arquétipo do Bobo - Otimismo, alegria e criatividade',    # Amarelo
    (0, 255, 0): 'Arquétipo do Cuidador - Crescimento, harmonia e renovação', # Verde
    (0, 0, 0): 'Arquétipo da Sombra - Mistério, poder e sofisticação',        # Preto
    (255, 255, 255): 'Arquétipo do Inocente - Pureza, simplicidade e novos começos', # Branco
    (128, 0, 128): 'Arquétipo do Mago - Espiritualidade, mistério e transformação',   # Roxo
    (255, 165, 0): 'Arquétipo do Explorador - Entusiasmo, aventura e vitalidade',      # Laranja
    (75, 0, 130): 'Arquétipo do Rebelde - Independência, poder e transformação',       # Índigo
    (255, 192, 203): 'Arquétipo do Amante - Amor, compaixão e sensualidade',           # Rosa
    (128, 128, 128): 'Arquétipo do Homem Comum - Neutralidade, simplicidade e realismo', # Cinza
    (255, 215, 0): 'Arquétipo do Governante - Autoridade, controle e responsabilidade',  # Dourado
    (139, 69, 19): 'Arquétipo da Terra - Segurança, conforto e resistência',            # Marrom
    (192, 192, 192): 'Arquétipo da Sabedoria - Discrição, equilíbrio e sofisticação',   # Prata
    (255, 105, 180): 'Arquétipo do Amante Jovial - Romance, empatia e diversão',        # Rosa Choque
    (0, 255, 255): 'Arquétipo da Liberdade - Integração, expansão e conexão',           # Ciano
    (34, 139, 34): 'Arquétipo do Construtor - Crescimento, perseverança e estabilidade', # Verde Escuro
    (210, 105, 30): 'Arquétipo do Alquimista - Transformação, criatividade e poder',    # Chocolate
    (123, 104, 238): 'Arquétipo do Visionário - Imaginação, inovação e possibilidades', # Azul Elétrico
    (250, 128, 114): 'Arquétipo da Paixão - Vitalidade, desejo e entusiasmo',           # Salmão
    (46, 139, 87): 'Arquétipo da Cura - Renovação, tranquilidade e naturalidade',       # Verde Marinho
    (105, 105, 105): 'Arquétipo da Neutralidade - Modéstia, resiliência e estabilidade', # Cinza Escuro
    (255, 69, 0): 'Arquétipo do Guerreiro - Coragem, impulso e força de vontade',       # Laranja Avermelhado
    (218, 112, 214): 'Arquétipo do Criativo - Originalidade, delicadeza e imaginação',  # Orquídea
    (64, 224, 208): 'Arquétipo do Pacificador - Serenidade, equilíbrio e paz',          # Turquesa
    (147, 112, 219): 'Arquétipo do Idealista - Inspiração, aspiração e sonho',          # Roxo Claro
    (0, 191, 255): 'Arquétipo do Comunicador - Clareza, expressão e intercâmbio',       # Azul Celeste
    (255, 20, 147): 'Arquétipo da Musa - Inspiração, fascínio e expressão artística',   # Rosa Escuro
    (186, 85, 211): 'Arquétipo do Místico - Magia, mistério e introspecção',            # Ameixa
    (127, 255, 0): 'Arquétipo do Naturalista - Sustentabilidade, vida e vigor',         # Verde Limão
    (255, 140, 0): 'Arquétipo da Criatividade - Energia, brilho e expressão criativa',  # Laranja Escuro
    (143, 188, 143): 'Arquétipo do Harmonizador - Equilíbrio, harmonia e paz interior', # Verde Marinho Claro
    (255, 248, 220): 'Arquétipo da Pureza - Calma, clareza e sinceridade',              # Bege
    (210, 180, 140): 'Arquétipo da Estabilidade - Segurança, tradição e consistência',  # Castanho Claro
    (238, 232, 170): 'Arquétipo da Prosperidade - Sucesso, confiança e abundância',     # Amarelo Palha
    (152, 251, 152): 'Arquétipo do Curador - Renovação, saúde e regeneração',           # Verde Claro
    (245, 222, 179): 'Arquétipo do Apoio - Hospitalidade, conforto e calor',            # Trigo
    (250, 235, 215): 'Arquétipo da Simplicidade - Pureza, honestidade e clareza',       # Branco Antigo
    (124, 252, 0): 'Arquétipo da Natureza - Crescimento, vitalidade e equilíbrio',      # Verde-Grama
    (250, 250, 210): 'Arquétipo da Calma - Tranquilidade, serenidade e estabilidade',   # Amarelo Claro
    (255, 239, 213): 'Arquétipo da Acolhida - Carinho, compreensão e receptividade',    # Papaya Whip
    (244, 164, 96): 'Arquétipo da Aventura - Descoberta, emoção e desafio',             # Areia
    (176, 224, 230): 'Arquétipo da Sensibilidade - Serenidade, equilíbrio e empatia',   # Azul Pálido
    (32, 178, 170): 'Arquétipo da Expansão - Progresso, mudança e liberdade',           # Verde-Água
    (70, 130, 180): 'Arquétipo do Realizador - Determinação, visão e resiliência',      # Azul Aço
    (169, 169, 169): 'Arquétipo da Resiliência - Força, estabilidade e consistência',   # Cinza Claro
    (250, 128, 114): 'Arquétipo da Paixão - Desejo, calor e ação',                      # Salmão
    (255, 228, 225): 'Arquétipo da Delicadeza - Afeto, compreensão e bondade',          # Rosado
    (240, 230, 140): 'Arquétipo da Prosperidade - Otimismo, positividade e energia',    # Amarelo Claro
    (255, 218, 185): 'Arquétipo da Alegria - Vibração, luz e entusiasmo',               # Pêssego
    (218, 165, 32): 'Arquétipo da Sabedoria - Experiência, valor e riqueza',            # Dourado Escuro
}


# Função para encontrar o arquétipo mais próximo usando tolerância
def find_closest_archetype(color_rgb, color_archetypes, tolerance):
    closest_archetype = "Desconhecido"
    min_dist = tolerance
    for archetype_rgb, description in color_archetypes.items():
        dist = distance.euclidean(color_rgb, archetype_rgb)
        if dist < min_dist:
            min_dist = dist
            closest_archetype = description
    return closest_archetype

# Função para criar uma imagem com borda preta de 1pt ao redor da cor
def create_color_block_with_border(color_rgb, border_color=(0, 0, 0), border_size=2, size=(50, 50)):
    color_block = np.ones((size[0], size[1], 3), np.uint8) * color_rgb[::-1]
    bordered_block = cv2.copyMakeBorder(color_block, border_size, border_size, border_size, border_size,
                                        cv2.BORDER_CONSTANT, value=border_color)
    return bordered_block

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
        unique, counts = np.unique(labels, return_counts=True)
        color_percentages = counts / len(flattened) * 100
        new_img = self.recreate_image(kmeans.cluster_centers_, labels, width, height)
        return new_img, kmeans.cluster_centers_, color_percentages

    def recreate_image(self, codebook, labels, width, height):
        vfunc = lambda x: codebook[labels[x]]
        out = vfunc(np.arange(width * height))
        return np.resize(out, (width, height, codebook.shape[1]))

# Interface Streamlit
st.image("clube.png")
st.title('Gerador de Paleta de Cores para Pintura por Números ')
st.subheader("Sketching and concept development")
st.write("Desenvolvido por Marcelo Claro")

# Configurações de tolerância no sidebar
st.sidebar.header("Configurações")
tolerance = st.sidebar.slider('Tolerância de cor', 0, 100, 40)

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

        st.subheader("Paleta de Cores e Harmonias")
        for i, (color, percentage) in enumerate(zip(colors, canvas.color_percentages)):
            color_rgb = [int(c * 255) for c in color]
            archetype_description = find_closest_archetype(color_rgb, color_archetypes, tolerance)
            
            with st.expander(f"Cor {i+1} - Arquétipo: {archetype_description.split('-')[0]}"):
                st.write(f"**Significado Psicológico:** {archetype_description}")
                st.write(f"**Percentual na Imagem:** {percentage:.2f}%")
                
                color_block_with_border = create_color_block_with_border(color_rgb, border_color=(0, 0, 0), border_size=2)
                st.image(color_block_with_border, width=60)

                r, g, b = color_rgb
                c, m, y, k = rgb_to_cmyk(r, g, b)
                c_ml, m_ml, y_ml, k_ml, white_ml = calculate_ml(c, m, y, k, total_ml, white_ratio=0.5)
                st.write(f"**Dosagem para obter a cor principal em {total_ml} ml:**")
                st.write(f"Branco (Acrílico): {white_ml:.2f} ml")
                st.write(f"Ciano (C): {c_ml:.2f} ml")
                st.write(f"Magenta (M): {m_ml:.2f} ml")
                st.write(f"Amarelo (Y): {y_ml:.2f} ml")
                st.write(f"Preto (K): {k_ml:.2f} ml")

                st.markdown("---")

                st.write("**Harmonias de Cor**")
                harmonized_colors = generate_color_harmony(color_rgb, harmony_type)
                for j, harmony_color in enumerate(harmonized_colors):
                    harmony_block_with_border = create_color_block_with_border(harmony_color, border_color=(0, 0, 0), border_size=2)
                    st.image(harmony_block_with_border, caption=f'Harmonia {j + 1} - RGB: {harmony_color}', width=60)

        result_bytes = cv2.imencode('.jpg', result)[1].tobytes()
        st.download_button("Baixar imagem resultante", data=result_bytes, file_name='result.jpg', mime='image/jpeg')
        
        segmented_image_bytes = cv2.imencode('.jpg', segmented_image)[1].tobytes()
        st.download_button("Baixar imagem segmentada", data=segmented_image_bytes, file_name='segmented.jpg', mime='image/jpeg')

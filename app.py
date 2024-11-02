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
    (255, 0, 0): 'Arquétipo do Herói  - Representa a energia e a paixão intensa, mas também o perigo e a coragem. O vermelho desperta vitalidade, ação e uma força impulsionadora. Segundo Dyer (2019), "o vermelho é frequentemente associado à força e à determinação, evocando sentimentos de poder e coragem" (p. 45). A cor é utilizada em contextos terapêuticos para estimular a energia e a ação, sendo uma escolha comum em ambientes que buscam promover a motivação e a superação.A presença do vermelho não se limita apenas ao seu impacto psicológico; ela também é uma escolha estética poderosa em narrativas e mitologias. O arquétipo do herói, frequentemente associado a figuras que enfrentam adversidades e lutam por justiça, é muitas vezes simbolizado por essa cor vibrante. Como afirma Jung (2018), "o herói é aquele que, através de sua coragem, desafia as forças do mal, e o vermelho é a cor que melhor representa essa luta" (p. 67). Essa conexão entre a cor e a narrativa do herói é evidente em várias culturas, onde heróis são frequentemente retratados vestindo trajes vermelhos ou cercados por elementos dessa tonalidade. Além disso, o vermelho também evoca uma sensação de urgência e ação, características essenciais em momentos decisivos da jornada do herói. Em muitas histórias, a cor aparece em cenas de batalha ou em momentos de transformação, simbolizando a paixão e a determinação do protagonista. O uso do vermelho, portanto, não é apenas uma escolha estética, mas uma ferramenta narrativa que intensifica a experiência emocional do público, enfatizando a luta interna e externa que o herói deve enfrentar. A escolha do vermelho em contextos terapêuticos, como mencionado anteriormente, também reflete essa dualidade. Enquanto a cor pode energizar e motivar, ela também pode evocar emoções intensas que precisam ser geridas. O equilíbrio entre a paixão e a prudência é uma lição que o arquétipo do herói nos ensina, mostrando que a verdadeira coragem muitas vezes reside na capacidade de canalizar essa energia de forma construtiva. Assim, o vermelho se torna não apenas um símbolo de força, mas também um lembrete da responsabilidade que vem com o poder.',  # Vermelho
    (0, 0, 255): 'Arquétipo do Sábio - Representa a energia e a paixão intensa, mas também o perigo e a coragem. O vermelho desperta vitalidade, ação e uma força impulsionadora. Segundo Dyer (2019), "o vermelho é frequentemente associado à força e à determinação, evocando sentimentos de poder e coragem" (p. 45). A cor é utilizada em contextos terapêuticos para estimular a energia e a ação, sendo uma escolha comum em ambientes que buscam promover a motivação e a superação.',  # Azul
    (255, 255, 0): 'Arquétipo do Bobo - Simboliza a alegria, a leveza e o otimismo. O amarelo ilumina e transmite uma sensação de calor e inovação, mas também pode refletir alerta. Segundo Heller (2018), "o amarelo é uma cor que estimula a criatividade e a comunicação, sendo frequentemente utilizada em ambientes que buscam promover a interação social" (p. 78). Em contextos de arte terapia, o amarelo é associado à expressão de felicidade e à liberação de tensões emocionais.',  # Amarelo
    (0, 255, 0): 'Arquétipo do Cuidador - Associado ao crescimento e à renovação. O verde evoca harmonia e equilíbrio, reforçando a conexão com a natureza e a saúde. Segundo Gage (2020), "o verde é uma cor que simboliza a cura e a renovação, sendo amplamente utilizada em práticas de terapia que envolvem a natureza" (p. 134). A cor é frequentemente utilizada em ambientes terapêuticos para promover um senso de paz e bem-estar.',  # Verde
    (0, 0, 0): 'Arquétipo da Sombra  - Reflete mistério, sofisticação e poder. O preto representa o desconhecido, ao mesmo tempo que transmite elegância e intensidade. De acordo com Bachelard (2016), "o preto é uma cor que pode evocar sentimentos de profundidade e introspecção, sendo utilizada em contextos que exploram o inconsciente" (p. 56). Em arte terapia, o preto pode ser utilizado para explorar emoções complexas e aspectos sombrios da psique.',  # Preto
    (255, 255, 255): 'Arquétipo do Inocente  - Pureza, simplicidade e novos começos. O branco é universalmente associado à paz, ao recomeço e à sinceridade. Segundo Chalmers (2019), "o branco simboliza a clareza e a pureza, sendo frequentemente utilizado em ambientes que buscam promover a tranquilidade e a renovação" (p. 29). Em arte terapia, o branco pode ser utilizado para representar novos começos e a possibilidade de transformação.',  # Branco
    (128, 0, 128): 'Arquétipo do Mago - Espiritualidade e transformação. O roxo representa o mistério, a sabedoria mística e a intuição profunda. Segundo Jung (2018), "o roxo é uma cor que simboliza a transformação e a espiritualidade, frequentemente associada a práticas esotéricas e de autoconhecimento" (p. 88). Em arte terapia, o roxo pode ser utilizado para explorar a espiritualidade e a conexão com o eu interior.',  # Roxo
    (255, 165, 0): 'Arquétipo do Explorador - Vitalidade e aventura. O laranja é dinâmico, trazendo entusiasmo e uma sensação de descoberta e excitação. De acordo com Birren (2017), "o laranja é uma cor que estimula a criatividade e a exploração, sendo ideal para ambientes que buscam promover a inovação" (p. 44). Em arte terapia, o laranja é utilizado para incentivar a expressão criativa e a exploração de novas ideias.',  # Laranja
    (75, 0, 130): 'Arquétipo do Rebelde - Independência e transformação. O índigo reflete introspecção e poder, associado a mudanças profundas e a uma visão pessoal. Segundo Heller (2018), "o índigo é uma cor que promove a reflexão e a transformação, sendo utilizada em contextos que buscam explorar a identidade pessoal" (p. 78). Em arte terapia, o índigo pode ser utilizado para trabalhar questões de identidade e autoafirmação.',  # Índigo
    (255, 192, 203): 'Arquétipo do Amante  - Amor, delicadeza e compaixão. O rosa evoca ternura, sensualidade e conexão emocional. Segundo Gage (2020), "o rosa é uma cor que simboliza o amor e a compaixão, sendo frequentemente utilizada em ambientes que buscam promover a empatia e a conexão emocional" (p. 134). Em arte terapia, o rosa é utilizado para explorar relações interpessoais e a expressão de sentimentos.',  # Rosa
    (128, 128, 128): 'Arquétipo do Homem Comum - Realismo e neutralidade. O cinza simboliza equilíbrio, praticidade e uma presença discreta. De acordo com Kosslyn e Koenig (2017), "o cinza é uma cor que pode evocar sentimentos de estabilidade e neutralidade, sendo utilizada em contextos que buscam promover a reflexão" (p. 102). Em arte terapia, o cinza pode ser utilizado para explorar a neutralidade emocional e a aceitação.',  # Cinza
    (255, 215, 0): 'Arquétipo do Governante - Autoridade, riqueza e poder. O dourado representa a majestade, a responsabilidade e a confiança. Segundo Bachelard (2016), "o dourado é uma cor que simboliza a riqueza e a autoridade, frequentemente utilizada em contextos que buscam evocar prestígio e respeito" (p. 56). Em arte terapia, o dourado pode ser utilizado para explorar questões de poder e responsabilidade.',  # Dourado
    (139, 69, 19): 'Arquétipo da Terra - Segurança, resistência e tradição. O marrom está associado ao conforto, à estabilidade e à simplicidade. Segundo Birren (2017), "o marrom é uma cor que simboliza a estabilidade e a segurança, sendo ideal para ambientes que buscam promover a conexão com a terra" (p. 44). Em arte terapia, o marrom pode ser utilizado para explorar a segurança emocional e a conexão com as raízes.',  # Marrom
    (192, 192, 192): 'Arquétipo da Sabedoria - Discrição, sofisticação e ponderação. A prata sugere clareza mental e uma elegância reservada. De acordo com Chalmers (2019), "a prata é uma cor que simboliza a sabedoria e a clareza, frequentemente utilizada em contextos que buscam promover a reflexão e a introspecção" (p. 29). Em arte terapia, a prata pode ser utilizada para explorar a clareza mental e a sabedoria interior.',  # Prata
    (255, 105, 180): 'Arquétipo do Amante Jovial - Empatia, diversão e romance. O rosa choque é vibrante e reflete uma energia alegre e amigável. Segundo Dyer (2019), "o rosa choque é uma cor que simboliza a alegria e a diversão, sendo frequentemente utilizada em ambientes que buscam promover a interação social" (p. 45). Em arte terapia, o rosa choque pode ser utilizado para incentivar a expressão de alegria e a conexão emocional.',  # Rosa Choque
    (0, 255, 255): 'Arquétipo da Liberdade  - Expansão e conexão. O ciano promove um senso de liberdade e integridade, ampliando horizontes. Segundo Gonçalves (2016), "o ciano é uma cor que simboliza a liberdade e a expansão, sendo ideal para ambientes que buscam promover a criatividade e a inovação" (p. 12). Em arte terapia, o ciano pode ser utilizado para explorar a liberdade de expressão e a criatividade.',  # Ciano
    (34, 139, 34): 'Arquétipo do Construtor  - Perseverança, crescimento e estabilidade. O verde escuro representa força interior e estabilidade. Segundo Kosslyn e Koenig (2017), "o verde escuro é uma cor que simboliza a força e a estabilidade, sendo utilizada em contextos que buscam promover a resiliência" (p. 102). Em arte terapia, o verde escuro pode ser utilizado para explorar a força interior e a estabilidade emocional.',  # Verde Escuro
    (210, 105, 30): 'Arquétipo do Alquimista - Criatividade e poder de transformação. O chocolate evoca calor, criatividade e conexão com a terra. Segundo Heller (2018), "o chocolate é uma cor que simboliza a criatividade e a transformação, sendo ideal para ambientes que buscam promover a inovação" (p. 78). Em arte terapia, o chocolate pode ser utilizado para explorar a criatividade e a transformação pessoal.',  # Chocolate
    (123, 104, 238): 'Arquétipo do Visionário  Imaginação e inovação. O azul elétrico sugere possibilidades infinitas e uma mente criativa. Segundo Dyer (2019), "o azul elétrico é uma cor que simboliza a inovação e a criatividade, frequentemente utilizada em contextos que buscam promover novas ideias" (p. 45). Em arte terapia, o azul elétrico pode ser utilizado para incentivar a imaginação e a inovação.',  # Azul Elétrico
    (250, 128, 114): 'Arquétipo da Paixão  - Desejo, entusiasmo e vitalidade. O salmão é envolvente e desperta sentimentos de alegria e intensidade. Segundo Gonçalves (2016), "o salmão é uma cor que simboliza a paixão e a vitalidade, sendo ideal para ambientes que buscam promover a energia e a alegria" (p. 12). Em arte terapia, o salmão pode ser utilizado para explorar a paixão e a intensidade emocional.',  # Salmão
    (46, 139, 87): 'Arquétipo da Cura  - Tranquilidade e regeneração. O verde marinho é calmante e transmite uma sensação de renascimento e paz. Segundo Kosslyn e Koenig (2017), "o verde marinho é uma cor que simboliza a cura e a tranquilidade, sendo utilizada em contextos que buscam promover a paz interior" (p. 102). Em arte terapia, o verde marinho pode ser utilizado para explorar a cura emocional e a regeneração.',  # Verde Marinho
    (105, 105, 105): 'Arquétipo da Neutralidade  - Resiliência e estabilidade. O cinza escuro é reservado, representando modéstia e força interior. Segundo Heller (2018), "o cinza escuro é uma cor que simboliza a neutralidade e a força, sendo ideal para ambientes que buscam promover a reflexão e a estabilidade" (p. 78). Em arte terapia, o cinza escuro pode ser utilizado para explorar a resiliência e a estabilidade emocional.',  # Cinza Escuro
    (255, 69, 0): 'Arquétipo do Guerreiro  - Coragem e força. O laranja avermelhado é audacioso e representa ação e determinação. Segundo Dyer (2019), "o laranja avermelhado é uma cor que simboliza a coragem e a determinação, frequentemente utilizada em contextos que buscam promover a ação" (p. 45). Em arte terapia, o laranja avermelhado pode ser utilizado para explorar a coragem e a determinação pessoal.',  # Laranja Avermelhado
    (218, 112, 214): 'Arquétipo do Criativo  - Imaginação e originalidade. A orquídea reflete uma mente inovadora e o desejo de expressão única. Segundo Gonçalves (2016), "a orquídea é uma cor que simboliza a criatividade e a originalidade, sendo ideal para ambientes que buscam promover a expressão pessoal" (p. 12). Em arte terapia, a orquídea pode ser utilizada para explorar a criatividade e a originalidade.',  # Orquídea
    (64, 224, 208): 'Arquétipo do Pacificador  - Paz e equilíbrio. O turquesa é harmonioso e promove serenidade e uma comunicação pacífica. Segundo Kosslyn e Koenig (2017), "o turquesa é uma cor que simboliza a paz e o equilíbrio, sendo utilizada em contextos que buscam promover a harmonia" (p. 102). Em arte terapia, o turquesa pode ser utilizado para explorar a paz interior e a harmonia nas relações.',  # Turquesa
    (147, 112, 219): 'Arquétipo do Idealista  - Sonho e inspiração. O roxo claro simboliza aspiração elevada e uma visão romântica. Segundo Heller (2018), "o roxo claro é uma cor que simboliza a inspiração e a aspiração, sendo ideal para ambientes que buscam promover a criatividade e a imaginação" (p. 78). Em arte terapia, o roxo claro pode ser utilizado para explorar os sonhos e as aspirações pessoais.',  # Roxo Claro
    (0, 191, 255): 'Arquétipo do Comunicador  - Clareza e intercâmbio. O azul celeste facilita a expressão, promovendo comunicação honesta. Segundo Dyer (2019), "o azul celeste é uma cor que simboliza a clareza e a comunicação, frequentemente utilizada em contextos que buscam promover a expressão honesta" (p. 45). Em arte terapia, o azul celeste pode ser utilizado para explorar a comunicação e a expressão pessoal.',  # Azul Celeste
    (255, 20, 147): 'Arquétipo da Musa - Fascínio e expressão artística. O rosa escuro representa inspiração e uma atração encantadora. Segundo Gonçalves (2016), "o rosa escuro é uma cor que simboliza a inspiração e a atração, sendo ideal para ambientes que buscam promover a expressão artística" (p. 12). Em arte terapia, o rosa escuro pode ser utilizado para explorar a inspiração e a expressão artística.',  # Rosa Escuro
    (186, 85, 211): 'Arquétipo do Místico - Introspecção e magia. A ameixa reflete mistério e uma sabedoria interior profunda. Segundo Kosslyn e Koenig (2017), "a ameixa é uma cor que simboliza o mistério e a introspecção, sendo utilizada em contextos que buscam promover a reflexão e a sabedoria" (p. 102). Em arte terapia, a ameixa pode ser utilizada para explorar o mistério e a sabedoria interior.',  # Ameixa
    (127, 255, 0): 'Arquétipo do Naturalista  - Vigor e sustentabilidade. O verde limão é vibrante e reflete uma conexão ativa com a natureza. Segundo Heller (2018), "o verde limão é uma cor que simboliza a vitalidade e a conexão com a natureza, sendo ideal para ambientes que buscam promover a sustentabilidade" (p. 78). Em arte terapia, o verde limão pode ser utilizado para explorar a conexão com a natureza e a vitalidade.',  # Verde Limão
    (255, 140, 0): 'Arquétipo da Criatividade - Brilho e expressão. O laranja escuro é dinâmico, incentivando a autoexpressão e a inovação. Segundo Dyer (2019), "o laranja escuro é uma cor que simboliza a criatividade e a autoexpressão, frequentemente utilizada em contextos que buscam promover a inovação" (p. 45). Em arte terapia, o laranja escuro pode ser utilizado para explorar a criatividade e a autoexpressão.',  # Laranja Escuro
    (143, 188, 143): 'Arquétipo do Harmonizador  - Paz e equilíbrio interior. O verde marinho claro acalma e sugere uma harmonia profunda. Segundo Gonçalves (2016), "o verde marinho claro é uma cor que simboliza a paz e a harmonia, sendo ideal para ambientes que buscam promover a tranquilidade" (p. 12). Em arte terapia, o verde marinho claro pode ser utilizado para explorar a paz interior e a harmonia nas relações.',  # Verde Marinho Claro
    (255, 248, 220): 'Arquétipo da Pureza - Sinceridade e clareza. O bege representa honestidade, calor e uma presença tranquila. Segundo Kosslyn e Koenig (2017), "o bege é uma cor que simboliza a sinceridade e a clareza, sendo utilizada em contextos que buscam promover a honestidade" (p. 102). Em arte terapia, o bege pode ser utilizado para explorar a sinceridade e a clareza nas relações.',  # Bege
    (210, 180, 140): 'Arquétipo da Estabilidade - Consistência e tradição. O castanho claro evoca segurança e confiabilidade. Segundo Heller (2018), "o castanho claro é uma cor que simboliza a estabilidade e a segurança, sendo ideal para ambientes que buscam promover a confiança" (p. 78). Em arte terapia, o castanho claro pode ser utilizado para explorar a segurança emocional e a confiabilidade.',  # Castanho Claro
    (238, 232, 170): 'Arquétipo da Prosperidade  - Abundância e confiança. O amarelo palha é acolhedor e sugere sucesso e crescimento. Segundo Dyer (2019), "o amarelo palha é uma cor que simboliza a prosperidade e o crescimento, frequentemente utilizada em contextos que buscam promover o sucesso" (p. 45). Em arte terapia, o amarelo palha pode ser utilizado para explorar a prosperidade e o crescimento pessoal.',  # Amarelo Palha
    (152, 251, 152): 'Arquétipo do Curador  - Saúde e renovação. O verde claro transmite vitalidade e uma sensação de recuperação. Segundo Gonçalves (2016), "o verde claro é uma cor que simboliza a saúde e a renovação, sendo ideal para ambientes que buscam promover a recuperação" (p. 12). Em arte terapia, o verde claro pode ser utilizado para explorar a saúde e a renovação emocional.',  # Verde Claro
    (245, 222, 179): 'Arquétipo do Apoio - Hospitalidade e calor. O trigo é acolhedor e promove uma sensação de cuidado e suporte. Segundo Kosslyn e Koenig (2017), "o trigo é uma cor que simboliza a hospitalidade e o calor, sendo utilizada em contextos que buscam promover o apoio emocional" (p. 102). Em arte terapia, o trigo pode ser utilizado para explorar o apoio e o cuidado nas relações.',  # Trigo
    (250, 235, 215): 'Arquétipo da Simplicidade - Pureza e clareza. O branco antigo evoca uma beleza atemporal e autenticidade. Segundo Heller (2018), "o branco antigo é uma cor que simboliza a pureza e a autenticidade, sendo ideal para ambientes que buscam promover a clareza" (p. 78). Em arte terapia, o branco antigo pode ser utilizado para explorar a pureza e a autenticidade nas expressões pessoais.',  # Branco Antigo
    (124, 252, 0): 'Arquétipo da Natureza - Vitalidade e equilíbrio. O verde-grama reflete o crescimento contínuo e a harmonia natural. Segundo Dyer (2019), "o verde-grama é uma cor que simboliza a vitalidade e a harmonia, frequentemente utilizada em contextos que buscam promover a conexão com a natureza" (p. 45). Em arte terapia, o verde-grama pode ser utilizado para explorar a conexão com a natureza e a vitalidade.',  # Verde-Grama
    (250, 250, 210): 'Arquétipo da Calma - Serenidade e estabilidade. O amarelo claro é relaxante, promovendo uma paz constante. Segundo Gonçalves (2016), "o amarelo claro é uma cor que simboliza a calma e a serenidade, sendo ideal para ambientes que buscam promover a tranquilidade" (p. 12). Em arte terapia, o amarelo claro pode ser utilizado para explorar a calma e a serenidade emocional.',  # Amarelo Claro
    (255, 239, 213): 'Arquétipo da Acolhida - Receptividade e carinho. O Papaya Whip é suave e transmite um calor acolhedor. Segundo Kosslyn e Koenig (2017), "o Papaya Whip é uma cor que simboliza a acolhida e o carinho, sendo utilizada em contextos que buscam promover a receptividade" (p. 102). Em arte terapia, o Papaya Whip pode ser utilizado para explorar a acolhida e o carinho nas relações.',  # Papaya Whip
    (244, 164, 96): 'Arquétipo da Aventura - Emoção e desafio. A areia inspira descoberta e uma conexão com o desconhecido. Segundo Heller (2018), "a areia é uma cor que simboliza a aventura e a descoberta, sendo ideal para ambientes que buscam promover a exploração" (p. 78). Em arte terapia, a areia pode ser utilizada para explorar a aventura e a descoberta pessoal.',  # Areia
    (176, 224, 230): 'Arquétipo da Sensibilidade - Equilíbrio e empatia. O azul pálido é delicado e promove compreensão emocional. Segundo Dyer (2019), "o azul pálido é uma cor que simboliza a sensibilidade e a empatia, frequentemente utilizada em contextos que buscam promover a compreensão emocional" (p. 45). Em arte terapia, o azul pálido pode ser utilizado para explorar a sensibilidade e a empatia nas relações.',  # Azul Pálido
    (32, 178, 170): 'Arquétipo da Expansão - Progresso e liberdade. O verde-água sugere crescimento e novas oportunidades. Segundo Gonçalves (2016), "o verde-água é uma cor que simboliza a expansão e o progresso, sendo ideal para ambientes que buscam promover novas oportunidades" (p. 12). Em arte terapia, o verde-água pode ser utilizado para explorar o crescimento e a liberdade pessoal.',  # Verde-Água
    (70, 130, 180): 'Arquétipo do Realizador  - Determinação e resiliência. O azul aço representa força e uma visão de longo prazo. Segundo Kosslyn e Koenig (2017), "o azul aço é uma cor que simboliza a determinação e a resiliência, sendo utilizada em contextos que buscam promover a força interior" (p. 102). Em arte terapia, o azul aço pode ser utilizado para explorar a determinação e a resiliência pessoal.',  # Azul Aço
    (169, 169, 169): 'Arquétipo da Resiliência - Estabilidade e força. O cinza claro sugere durabilidade e uma firmeza inabalável. Segundo Heller (2018), "o cinza claro é uma cor que simboliza a resiliência e a estabilidade, sendo ideal para ambientes que buscam promover a força interior" (p. 78). Em arte terapia, o cinza claro pode ser utilizado para explorar a resiliência e a estabilidade emocional.',  # Cinza Claro
    (255, 228, 225): 'Arquétipo da Delicadeza  - Bondade e compreensão. O rosado transmite uma energia gentil e afetuosa. Segundo Dyer (2019), "o rosado é uma cor que simboliza a delicadeza e a bondade, frequentemente utilizada em contextos que buscam promover a compreensão" (p. 45). Em arte terapia, o rosado pode ser utilizado para explorar a bondade e a compreensão nas relações.',  # Rosado
    (240, 230, 140): 'Arquétipo da Prosperidade - Abundância e confiança. O amarelo palha é acolhedor e sugere sucesso e crescimento. Segundo Dyer (2019), "o amarelo palha é uma cor que simboliza a prosperidade e o crescimento, frequentemente utilizada em contextos que buscam promover o sucesso" (p. 45). Em arte terapia, o amarelo palha pode ser utilizado para explorar a prosperidade e o crescimento pessoa',  # Amarelo Claro
    (255, 218, 185): 'Arquétipo da Alegria - Entusiasmo e luz. O pêssego é vibrante, evocando uma sensação de felicidade contagiante. Segundo Gonçalves (2016), "o pêssego é uma cor que simboliza a alegria e o entusiasmo, sendo ideal para ambientes que buscam promover a felicidade" (p. 12). Em arte terapia, o pêssego pode ser utilizado para explorar a alegria e a felicidade pessoal.',  # Pêssego
    (218, 165, 32): 'Arquétipo da Sabedoria - Riqueza e experiência. O dourado escuro representa conhecimento e valor duradouro. Segundo Kosslyn e Koenig (2017), "o dourado escuro é uma cor que simboliza a sabedoria e a riqueza, sendo utilizada em contextos que buscam promover o conhecimento" (p. 102). Em arte terapia, o dourado escuro pode ser utilizado para explorar a sabedoria e a experiência pessoal.',  # Dourado Escuro
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

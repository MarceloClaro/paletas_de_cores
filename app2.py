import numpy as np
import cv2
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

# Função para segmentar imagem em camadas de cor
def segment_image_into_layers(image, nb_color=5):
    # Redimensiona a imagem para análise mais rápida
    small_image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)
    small_image = np.float32(small_image) / 255.0
    
    # Clusterização das cores
    h, w, ch = small_image.shape
    data = small_image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=nb_color, random_state=42).fit(data)
    labels = kmeans.predict(data)
    
    # Processa cada camada de cor
    color_layers = []
    for i in range(nb_color):
        mask = (labels.reshape(h, w) == i).astype(np.uint8) * 255
        color_layer = cv2.bitwise_and(small_image, small_image, mask=mask)
        color_layers.append(cv2.resize(color_layer, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST))
    
    return color_layers, kmeans.cluster_centers_

# Função para preparar cada camada para o corte em MDF (camadas sólidas)
def prepare_layers_for_mdf(color_layers):
    mdf_layers = []
    for idx, layer in enumerate(color_layers):
        gray = cv2.cvtColor((layer * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        
        # Processa para obter contornos sólidos
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solid_layer = np.zeros_like(gray)
        
        # Preenche as áreas internas
        cv2.drawContours(solid_layer, contours, -1, 255, thickness=cv2.FILLED)
        solid_layer_colored = cv2.merge([solid_layer] * 3)  # Converte para 3 canais
        
        # Adiciona um contorno para o efeito de "curva de nível"
        cv2.drawContours(solid_layer_colored, contours, -1, (0, 0, 0), thickness=2)
        
        mdf_layers.append(solid_layer_colored)
    return mdf_layers

# Interface no Streamlit para exibir camadas e o resultado final
st.title('Mapa Topográfico em Camadas')
uploaded_file = st.file_uploader("Carregue uma imagem", type=["jpg", "png"])

if uploaded_file:
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    st.image(image, caption='Imagem Carregada', use_column_width=True)
    
    nb_color = st.slider('Número de Cores (Camadas)', 1, 10, 5)
    color_layers, color_centers = segment_image_into_layers(image, nb_color)
    
    # Exibe as camadas processadas para MDF
    st.subheader("Camadas Segmentadas para Corte")
    mdf_layers = prepare_layers_for_mdf(color_layers)
    
    for idx, layer in enumerate(mdf_layers):
        st.image(layer, caption=f"Camada {idx + 1}", use_column_width=True)
    
    # Mostra imagem sobreposta (camadas empilhadas)
    stacked_image = np.zeros_like(image)
    for idx, layer in enumerate(mdf_layers):
        stacked_image = cv2.add(stacked_image, layer)
        
    st.subheader("Mapa Topográfico Empilhado (Visualização)")
    st.image(stacked_image, caption="Mapa Topográfico em Camadas", use_column_width=True)

    # Salva a imagem empilhada
    result_bytes = cv2.imencode('.png', stacked_image)[1].tobytes()
    st.download_button("Baixar Mapa Topográfico Empilhado", data=result_bytes, file_name='mapa_topografico.png', mime='image/png')
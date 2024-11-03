import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Configuração inicial do Streamlit
st.title("Visualização de Paletas de Cores")

# Função para carregar imagem
def load_image(uploaded_file):
    image = np.array(Image.open(uploaded_file))
    return image

# Upload da imagem
uploaded_file = st.file_uploader("Carregar imagem", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = load_image(uploaded_file)
    st.image(image, caption="Imagem Original", use_column_width=True)

    # Conversão da imagem para escala de cinza
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Configuração das camadas a serem geradas (exemplo: aplicar segmentação)
    mdf_layers = [cv2.Canny(gray_image, 100, 200)]  # Exemplo de camada com bordas
    
    # Exibição das camadas
    st.subheader("Camadas Geradas")
    for idx, layer in enumerate(mdf_layers):
        st.image(layer, caption=f"Camada {idx + 1}", use_column_width=True, clamp=True)

    # Empilhamento das camadas (garante que o tipo de dado seja compatível)
    stacked_image = np.zeros_like(gray_image)
    for idx, layer in enumerate(mdf_layers):
        if layer.shape == stacked_image.shape:
            stacked_image = cv2.add(stacked_image, layer)
        else:
            st.warning("Dimensões incompatíveis na camada empilhada.")

    # Exibição da imagem empilhada
    st.subheader("Mapa Topográfico Empilhado (Visualização)")
    st.image(stacked_image, caption="Mapa Topográfico em Camadas", use_column_width=True, clamp=True)

    # Salva a imagem empilhada para download
    result_bytes = cv2.imencode('.png', stacked_image)[1].tobytes()
    st.download_button("Baixar Mapa Topográfico Empilhado", data=result_bytes, file_name="stacked_image.png", mime="image/png")

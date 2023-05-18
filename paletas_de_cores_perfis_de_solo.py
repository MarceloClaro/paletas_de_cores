import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans
from PIL import Image, ImageFont, ImageDraw

@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img

def apply_canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    return edges

def plot_colors(colors):
    cluster_ids = range(len(colors))
    plt.bar(cluster_ids, len(colors)*[1], color=colors/255.0, tick_label=cluster_ids)
    plt.xlabel('Cluster ID')
    plt.ylabel('Count')
    plt.title('Colors of Clusters')
    st.pyplot(plt.gcf())  # st.pyplot() requires a matplotlib.pyplot figure, not AxesSubplot. plt.gcf() gets the current figure.

def main():
    st.title("App de Pintura por Números com K-means")
    uploaded_file = st.file_uploader("Escolha uma imagem...", type=['jpg','png','jpeg'])
    
    if uploaded_file is not None:
        img = load_image(uploaded_file)
        st.image(img, caption='Imagem original.', use_column_width=True)
        
        img = np.array(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = apply_canny(img_gray)
        
        st.image(edges, caption='Bordas detectadas.', use_column_width=True)
        
        # Aplicar K-means
        k = st.slider("Número de cores", 1, 10, 3)
        img = img.reshape((-1, 3))
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(img)
        colors = kmeans.cluster_centers_
        
        # Plotar as cores
        plot_colors(colors)

if __name__ == "__main__":
    main()

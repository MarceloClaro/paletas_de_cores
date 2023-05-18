import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

@st.cache(allow_output_mutation=True)
def load_image(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

def apply_canny(image):
    if len(image.shape) > 2:  # if image has more than one channel
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # image is already grayscale
    edges = cv2.Canny(gray, 30, 100)
    return edges

def apply_kmeans(image, n_clusters):
    h, w, _ = image.shape
    image = image.reshape(h*w, 3)
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(image)
    cluster_centers = kmeans.cluster_centers_
    return cluster_centers, labels, w, h

def plot_rgb(cluster_centers):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    rgb = [[c/256 for c in center] for center in cluster_centers]
    rgb = np.array(rgb)
    ax.scatter(rgb[:,0], rgb[:,1], rgb[:,2], color=rgb)
    return fig

def main():
    image_file = st.file_uploader("Upload Image", type=['jpeg', 'png', 'jpg', 'webp'])
    n_clusters = st.sidebar.slider('Número de Cores', 1, 30, 10)

    if image_file is not None:
        img = load_image(image_file)
        st.image(img, channels='BGR')
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = apply_canny(img_gray)
        st.image(edges)
        cluster_centers, labels, w, h = apply_kmeans(img, n_clusters)
        img_kmean = cluster_centers[labels].reshape(h, w, 3)
        st.image(img_kmean/255.0, channels='BGR')
        fig = plot_rgb(cluster_centers)
        st.pyplot(fig)

if __name__ == "__main__":
    main()

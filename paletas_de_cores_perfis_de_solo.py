if imagem_up is not None:
    imagem = Image.open(imagem_up)
    imagem_cv = np.array(imagem)

    if imagem_cv.size == 0:  # Teste para imagem vazia
        st.error("A imagem carregada está vazia. Por favor, carregue uma imagem válida.")
    else:
        st.image(imagem_cv, caption='Imagem Original', use_column_width=True)

        centers, segmented_image = extrair_paleta(imagem_cv, n_cores)

        # Converte a imagem segmentada de volta para uma imagem de 8 bits
        segmented_image = cv2.convertScaleAbs(segmented_image)

        st.image(segmented_image, caption='Imagem Segmentada', use_column_width=True)

        st.subheader('Paleta de Cores:')
        plt.figure(figsize=(5, 2))
        plt.imshow([centers.astype(int)], aspect='auto')
        plt.axis('off')
        st.pyplot(plt)

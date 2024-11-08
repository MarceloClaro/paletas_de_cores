import os
import json
import re
import pandas as pd
import streamlit as st
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import cv2
import colorsys
from scipy.spatial import distance
from gtts import gTTS
import tempfile

# Importação do módulo groq
from groq import Groq  # Certifique-se de que a biblioteca groq está instalada

# Verificação da versão do Python para importação correta
import sys
if sys.version_info >= (3, 9):
    from collections.abc import Callable
else:
    from typing import Callable

# Importação do Tuple para anotações de tipo
try:
    from typing import Tuple
except ImportError:
    Tuple = tuple

# Configurações da página do Streamlit
st.set_page_config(
    page_title="Consultor de PDFs + IA",
    page_icon="logo.png",
    layout="wide",
)

# Definição de constantes
FILEPATH = "agents.json"
CHAT_HISTORY_FILE = 'chat_history.json'
API_USAGE_FILE = 'api_usage.json'

MODEL_MAX_TOKENS = {
    'mixtral-8x7b-32768': 32768,
    'llama3-70b-8192': 8192,
    'llama3-8b-8192': 8192,
    'gemma-7b-it': 8192,
}

# Definição das chaves de API (substitua pelas suas chaves reais)
API_KEYS = {
    "fetch": ["SUA_API_KEY_FETCH_1", "SUA_API_KEY_FETCH_2"],
    "refine": ["SUA_API_KEY_REFINE_1", "SUA_API_KEY_REFINE_2"],
    "evaluate": ["SUA_API_KEY_EVALUATE_1", "SUA_API_KEY_EVALUATE_2"]
}

# Variáveis para manter o estado das chaves de API
CURRENT_API_KEY_INDEX = {
    "fetch": 0,
    "refine": 0,
    "evaluate": 0
}

# [As funções anteriores permanecem as mesmas...]

# Função para gerar interpretação personalizada usando groq
def generate_personalized_interpretation(color_rgb, archetype_description, user_question):
    # Inicializa o modelo Groq
    groq_model = Groq()
    prompt = f"A cor {color_rgb} é associada a {archetype_description}. Em relação à pergunta '{user_question}', como essa cor pode representar aspectos ligados a esse arquétipo?"
    # Gera a interpretação usando o modelo Groq
    interpretation = groq_model.generate(prompt)
    return interpretation

# Interface Principal com Streamlit

# Inicialização do estado da sessão
if 'resposta_assistente' not in st.session_state:
    st.session_state.resposta_assistente = ""
if 'descricao_especialista_ideal' not in st.session_state:
    st.session_state.descricao_especialista_ideal = ""
if 'resposta_refinada' not in st.session_state:
    st.session_state.resposta_refinada = ""
if 'resposta_original' not in st.session_state:
    st.session_state.resposta_original = ""
if 'rag_resposta' not in st.session_state:
    st.session_state.rag_resposta = ""
if 'references_df' not in st.session_state:
    st.session_state.references_df = pd.DataFrame()

agent_options = load_agent_options()

#st.image('updating (2).gif', width=100, caption='Consultor de PDFs + IA', use_column_width='always', output_format='auto')
st.markdown("<h1 style='text-align: center;'>Consultor de PDFs</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Utilize nossa plataforma para consultas detalhadas em PDFs.</h2>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

memory_selection = st.selectbox("Selecione a quantidade de interações para lembrar:", options=[5, 10, 15, 25, 50, 100, 150, 300, 450])

st.write("Digite sua solicitação para que ela seja respondida pelo especialista ideal.")
col1, col2 = st.columns(2)

with col1:
    user_input = st.text_area("Por favor, insira sua solicitação:", height=200, key="entrada_usuario")
    user_prompt = st.text_area("Escreva um prompt ou coloque o texto para consulta para o especialista (opcional):", height=200, key="prompt_usuario")
    agent_selection = st.selectbox("Escolha um Especialista", options=agent_options, index=0, key="selecao_agente")
    model_name = st.selectbox("Escolha um Modelo", list(MODEL_MAX_TOKENS.keys()), index=0, key="nome_modelo")
    temperature = st.slider("Nível de Criatividade", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="temperatura")
    interaction_number = len(load_api_usage()) + 1

    fetch_clicked = st.button("Buscar Resposta")
    refine_clicked = st.button("Refinar Resposta")
    evaluate_clicked = st.button("Avaliar Resposta com RAG")
    refresh_clicked = st.button("Apagar")

    references_file = st.file_uploader("Upload do arquivo JSON ou PDF com referências (opcional)", type=["json", "pdf"], key="arquivo_referencias")

with col2:
    container_saida = st.container()

    chat_history = load_chat_history()[-memory_selection:]

    if fetch_clicked:
        if references_file:
            df = upload_and_extract_references(references_file)
            if isinstance(df, pd.DataFrame):
                st.write("### Dados Extraídos do PDF")
                st.dataframe(df)
                st.session_state.references_path = "references.csv"
                st.session_state.references_df = df

        expert_description, assistant_response = fetch_assistant_response(user_input, user_prompt, model_name, temperature, agent_selection, chat_history, interaction_number, st.session_state.get('references_df'))
        st.session_state.descricao_especialista_ideal = expert_description
        st.session_state.resposta_assistente = assistant_response
        st.session_state.resposta_original = assistant_response
        st.session_state.resposta_refinada = ""
        save_chat_history(user_input, user_prompt, assistant_response)

    if refine_clicked:
        if st.session_state.resposta_assistente:
            references_context = ""
            if not st.session_state.references_df.empty:
                for index, row in st.session_state.references_df.iterrows():
                    titulo = row.get('titulo', row['Text'][:50] + '...')
                    autor = row.get('autor', 'Autor Desconhecido')
                    ano = row.get('ano', 'Ano Desconhecido')
                    paginas = row.get('Page', 'Página Desconhecida')
                    references_context += f"Título: {titulo}\nAutor: {autor}\nAno: {ano}\nPágina: {paginas}\n\n"
            st.session_state.resposta_refinada = refine_response(st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente, user_input, user_prompt, model_name, temperature, references_context, chat_history, interaction_number)
            save_chat_history(user_input, user_prompt, st.session_state.resposta_refinada)
        else:
            st.warning("Por favor, busque uma resposta antes de refinar.")

    if evaluate_clicked:
        if st.session_state.resposta_assistente and st.session_state.descricao_especialista_ideal:
            st.session_state.rag_resposta = evaluate_response_with_rag(user_input, user_prompt, st.session_state.descricao_especialista_ideal, st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente, model_name, temperature, chat_history, interaction_number)
            save_chat_history(user_input, user_prompt, st.session_state.rag_resposta)
        else:
            st.warning("Por favor, busque uma resposta e forneça uma descrição do especialista antes de avaliar com RAG.")

    with container_saida:
        st.write(f"**#Análise do Especialista:**\n{st.session_state.descricao_especialista_ideal}")
        st.write(f"\n**#Resposta do Especialista:**\n{st.session_state.resposta_original}")

        if st.session_state.resposta_original:
            # Converter a resposta em fala
            tts = gTTS(st.session_state.resposta_original, lang='pt')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                # Reproduzir o áudio na aplicação
                st.audio(fp.name, format='audio/mp3')

        if st.session_state.resposta_refinada:
            st.write(f"\n**#Resposta Refinada:**\n{st.session_state.resposta_refinada}")
            # Converter a resposta refinada em fala
            tts_refinada = gTTS(st.session_state.resposta_refinada, lang='pt')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp_refinada:
                tts_refinada.save(fp_refinada.name)
                st.audio(fp_refinada.name, format='audio/mp3')

        if st.session_state.rag_resposta:
            st.write(f"\n**#Avaliação com RAG:**\n{st.session_state.rag_resposta}")
            # Converter a avaliação com RAG em fala
            tts_rag = gTTS(st.session_state.rag_resposta, lang='pt')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp_rag:
                tts_rag.save(fp_rag.name)
                st.audio(fp_rag.name, format='audio/mp3')

    st.markdown("### Histórico do Chat")
    if chat_history:
        tab_titles = [f"Interação {i+1}" for i in range(len(chat_history))]
        tabs = st.tabs(tab_titles)
        
        for i, entry in enumerate(chat_history):
            with tabs[i]:
                st.write(f"**Entrada do Usuário:** {entry['user_input']}")
                st.write(f"**Prompt do Usuário:** {entry['user_prompt']}")
                st.write(f"**Resposta do Especialista:** {entry['expert_response']}")
                st.markdown("---")

if refresh_clicked:
    clear_chat_history()
    st.session_state.clear()
    st.experimental_rerun()

# Adição da nova funcionalidade de interpretação de cores

st.header('Gerador de Paleta de Cores para Pintura por Números')
st.subheader("Desenvolvido por Marcelo Claro")

# Configurações de tolerância no sidebar
st.sidebar.header("Configurações de Cores")
tolerance = st.sidebar.slider('Tolerância de cor', 0, 100, 40)
harmony_type = st.sidebar.selectbox("Escolha a harmonia de cores", ["Análoga", "Complementar", "Tríade", "Tetrádica"])

# Entrada da pergunta do usuário
user_question = st.text_input("Digite sua pergunta para interpretar as cores:", "")

# Upload da imagem
uploaded_file = st.file_uploader("Escolha uma imagem para gerar a paleta de cores", type=["jpg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption='Imagem Carregada', use_column_width=True)

    nb_color = st.slider('Escolha o número de cores para pintar', 1, 80, 5)
    total_ml = st.slider('Escolha o total em ml da tinta de cada cor', 1, 1000, 10)
    pixel_size = st.slider('Escolha o tamanho do pixel da pintura', 500, 8000, 4000)

    if st.button('Gerar Paleta de Cores'):
        canvas = Canvas(image, nb_color, pixel_size)
        result, colors, segmented_image = canvas.generate()
        segmented_image = (segmented_image * 255).astype(np.uint8)
        segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
        
        st.image(result, caption='Imagem Resultante', use_column_width=True)
        st.image(segmented_image, caption='Imagem Segmentada', use_column_width=True)

        st.subheader("Paleta de Cores e Interpretações")
        for i, (color, percentage) in enumerate(zip(colors, canvas.color_percentages)):
            color_rgb = [int(c * 255) for c in color]
            archetype_description = find_closest_archetype(color_rgb, color_archetypes, tolerance)
            
            # Gerar interpretação personalizada com base na pergunta do usuário usando groq
            interpretation = generate_personalized_interpretation(color_rgb, archetype_description, user_question)
            
            with st.expander(f"Cor {i+1} - {archetype_description.split('-')[0]}"):
                st.write(f"**Interpretação Personalizada:** {interpretation}")
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

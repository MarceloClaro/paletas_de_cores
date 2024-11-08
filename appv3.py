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

# Função para obter a próxima chave de API disponível
def get_next_api_key(action: str) -> str:
    keys = API_KEYS.get(action, [])
    if keys:
        key_index = CURRENT_API_KEY_INDEX[action]
        api_key = keys[key_index]
        CURRENT_API_KEY_INDEX[action] = (key_index + 1) % len(keys)
        return api_key
    else:
        raise ValueError(f"Nenhuma chave de API disponível para a ação: {action}")

# Funções para carregar as opções de agentes
def load_agent_options():
    if os.path.exists(FILEPATH):
        with open(FILEPATH, 'r', encoding='utf-8') as file:
            try:
                agents = json.load(file)
                return [agent["agente"] for agent in agents]
            except json.JSONDecodeError:
                return []
    else:
        return []

# Função para extrair texto de um PDF
def extrair_texto_pdf(file):
    import pdfplumber
    with pdfplumber.open(file) as pdf:
        texto_paginas = [pagina.extract_text() for pagina in pdf.pages]
    return texto_paginas

# Função para converter o texto extraído em um DataFrame
def text_to_dataframe(texto_paginas):
    data = []
    for num_pagina, texto in enumerate(texto_paginas, start=1):
        if texto:
            data.append({"Page": num_pagina, "Text": texto})
    df = pd.DataFrame(data)
    return df

# Função para fazer upload e extrair referências
def upload_and_extract_references(uploaded_file):
    if uploaded_file.name.endswith('.pdf'):
        texto_paginas = extrair_texto_pdf(uploaded_file)
        df = text_to_dataframe(texto_paginas)
        df.to_csv('references.csv', index=False)
        return df
    elif uploaded_file.name.endswith('.json'):
        df = pd.read_json(uploaded_file)
        df.to_csv('references.csv', index=False)
        return df
    else:
        st.warning("Formato de arquivo não suportado. Por favor, carregue um arquivo PDF ou JSON.")
        return pd.DataFrame()

# Função para obter o número máximo de tokens do modelo
def get_max_tokens(model_name: str) -> int:
    return MODEL_MAX_TOKENS.get(model_name, 8192)

# Função para registrar o uso da API
def log_api_usage(action, interaction_number, tokens_used, time_taken, user_input, user_prompt, api_response, expert_title, expert_description):
    log_entry = {
        "action": action,
        "interaction_number": interaction_number,
        "tokens_used": tokens_used,
        "time_taken": time_taken,
        "user_input": user_input,
        "user_prompt": user_prompt,
        "api_response": api_response,
        "expert_title": expert_title,
        "expert_description": expert_description,
        "timestamp": time.time()
    }
    if os.path.exists(API_USAGE_FILE):
        with open(API_USAGE_FILE, 'r+', encoding='utf-8') as file:
            try:
                usage_data = json.load(file)
            except json.JSONDecodeError:
                usage_data = []
            usage_data.append(log_entry)
            file.seek(0)
            json.dump(usage_data, file, indent=4, ensure_ascii=False)
    else:
        with open(API_USAGE_FILE, 'w', encoding='utf-8') as file:
            json.dump([log_entry], file, indent=4, ensure_ascii=False)

# Funções para manipular o histórico de chat
def save_chat_history(user_input, user_prompt, expert_response):
    chat_entry = {
        "user_input": user_input,
        "user_prompt": user_prompt,
        "expert_response": expert_response
    }
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'r+', encoding='utf-8') as file:
            try:
                chat_history = json.load(file)
            except json.JSONDecodeError:
                chat_history = []
            chat_history.append(chat_entry)
            file.seek(0)
            json.dump(chat_history, file, indent=4, ensure_ascii=False)
    else:
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as file:
            json.dump([chat_entry], file, indent=4, ensure_ascii=False)

def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as file:
            try:
                chat_history = json.load(file)
                return chat_history
            except json.JSONDecodeError:
                return []
    else:
        return []

def clear_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        os.remove(CHAT_HISTORY_FILE)

# Funções para manipular o uso da API
def load_api_usage():
    if os.path.exists(API_USAGE_FILE):
        with open(API_USAGE_FILE, 'r', encoding='utf-8') as file:
            try:
                usage_data = json.load(file)
                return usage_data
            except json.JSONDecodeError:
                return []
    else:
        return []

def reset_api_usage():
    if os.path.exists(API_USAGE_FILE):
        os.remove(API_USAGE_FILE)

# Função para buscar a resposta do assistente
def fetch_assistant_response(user_input: str, user_prompt: str, model_name: str, temperature: float,
                             agent_selection: str, chat_history: list, interaction_number: int,
                             references_df: pd.DataFrame = None):
    try:
        # Simulação de chamada à API do modelo
        assistant_response = f"Resposta gerada pelo modelo para a entrada: {user_input}"
        expert_description = f"Descrição do especialista selecionado: {agent_selection}"
        # Registro do uso da API
        tokens_used = len(assistant_response.split())
        time_taken = 1  # Simulação
        log_api_usage('fetch', interaction_number, tokens_used, time_taken, user_input, user_prompt,
                      assistant_response, agent_selection, expert_description)
        return expert_description, assistant_response
    except Exception as e:
        st.error(f"Ocorreu um erro ao buscar a resposta do assistente: {e}")
        return "", ""

# Função para refinar a resposta
def refine_response(expert_description: str, assistant_response: str, user_input: str, user_prompt: str,
                    model_name: str, temperature: float, references_context: str, chat_history: list,
                    interaction_number: int):
    try:
        # Simulação de refinamento da resposta
        refined_response = f"Resposta refinada: {assistant_response} com base em {references_context}"
        # Registro do uso da API
        tokens_used = len(refined_response.split())
        time_taken = 1  # Simulação
        log_api_usage('refine', interaction_number, tokens_used, time_taken, user_input, user_prompt,
                      refined_response, "", expert_description)
        return refined_response
    except Exception as e:
        st.error(f"Ocorreu um erro durante o refinamento da resposta: {e}")
        return ""

# Função para avaliar a resposta com RAG
def evaluate_response_with_rag(user_input: str, user_prompt: str, expert_title: str, expert_description: str,
                               assistant_response: str, model_name: str, temperature: float, chat_history: list,
                               interaction_number: int):
    try:
        # Simulação de avaliação com RAG
        rag_response = f"Avaliação da resposta: {assistant_response}"
        # Registro do uso da API
        tokens_used = len(rag_response.split())
        time_taken = 1  # Simulação
        log_api_usage('evaluate', interaction_number, tokens_used, time_taken, user_input, user_prompt,
                      rag_response, expert_title, expert_description)
        return rag_response
    except Exception as e:
        st.error(f"Ocorreu um erro durante a avaliação com RAG: {e}")
        return ""

# Função para salvar o especialista
def save_expert(expert_title: str, expert_description: str):
    new_expert = {
        "agente": expert_title,
        "descricao": expert_description
    }
    if os.path.exists(FILEPATH):
        with open(FILEPATH, 'r+', encoding='utf-8') as file:
            try:
                agents = json.load(file)
            except json.JSONDecodeError:
                agents = []
            agents.append(new_expert)
            file.seek(0)
            json.dump(agents, file, indent=4, ensure_ascii=False)
    else:
        with open(FILEPATH, 'w', encoding='utf-8') as file:
            json.dump([new_expert], file, indent=4, ensure_ascii=False)

# Funções adicionais para interpretação de cores

# Função para converter RGB em CMYK
def rgb_to_cmyk(r, g, b):
    if (r == 0) and (g == 0) and (b == 0):
        return 0, 0, 0, 1
    c = 1 - r / 255
    m = 1 - g / 255
    y = 1 - b / 255

    min_cmy = min(c, m, y)
    c = (c - min_cmy) / (1 - min_cmy) if (1 - min_cmy) != 0 else 0
    m = (m - min_cmy) / (1 - min_cmy) if (1 - min_cmy) != 0 else 0
    y = (y - min_cmy) / (1 - min_cmy) if (1 - min_cmy) != 0 else 0
    k = min_cmy

    return c, m, y, k

# Função para calcular quantidade de tinta em ml para cada componente CMYK e Branco
def calculate_ml(c, m, y, k, total_ml, white_ratio=0.5):
    white_ml = total_ml * white_ratio
    remaining_ml = total_ml - white_ml
    total_ink = c + m + y + k
    if total_ink == 0:
        return 0, 0, 0, 0, white_ml
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
        tuple(int(x * 255) for x in colorsys.hsv_to_rgb(hue, s, v)) for hue in h_adj
    ]
    return harmonized_colors

# Dicionário com significados simplificados dos arquétipos junguianos
color_archetypes = {
    (255, 0, 0): 'Arquétipo do Herói - Representa energia e paixão intensa.',  # Vermelho
    (0, 0, 255): 'Arquétipo do Sábio - Simboliza paz e sabedoria.',  # Azul
    (255, 255, 0): 'Arquétipo do Bobo - Reflete alegria e otimismo.',  # Amarelo
    (0, 255, 0): 'Arquétipo do Cuidador - Associado ao crescimento e harmonia.',  # Verde
    (0, 0, 0): 'Arquétipo da Sombra - Representa mistério e poder.',  # Preto
    (255, 255, 255): 'Arquétipo do Inocente - Simboliza pureza e novos começos.',  # Branco
    # Outras cores podem ser adicionadas aqui
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
        sample = shuffle(flattened, random_state=0)[:1000]
        kmeans = KMeans(n_clusters=self.nb_color, random_state=0).fit(sample)
        labels = kmeans.predict(flattened)
        unique, counts = np.unique(labels, return_counts=True)
        color_percentages = counts / len(flattened) * 100
        new_img = self.recreate_image(kmeans.cluster_centers_, labels, width, height)
        return new_img, kmeans.cluster_centers_, color_percentages

    def recreate_image(self, codebook, labels, width, height):
        vfunc = lambda x: codebook[labels[x]]
        out = vfunc(np.arange(width * height))
        return np.resize(out, (width, height, codebook.shape[1]))

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
            
            # Gerar interpretação personalizada com base na pergunta do usuário
            interpretation = f"A cor {color_rgb} é associada a {archetype_description}. Em relação à sua pergunta, ela pode representar aspectos de '{user_question}' ligados a esse arquétipo."
            
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

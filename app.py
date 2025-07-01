import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Carregar o modelo treinado
model = joblib.load('modelo_anime.pkl')
modeloScaler = joblib.load('modelo_scaler.pkl')
st.title('Previsão de Rating de Animes')

st.write('Preencha os dados abaixo para prever o rating:')

# ==== FEATURES ====
feature_names = ['episodes', 'members', 
                 'Ação / Aventura / Conflito', 'Comédia e Paródia', 'Conteúdo Adulto / Sensível',
                 'Fantasia / Sobrenatural', 'Ficção Científica / Futuro', 'Psicológico / Mistério / Suspense',
                 'Público-Alvo / Demográfico', 'Relacionamentos (BL / GL)', 'Vida Real / Drama',
                 'type_Movie', 'type_Music', 'type_ONA', 'type_OVA', 'type_Special', 'type_TV']

# ==== Inputs numéricos (Inteiros) ====
st.subheader('Informações Numéricas:')
input_data = []

numeric_features = ['episodes', 'members']
for feature in numeric_features:
    value = st.number_input(f'{feature}', min_value=0, step=1)
    input_data.append(value)

# ==== Seleção de Gêneros (Multi-Select) ====
st.subheader('Gêneros do Anime (pode marcar mais de um):')
genre_options = ['Ação / Aventura / Conflito', 'Comédia e Paródia', 'Conteúdo Adulto / Sensível',
                 'Fantasia / Sobrenatural', 'Ficção Científica / Futuro', 'Psicológico / Mistério / Suspense',
                 'Público-Alvo / Demográfico', 'Relacionamentos (BL / GL)', 'Vida Real / Drama']

selected_genres = st.multiselect('Selecione os gêneros:', genre_options)

# One Hot Encoding manual dos gêneros
genre_one_hot = [1 if genre in selected_genres else 0 for genre in genre_options]
input_data.extend(genre_one_hot)

# ==== Seleção de Tipo (Multi-Select) ====
st.subheader('Tipo de Anime (pode marcar mais de um):')
type_options = ['Movie', 'Music', 'ONA', 'OVA', 'Special', 'TV']
selected_types = st.multiselect('Selecione os tipos:', type_options)

type_features = ['type_Movie', 'type_Music', 'type_ONA', 'type_OVA', 'type_Special', 'type_TV']
type_one_hot = [1 if col.replace('type_', '') in selected_types else 0 for col in type_features]
input_data.extend(type_one_hot)

# ==== Fazer a previsão ====
if st.button('Prever Rating'):
    final_array = np.array(input_data).reshape(1, -1)
    final_array2= modeloScaler.transform(final_array)
    prediction = model.predict(final_array2)[0]
    st.success(f'Rating previsto: {prediction:.2f}')

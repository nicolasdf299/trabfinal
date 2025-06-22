import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Carregar o modelo treinado
model = joblib.load('modelo_anime.pkl')

st.title('Previsão de Rating de Animes')

st.write('Preencha os dados abaixo para prever o rating:')

# ==== FEATURES que o modelo espera ====
feature_names = ['anime_id', 'episodes', 'members', 
                 'Ação / Aventura / Conflito', 'Comédia e Paródia', 'Conteúdo Adulto / Sensível',
                 'Fantasia / Sobrenatural', 'Ficção Científica / Futuro', 'Psicológico / Mistério / Suspense',
                 'Público-Alvo / Demográfico', 'Relacionamentos (BL / GL)', 'Vida Real / Drama',
                 'type_Movie', 'type_Music', 'type_ONA', 'type_OVA', 'type_Special', 'type_TV']

# ==== Inputs numéricos ====
st.subheader('Informações Numéricas:')
input_data = []

numeric_features = ['anime_id', 'episodes', 'members']
for feature in numeric_features:
    value = st.number_input(f'{feature}', value=0.0)
    input_data.append(value)

# ==== Seleção do Gênero (One Hot) ====
st.subheader('Gênero Principal:')
genre_options = ['Ação / Aventura / Conflito', 'Comédia e Paródia', 'Conteúdo Adulto / Sensível',
                 'Fantasia / Sobrenatural', 'Ficção Científica / Futuro', 'Psicológico / Mistério / Suspense',
                 'Público-Alvo / Demográfico', 'Relacionamentos (BL / GL)', 'Vida Real / Drama']

selected_genre = st.selectbox('Escolha o gênero principal:', genre_options)

# One Hot manual para os gêneros
genre_one_hot = [1 if selected_genre == genre else 0 for genre in genre_options]
input_data.extend(genre_one_hot)

# ==== Seleção do Tipo (One Hot) ====
st.subheader('Tipo de Anime:')
type_options = ['Movie', 'Music', 'ONA', 'OVA', 'Special', 'TV']
selected_type = st.selectbox('Escolha o tipo:', type_options)

type_features = ['type_Movie', 'type_Music', 'type_ONA', 'type_OVA', 'type_Special', 'type_TV']
type_one_hot = [1 if f'type_{selected_type}' == col else 0 for col in type_features]
input_data.extend(type_one_hot)

# ==== Previsão ====
if st.button('Prever Rating'):
    final_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(final_array)[0]
    st.success(f'Rating previsto: {prediction:.2f}')

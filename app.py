import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Carregando o modelo treinado
model = joblib.load('model.pkl')

st.title('Previsão de Rating de Animes')

st.write('Preencha os dados abaixo para prever o rating:')

# Lista das features (mesmo que usou no treinamento)
feature_names = ['anime_id', 'episodes', 'members', 
                 'Ação / Aventura / Conflito', 'Comédia e Paródia', 'Conteúdo Adulto / Sensível',
                 'Fantasia / Sobrenatural', 'Ficção Científica / Futuro', 'Psicológico / Mistério / Suspense',
                 'Público-Alvo / Demográfico', 'Relacionamentos (BL / GL)', 'Vida Real / Drama',
                 'rating_notfilled', 'episodes_notfilled',
                 'type_Movie', 'type_Music', 'type_ONA', 'type_OVA', 'type_Special', 'type_TV']

# Criando campos de input para o usuário
input_data = []
for feature in feature_names:
    value = st.number_input(f'Valor de {feature}', value=0.0)
    input_data.append(value)

# Botão para fazer a previsão
if st.button('Prever Rating'):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    st.success(f'Rating previsto: {prediction:.2f}')

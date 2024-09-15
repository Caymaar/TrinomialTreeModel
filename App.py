import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Classes.Model import TrinomialTreeModel
from Classes.Market import Market
from Classes.Option import Option

st.title('Trinomial Tree Model')

# Récupère les paramètres S0, K, T, r, sigma, N dans la barre latérale
st.sidebar.header('Paramètres de l\'option')

S0 = st.sidebar.number_input('Prix initial de l\'actif sous-jacent (S0)', value=100)
K = st.sidebar.number_input('Prix d\'exercice de l\'option (K)', value=100)
T = st.sidebar.number_input('Temps jusqu\'à l\'échéance (T) en années', value=1)
rate = st.sidebar.number_input('Taux d\'intérêt sans risque (r)', value=0.05)
sigma = st.sidebar.number_input('Volatilité de l\'actif sous-jacent (sigma)', value=0.2)
N = st.sidebar.number_input('Nombre de périodes (N)', value=100)
dividend = st.sidebar.number_input('Dividende attendu (dividend)', value=0.0)
ex_div_date = st.sidebar.number_input('Date ex-dividende (ex_div_date)', value=0.40)


# Créer deux colonnes
col1, col2 = st.columns(2)

with col1:
    option_type = st.radio('Sélectionnez le type d\'option', ['call', 'put'])
with col2:
    option_style = st.radio('Sélectionnez le style d\'option', ['european', 'american'])

# Créer une instance de la classe TrinomialTreeModel
model = TrinomialTreeModel(S0, rate, sigma, K, T, option_type, option_style, N, dividend, ex_div_date)

c_factor = st.sidebar.container()

with c_factor:
    col1, col2 = st.columns(2)

    with col1:
        allow_factor = st.checkbox('Allowing factor')

    if allow_factor:
        factor = st.slider('Factor', 0, 5, 2)
        with col2:
            st.write(f"Le facteur sélectionné est : {factor}")
    else:
        factor = 0

c_threshold = st.sidebar.container()

with c_threshold:
    col1, col2 = st.columns(2)

    with col1:
        allow_threshold = st.checkbox('Allow threshold')

    if allow_threshold:
        decimal_places = st.slider('Nombre de chiffres après la virgule', 4, 10, 7)
        threshold = round(1 / 10**decimal_places, decimal_places)
        with col2:
            st.write(f"Le threshold sélectionné est : {threshold}")
    else:
        threshold = 0



model.build_tree(factor)

# Afficher le prix de l'option
st.write(f'Price : {model.calculate_option_price():.5f}')

# Afficher le graphique
st.write('Graphique')
# show the graph from model.visualize_tree()
st.pyplot(model.visualize_tree(threshold=threshold))

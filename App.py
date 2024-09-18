import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Classes.Model import TrinomialTreeModel, BlackScholesModel
from Classes.Market import Market
from Classes.Option import Option
from Classes.Node import Node
from datetime import date


st.set_page_config(layout="wide")

st.title('Trinomial Tree Model')

st.sidebar.header('Paramètres de l\'option')

st.sidebar.write('Paramètres de marché')
S0 = st.sidebar.number_input('Prix initial de l\'actif sous-jacent (S0)', value=100)
rate = st.sidebar.number_input('Taux d\'intérêt sans risque (r)', value=0.05)
sigma = st.sidebar.number_input('Volatilité de l\'actif sous-jacent (sigma)', value=0.2)
dividend = st.sidebar.number_input('Dividende attendu (dividend)', value=0.0)
dividend_date = st.sidebar.date_input('Sélectionnez une date de dividende', value=date.today() + pd.Timedelta(days=90))


st.sidebar.write('Paramètres de l\'option')
K = st.sidebar.number_input('Prix d\'exercice de l\'option (K)', value=100)
today = st.sidebar.date_input('Sélectionnez une date', value=date.today())
maturity = st.sidebar.date_input('Sélectionnez une date d\'échéance', value=date.today() + pd.Timedelta(days=365))

st.sidebar.write('Paramètres du modèle')
N = st.sidebar.number_input('Nombre de périodes (N)', value=100)

T = (maturity - today).days / 365
ex_div_date = (dividend_date - today).days / 365

c_factor = st.sidebar.container()
with c_factor:
    col1, col2 = st.columns(2)

    with col1:
        allow_factor = st.checkbox('Allowing factor')

    if allow_factor:
        factor = st.slider('Factor', 0.00, 0.99, 0.00)
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
        decimal_places = st.slider('Nombre de chiffres après la virgule', 2, 50, 10)
        threshold = round(1 / 10**decimal_places, decimal_places)
        with col2:
            st.write(f"Le threshold sélectionné est : {threshold}")
    else:
        threshold = 0

col1, col2 = st.columns(2)
with col1:
    option_type = st.radio('Sélectionnez le type d\'option', ['call', 'put'])
with col2:
    option_style = st.radio('Sélectionnez le style d\'option', ['european', 'american'])

ttm = TrinomialTreeModel(S0, rate, sigma, K, T, option_type, option_style, N, dividend, ex_div_date)
bsm = BlackScholesModel(S0, rate, sigma, K, T, option_type, option_style)

start_time = pd.Timestamp.now()
ttm.build_tree(factor=factor, threshold=threshold)
end_time = pd.Timestamp.now()

col1, col2 = st.columns(2)
with col1:
    st.write('Prix du modèle :')
    st.write(f'{ttm.calculate_option_price():.5f}')
with col2:
    st.write(f'Prix du modèle Black-Scholes (Européenne sans dividende) :')
    st.write(f'{bsm.option_price():.5f}')

st.write('Graphique')
st.pyplot(ttm.visualize_tree())

time_diff_ms = (end_time - start_time).total_seconds() * 1000

st.write(f"Temps d'exécution : {time_diff_ms/1000:.5f} s")
st.write(f"Temps d'exécution par étape : {time_diff_ms/N:.5f} ms")
st.write(f"Temps d'exécution par nœud : {time_diff_ms/ttm.number_of_nodes:.5f} ms")
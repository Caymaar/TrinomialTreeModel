import streamlit as st
import pandas as pd
from Classes.Tree import Tree
from Classes.BlackScholes import BlackScholes
from Classes.Market import Market
from Classes.Option import Option
from Classes.Greeks import Greeks
from datetime import date
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components

# Configuration de la page
st.set_page_config(layout="wide")
st.title('Trinomial Tree Model')

######################
# Sidebar
######################

st.sidebar.header('Parameters')

# Paramètres du marché
st.sidebar.write('Market Parameters')
S0 = st.sidebar.number_input('Initial price of the underlying asset (S0)', value=100.0, format="%.4f")
rate = st.sidebar.number_input('Risk-free interest rate (r)', value=0.05, format="%.4f")
sigma = st.sidebar.number_input('Volatility of the underlying asset (sigma)', value=0.2, format="%.4f")
dividend = st.sidebar.number_input('Expected dividend (dividend)', value=0.0, format="%.4f")
dividend_date = st.sidebar.date_input('Select a dividend date', value=date.today() + pd.Timedelta(days=90))

# Paramètres de l'option
st.sidebar.write('Option Parameters')
K = st.sidebar.number_input('Option strike price (K)', value=100.0, format="%.4f")
today = st.sidebar.date_input('Select a date', value=date.today())
maturity = st.sidebar.date_input('Select an expiration date', value=date.today() + pd.Timedelta(days=365))

# Calcul de T et ex_div_date
T = (maturity - today).days / 365
ex_div_date = (dividend_date - today).days / 365

# Menu de navigation
selected_tab = option_menu(
    menu_title=None,
    options=["Main", "Metrics", "Greeks"],
    icons=["house", "bar-chart", "calculator"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "transparent"},
        "icon": {"color": "black", "font-size": "20px"},
        "nav-link": {
            "font-size": "20px",
            "font-weight": "bold",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "#eee",
            "padding": "5px 10px",
            "width": "90%",
        },
        "nav-link-selected": {"background-color": "#ff9999", "color": "white"},
    },
)

st.divider()

# Contenu pour l'onglet "Main"
if selected_tab == "Main":
    with st.container():
        # Colonnes pour les titres et les menus d'options
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3 class='centered-title'>Option type</h3>", unsafe_allow_html=True)
            option_type = option_menu(
                menu_title=None,
                options=["Call", "Put"],
                icons=["graph-up", "graph-down"],
                menu_icon="cast",
                default_index=0,
                orientation="horizontal",
                styles={
                    "container": {"padding": "0!important", "background-color": "transparent"},
                    "icon": {"color": "white", "font-size": "16px"},
                    "nav-link": {
                        "font-size": "16px",
                        "text-align": "center",
                        "--hover-color": "#eee",
                    },
                    "nav-link-selected": {"background-color": "salmon", "color": "white"},
                },
            )

        with col2:
            st.markdown("<h3 class='centered-title'>Option style</h3>", unsafe_allow_html=True)
            option_style = option_menu(
                menu_title=None,
                options=["European", "American"],
                icons=["globe", "flag"],
                menu_icon="cast",
                default_index=0,
                orientation="horizontal",
                styles={
                    "container": {"padding": "0!important", "background-color": "transparent"},
                    "icon": {"color": "white", "font-size": "16px"},
                    "nav-link": {
                        "font-size": "16px",
                        "text-align": "center",
                        "--hover-color": "#eee",
                    },
                    "nav-link-selected": {"background-color": "indianred", "color": "white"},
                },
            )

    # Sélection du modèle
    model = st.selectbox(
        'Select model', 
        ['Trinomial Tree Model', 'Trinomial Tree Model with memory optimization (including threshold, No graphic available)']
    )

    # Options pour le modèle Trinomial Tree Model
    if model == 'Trinomial Tree Model':
        light_mode = False
        
        # Colonnes pour Factor et Threshold
        col1, col2 = st.columns(2)

        # Paramètres du facteur
        with col1:
            with st.expander("Factor"):
                st.markdown("You can adjust the factor to increase the pace of the tree building process. The factor should be between -1 and 1. 0 means no factor.")
                allow_factor = st.checkbox('Allowing factor')
                factor = st.slider('Factor', -0.99, 0.99, 0.00) if allow_factor else 0

        # Paramètres du seuil
        with col2:
            with st.expander("Threshold"):
                st.markdown("You can adjust the threshold to control the precision of the calculations. The threshold should be close to zero.")
                allow_threshold = st.checkbox('Allow threshold')
                decimal_places = st.slider('Number of decimal places', 2, 50, 10) if allow_threshold else 0
                threshold = round(1 / 10**decimal_places, decimal_places) if allow_threshold else 0 

    else:
        light_mode = True

        # Paramètres du facteur dans la seule colonne
        col1 = st.columns([1])[0]  # Une seule colonne
        with col1:
            with st.expander("Factor"):
                st.markdown("You can adjust the factor to increase the pace of the tree building process. The factor should be between -1 and 1. 0 means no factor.")
                allow_factor = st.checkbox('Allowing factor')
                factor = st.slider('Factor', -0.99, 0.99, 0.00) if allow_factor else 0

    # Ajout de CSS personnalisé pour centrer le bouton
    st.markdown("""
        <style>
        .center-btn {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%;  /* Prend toute la hauteur disponible */
        }
        .stButton button {
            width: 50%;  /* Largeur du bouton */
        }
        </style>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    # Injecter du CSS pour forcer le bouton à prendre toute la largeur
    st.markdown("""
        <style>
        .stButton button {
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True)

    with col1:
        N = st.number_input('Number of periods (N)', value=100)

        # Création des modèles
        market = Market(S0, rate, sigma, dividend, ex_div_date)
        option = Option(K, T, option_type.lower(), option_style.lower())
        TreeModel = Tree(market, option, N)
        TreeModelGreeks = Tree(market, option, N)
        BSModel = BlackScholes(market, option)

        # Bouton pour générer le prix
        if st.button('Generate Price'):
            if light_mode:
                start_time = pd.Timestamp.now()
                TreeModel.light_build_tree(factor=factor)
                price = TreeModel.root.option_price
                price_time = (pd.Timestamp.now() - start_time).total_seconds()
                building_time = 0

            else:
                # Temps de construction de l'arbre
                start_time = pd.Timestamp.now()
                TreeModel.build_tree(factor=factor, threshold=threshold)
                building_time = (pd.Timestamp.now() - start_time).total_seconds()

                # Temps de calcul du prix de l'option
                start_time = pd.Timestamp.now()
                price = TreeModel.calculate_option_price()
                price_time = (pd.Timestamp.now() - start_time).total_seconds()

            total_time = building_time + price_time

            # Prix de Black-Scholes
            bsm_price = BSModel.option_price()

            # Écart entre les prix
            gap_price = abs(price - bsm_price) / bsm_price

            # Calcul des Greeks
            greeks = Greeks(TreeModelGreeks)

            if greeks.tree_model.N >= 1000:
                greeks.tree_model.N = 1000

            status = True

            delta, gamma, theta, vega, rho = greeks.get_greeks()
        else:
            price = bsm_price = gap_price = building_time = price_time = total_time = 0

    # Table de résultats
    with col2:
        # Créer une table HTML pour afficher les résultats
        table_html = f"""
        <table class="custom-table" align="right">
        <tr>
            <td><strong>Model Price</strong></td>
            <td>{price:.6f}</td>
            <td><strong>Building Time</strong></td>
            <td>{building_time:.5f}</td>
        </tr>
        <tr>
            <td><strong>Black-Scholes Price</strong></td>
            <td>{bsm_price:.6f}</td>
            <td><strong>Pricing Time</strong></td>
            <td>{price_time:.5f}</td>
        </tr>
        <tr>
            <td><strong>Gap</strong></td>
            <td>{gap_price:.6%}</td>
            <td><strong>Execution Time</strong></td>
            <td>{total_time:.5f}</td>
        </tr>
        </table>
        """
        st.markdown(table_html, unsafe_allow_html=True)

    # Affichage des Greeks si calculés
    if status:
        # Table HTML pour les Greeks
        greeks_table_html = f"""
        <style>
            .custom-table {{
                width: 100%;  /* Prendre toute la largeur */
                border-collapse: collapse;
            }}
            .custom-table td {{
                border: 1px solid #ddd;
                padding: 8px;
            }}
            .custom-table tr:nth-child(even) {{background-color: #f2f2f2;}}
            .custom-table tr:hover {{background-color: #ddd;}}
            .custom-table th {{
                padding-top: 12px;
                padding-bottom: 12px;
                text-align: left;
                background-color: #4CAF50;
                color: white;
            }}
        </style>
        <table class="custom-table">
            <tr>
                <td><strong>Delta</strong></td>
                <td>{delta:.6f}</td>
                <td><strong>Gamma</strong></td>
                <td>{gamma:.6f}</td>
                <td><strong>Theta</strong></td>
                <td>{theta:.6f}</td>
                <td><strong>Vega</strong></td>
                <td>{vega:.6f}</td>
                <td><strong>Rho</strong></td>
                <td>{rho:.6f}</td>
            </tr>
        </table>
        """
        st.markdown(greeks_table_html, unsafe_allow_html=True)

    # Graphique de l'arbre
    if not light_mode and hasattr(TreeModel, 'root'):
        st.markdown('<div class="center-text">Tree Chart</div>', unsafe_allow_html=True)
        if N <= 100:
            st.plotly_chart(TreeModel.visualize_tree(show=False))
        else:
            st.warning("The tree is too large to be displayed. Reduce the number of periods (N) to at least 100 to display the chart.")

# Contenu pour l'onglet "Metrics"
elif selected_tab == "Metrics":
    # Charger les fichiers HTML dans Streamlit
    for i in range(1, 7):
        components.html(open(f"Metrics/fig{i}.html", "r").read(), height=600)

# Contenu pour l'onglet "Greeks"
elif selected_tab == "Greeks":
    st.write("Contenu de l'onglet Greeks")
    # Ajoutez ici le contenu spécifique à l'onglet Greeks
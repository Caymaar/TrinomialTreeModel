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

st.set_page_config(layout="wide")

st.title('Trinomial Tree Model')

######################
# Sidebar
######################

st.sidebar.header('Parameters')

st.sidebar.write('Market Parameters')
S0 = st.sidebar.number_input('Initial price of the underlying asset (S0)', value=100.0)
rate = st.sidebar.number_input('Risk-free interest rate (r)', value=0.050)
sigma = st.sidebar.number_input('Volatility of the underlying asset (sigma)', value=0.200)
dividend = st.sidebar.number_input('Expected dividend (dividend)', value=0.0)
dividend_date = st.sidebar.date_input('Select a dividend date', value=date.today() + pd.Timedelta(days=90))

st.sidebar.write('Option Parameters')
K = st.sidebar.number_input('Option strike price (K)', value=100.0)
today = st.sidebar.date_input('Select a date', value=date.today())
maturity = st.sidebar.date_input('Select an expiration date', value=date.today() + pd.Timedelta(days=365))


T = (maturity - today).days / 365
ex_div_date = (dividend_date - today).days / 365
import streamlit as st
from streamlit_option_menu import option_menu

status = False

# CSS personnalisé pour centrer les titres
st.markdown("""
    <style>
    .centered-title {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Créer un menu horizontal pour les onglets
# Créer un menu horizontal pour les onglets
selected_tab = option_menu(
    menu_title=None,
    options=["Main", "Metrics", "Greeks"],
    icons=["house", "bar-chart", "calculator"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "transparent"},
        "icon": {"color": "black", "font-size": "20px"},  # Taille de l'icône réduite
        "nav-link": {
            "font-size": "20px",  # Taille de la police des étiquettes réduite
            "font-weight": "bold",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "#eee",
            "padding": "5px 10px",  # Ajuster le padding pour réduire la taille des boutons
            "width": "90%",  # Ajuster la largeur pour réduire la taille des boutons
        },
        "nav-link-selected": {"background-color": "#ff9999", "color": "white"},
    },
)

st.divider()

# Contenu pour l'onglet "Main"
if selected_tab == "Main":
    with st.container():

        # Créer deux colonnes pour les titres et les menus d'options
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

    if model == 'Trinomial Tree Model':
        light_mode = False
        
        # Créer deux colonnes pour Factor et Threshold
        col1, col2 = st.columns(2)

        # Paramètres du facteur dans la première colonne (col1)
        with col1:
            with st.expander("Factor"):
                st.markdown("You can adjust the factor to increase the pace of the tree building process. The factor should be between -1 and 1. 0 means no factor.")
                allow_factor = st.checkbox('Allowing factor')
                if allow_factor:
                    factor = st.slider('Factor', -0.99, 0.99, 0.00)
                    st.write(f"The selected Factor is : {factor}")
                else:
                    factor = 0

        # Paramètres du seuil dans la deuxième colonne (col2)
        with col2:
            with st.expander("Threshold"):
                st.markdown("You can adjust the threshold to control the precision of the calculations. The threshold should be close to zero.")
                allow_threshold = st.checkbox('Allow threshold')
                if allow_threshold:
                    decimal_places = st.slider('Number of decimal places', 2, 50, 10)
                    threshold = round(1 / 10**decimal_places, decimal_places)
                    st.write(f"The Threshold selected is : {threshold}")
                else:
                    threshold = 0 

    else:
        light_mode = True

        # Créer une seule colonne qui occupe toute la largeur pour Factor (col1 fait 100%)
        col1 = st.columns([1])[0]  # Une seule colonne

        # Paramètres du facteur dans la seule colonne (col1)
        with col1:
            with st.expander("Factor"):
                st.markdown("You can adjust the factor to increase the pace of the tree building process. The factor should be between -1 and 1. 0 means no factor.")
                allow_factor = st.checkbox('Allowing factor')
                if allow_factor:
                    factor = st.slider('Factor', -0.99, 0.99, 0.00)
                    st.write(f"The selected Factor is : {factor}")
                else:
                    factor = 0

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

        market = Market(S0, rate, sigma, dividend, ex_div_date)
        option = Option(K, T, option_type.lower(), option_style.lower())

        TreeModel = Tree(market, option, N)
        TreeModelGreeks = Tree(market, option, N)
        BSModel = BlackScholes(market, option)

        # Bouton qui prend toute la largeur de la colonne
        if st.button('Generate Price'):
            if light_mode:
                building_time = 0

                start_time = pd.Timestamp.now()
                TreeModel.light_build_tree(factor=factor)
                end_time = pd.Timestamp.now()

                price = TreeModel.root.option_price

                price_time = (end_time - start_time).total_seconds()

            else:
                # Temps de construction de l'arbre
                start_time = pd.Timestamp.now()
                TreeModel.build_tree(factor=factor, threshold=threshold)
                end_time = pd.Timestamp.now()

                building_time = (end_time - start_time).total_seconds()

                # Temps de calcul du prix de l'option
                start_time = pd.Timestamp.now()
                price = TreeModel.calculate_option_price()
                end_time = pd.Timestamp.now()

                price_time = (end_time - start_time).total_seconds()

            total_time = building_time + price_time

            # Prix de Black-Scholes
            bsm_price = BSModel.option_price()

            # Écart entre le prix du modèle d'arbre et le prix de Black-Scholes
            gap_price = abs(price - bsm_price) / bsm_price

            # Supposons que delta, gamma, theta, vega, et rho soient définis ailleurs dans votre code
            greeks = Greeks(TreeModelGreeks)

            if greeks.tree_model.N >= 1000:
                greeks.tree_model.N = 1000

            status = True

            delta, gamma, theta, vega, rho = greeks.get_greeks()
        else:
            price = 0
            bsm_price = 0
            gap_price = 0
            building_time = 0
            price_time = 0
            total_time = 0
        st.markdown('</div>', unsafe_allow_html=True)  # Fin du div pour centrer

    # Injecter du CSS pour décaler le tableau vers le bas
    st.markdown("""
        <style>
        .custom-table {
            margin-top: 24px;  /* Ajustez cette valeur pour déplacer le tableau plus ou moins bas */
        }
        </style>
        """, unsafe_allow_html=True)

    # Contenu dans la deuxième colonne (droite) avec le tableau HTML
    with col2:
        # Example data for HTML table
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

        # Display the HTML table in the second column
        st.markdown(table_html, unsafe_allow_html=True)
    
    if status:
        # Créer une table HTML pour afficher les valeurs des grecs
        # Créer une table HTML pour afficher les valeurs des grecs
        greeks_table_html = f"""
        <style>
            .custom-table {{
                width: 100%;  /* Forcer la table à prendre toute la largeur de la page */
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

        # Afficher la table HTML dans Streamlit
        st.markdown(greeks_table_html, unsafe_allow_html=True)

    if not light_mode and hasattr(TreeModel, 'root'):
        # Injecter du CSS pour centrer le texte
        st.markdown("""
            <style>
            .center-text {
                text-align: center;
                font-size: 20px;
                font-weight: bold;
            }
            </style>
        """, unsafe_allow_html=True)

        # Graphique de l'arbre
        if N <= 100:
            st.markdown('<div class="center-text">Tree Chart</div>', unsafe_allow_html=True)
            st.plotly_chart(TreeModel.visualize_tree(show=False))
        else:
            st.warning("The tree is too large to be displayed. Reduce the number of periods (N) to at least 100 to display the chart.")


  

# Contenu pour l'onglet "Metrics"
elif selected_tab == "Metrics":
    
    
    # Charger les fichiers HTML dans Streamlit
    components.html(open("Metrics/fig1.html", "r").read(), height=600)

    components.html(open("Metrics/fig2.html", "r").read(), height=600)

    components.html(open("Metrics/fig3.html", "r").read(), height=600)

    components.html(open("Metrics/fig4.html", "r").read(), height=600)

    components.html(open("Metrics/fig5.html", "r").read(), height=600)

    components.html(open("Metrics/fig6.html", "r").read(), height=600)

# Contenu pour l'onglet "Greeks"
elif selected_tab == "Greeks":
    st.write("Contenu de l'onglet Greeks")
    # Ajoutez ici le contenu spécifique à l'onglet Greeks




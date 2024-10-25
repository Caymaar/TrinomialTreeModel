# Trinomial Tree Model for Option Pricing

## Description
Ce projet implémente un modèle d'arbre trinomial pour évaluer les options financières, permettant de comparer les résultats avec le modèle Black-Scholes. Il inclut également des calculs des grecs (Delta, Gamma, Theta, Vega, Rho) ainsi que des métriques de performance.

## Structure du Projet

Le projet est organisé de la manière suivante :

├── Classes\\n
    ├── BlackScholes.py\\n
    ├── Greeks.py\n
    ├── Market.py\n
    ├── Metrics.py\n
    ├── Node.py\n
    ├── Option.py\n
    └── Tree.py\n
├── TrinomialTreeModel.xlsm\n
├── requirements.txt\n
├── streamlit_main.py\n
└── xlwings_main.py\n

### Détails des Classes

- **BlackScholes.py**: Contient la classe `BlackScholes` qui calcule le prix des options et leurs grecs selon le modèle Black-Scholes.
- **Greeks.py**: Contient la classe `Greeks` qui calcule les grecs d'une option à partir d'un modèle d'arbre trinomial.
- **Market.py**: Définit la classe `Market` qui représente le marché avec ses paramètres.
- **Metrics.py**: Contient la classe `Metrics` qui calcule et affiche les métriques de performance de l'arbre trinomial.
- **Node.py**: Définit la structure de chaque nœud de l'arbre trinomial.
- **Option.py**: Contient la classe `Option` qui définit les caractéristiques des options.
- **Tree.py**: Implémente la structure de l'arbre trinomial et les méthodes de calcul du prix des options.

### Fichiers Principaux

- **streamlit_main.py**: Interface utilisateur construite avec Streamlit pour interagir avec le modèle d'arbre trinomial, permettant de saisir les paramètres du marché et des options et d'afficher les résultats.
- **xlwings_main.py**: Intégration avec Excel pour interagir avec le modèle via un classeur Excel, permettant de récupérer des paramètres et d'afficher les résultats dans Excel.

## Installation

1. Clonez le dépôt :
   ```bash
   git clone <URL_DU_DEPOT>
   cd <NOM_DU_DOSSIER>
   ```

2.	Installez les dépendances requises :

  ```bash
   pip install -r requirements.txt
  ```
  Assurez-vous d’avoir installé les bibliothèques suivantes : numpy, scipy, pandas, plotly, streamlit, xlwings et psutil.

3.	Exécutez l’application Streamlit :

  ```bash
   streamlit run streamlit_main.py
  ```

Ou, pour utiliser Excel avec xlwings :
Ouvrez votre classeur Excel et exécutez les fonctions définies dans xlwings_main.py.

Utilisation

- Streamlit : Entrez les paramètres du marché et de l’option dans l’interface utilisateur et cliquez sur “Generate Price” pour calculer le prix de l’option. Les grecs et les métriques seront également affichés.
- Excel : Remplissez les cellules du classeur Excel avec les paramètres appropriés et exécutez les fonctions fournies pour obtenir les prix des options et les grecs.

  

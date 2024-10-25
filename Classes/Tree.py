import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Classes.Node import Node
import plotly.graph_objs as go
import networkx as nx

class Tree:
    """
    Classe représentant un arbre trinomial pour le pricing des options.
    
    Attributes:
        market: Un objet Market contenant les paramètres du marché.
        option: Un objet Option contenant les paramètres de l'option.
        N: Nombre d'étapes dans l'arbre.
        deltaT_array: Tableau contenant les intervalles de temps (deltaT) pour chaque étape.
        dividend_step: Étape à laquelle un dividende est distribué.
        root: Nœud racine de l'arbre.
        last_trunc: Dernier nœud de tronc dans l'arbre.
        number_of_nodes: Nombre total de nœuds dans l'arbre.
        threshold: Seuil pour le calcul des probabilités.
    """

    def __init__(self, market, option, N):
        """
        Initialise un arbre trinomial avec les paramètres du marché, de l'option et le nombre d'étapes.
        
        Args:
            market (Market): Paramètres du marché.
            option (Option): Paramètres de l'option.
            N (int): Nombre d'étapes dans l'arbre.
        """
        self.market = market
        self.option = option
        self.N = N

    def get_deltaT(self, factor=0):
        """
        Calcule les intervalles de temps (deltaT) normalisés pour chaque étape.
        
        Args:
            factor (float): Facteur d'échelle exponentielle pour deltaT.
        """
        exp_factors = np.exp(np.linspace(0, factor, self.N))  # Facteurs d'échelle exponentielle
        self.deltaT_array = exp_factors / np.sum(exp_factors) * self.option.T  # Normalisation pour que la somme soit égale à T

    def dividend_value(self, step):
        """
        Renvoie la valeur du dividende à l'étape spécifiée.
        
        Args:
            step (int): Étape pour laquelle la valeur du dividende est demandée.
        
        Returns:
            float: Valeur du dividende si c'est l'étape du dividende, sinon 0.
        """
        if self.dividend_step == step:
            return self.market.dividend
        return 0.0

    def build_tree(self, factor=0, threshold=0):
        """
        Construit l'arbre trinomial en générant les nœuds et leurs voisins.
        
        Args:
            factor (float): Facteur d'échelle pour deltaT.
            threshold (float): Seuil pour le calcul des probabilités.
        """
        self.root = Node(self.market.S0, 0, self)  # Création de la racine de l'arbre
        self.root.cum_prob = 1.0
        self.number_of_nodes = 0

        self.threshold = threshold
            
        self.get_deltaT(factor)  # Calcul des intervalles de temps
        self.dividend_step = math.ceil((self.market.ex_div_date / self.option.T) * self.N)  # Étape du dividende

        trunc = self.root

        for _ in range(0, self.N, 1): 
            # Création des voisins pour chaque étape
            trunc.create_forward_neighbors()
            trunc.generate_upper_neighbors()
            trunc.generate_lower_neighbors()
            trunc = trunc.forward_mid_neighbor  # Passage au nœud suivant

        self.last_trunc = trunc  # Dernier nœud de tronc

    def compute_payoff(self):
        """
        Calcule le payoff de l'option pour chaque nœud de l'arbre.
        
        Returns:
            Node: Le dernier nœud de tronc après le calcul des payoffs.
        """
        last_node = trunc = self.last_trunc

        trunc.option_price = trunc.tree.option.payoff(last_node.value)  # Calcul du prix à maturité

        while last_node.up_neighbor is not None:
            last_node = last_node.up_neighbor
            last_node.option_price = last_node.tree.option.payoff(last_node.value)

        last_node = trunc

        while last_node.down_neighbor is not None:
            last_node = last_node.down_neighbor
            last_node.option_price = last_node.tree.option.payoff(last_node.value)

        return trunc
    
    def backpropagation(self):
        """
        Exécute la rétropropagation pour calculer les prix de l'option à partir des feuilles jusqu'à la racine.
        """
        last_trunc = self.last_trunc

        while last_trunc.backward_neighbor is not None:
            last_trunc = last_trunc.backward_neighbor
            last_trunc.compute_option_price()

            node_to_compute = last_trunc

            while node_to_compute.up_neighbor is not None:
                node_to_compute = node_to_compute.up_neighbor
                node_to_compute.compute_option_price()

            node_to_compute = last_trunc

            while node_to_compute.down_neighbor is not None:
                node_to_compute = node_to_compute.down_neighbor
                node_to_compute.compute_option_price()

    def calculate_option_price(self):
        """
        Calcule le prix de l'option à partir de l'arbre.

        Returns:
            float: Prix de l'option au niveau de la racine.
        """
        self.compute_payoff()  # Calcul du payoff
        self.backpropagation()  # Calcul des prix des nœuds
        return self.root.option_price
    
    ########################### Light Version ###########################

    def light_build_tree(self, factor=0, NbSigma=5.63):
        """
        Construit l'arbre trinomial en version légère.
        
        Args:
            factor (float): Facteur d'échelle pour deltaT.
            NbSigma (float): Nombre d'écarts-types pour la génération des voisins.
        """
        self.root = Node(self.market.S0, 0, self)  # Création de la racine
        self.root.cum_prob = 1.0
        self.number_of_nodes = 0
            
        self.get_deltaT(factor)  # Calcul des intervalles de temps
        self.dividend_step = math.ceil((self.market.ex_div_date / self.option.T) * self.N)  # Étape du dividende

        trunc = self.root
        
        for _ in range(0, self.N, 1): 
            # Création des voisins en version légère
            trunc.light_forward_neighbor()
            trunc = trunc.forward_mid_neighbor

        for i in range(self.N, 0, -1): 
            trunc.light_generating_neighbors(NbSigma)  # Génération des voisins
            if trunc.step != self.N:
                trunc.light_manage_forward_for_trunc()  # Gestion des voisins en tronc
            trunc = trunc.backward_neighbor

        trunc.light_manage_forward_for_trunc()  # Finalisation de la gestion des voisins

    ########################### Utilitaire ###########################

    def get_number_of_nodes(self):
        """
        Renvoie le nombre total de nœuds dans l'arbre.
        
        Returns:
            int: Nombre total de nœuds.
        """
        self.number_of_nodes = 0

        trunc = self.root

        for _ in range(0, self.N + 1):
            current_node = trunc

            # Compter les nœuds en descendant
            while current_node is not None:
                self.number_of_nodes += 1
                current_node = current_node.down_neighbor

            current_node = trunc 
            
            # Compter les nœuds en montant
            while current_node is not None:
                self.number_of_nodes += 1
                current_node = current_node.up_neighbor

            trunc = trunc.forward_mid_neighbor

        return self.number_of_nodes

    def get_number_of_nodes_per_step(self):
        """
        Renvoie un dictionnaire contenant le nombre de nœuds à chaque étape.
        
        Returns:
            dict: Dictionnaire avec le nombre de nœuds par étape.
        """
        # Dictionnaire pour stocker le nombre de nœuds par étape
        nodes_per_step = {}

        trunc = self.root

        for step in range(0, self.N + 1):
            count = 0  # Compteur pour le nombre de nœuds à cette étape

            current_node = trunc

            # Compter les nœuds en descendant
            while current_node is not None:
                count += 1
                current_node = current_node.down_neighbor

            current_node = trunc

            # Compter les nœuds en montant
            while current_node is not None:
                count += 1
                current_node = current_node.up_neighbor

            # Stocker le nombre de nœuds pour cette étape dans le dictionnaire
            nodes_per_step[step] = count

            # Passer au nœud suivant dans la direction forward_mid_neighbor
            trunc = trunc.forward_mid_neighbor

        return nodes_per_step

    def get_option_price(self, light_mode=False, factor=0, threshold=0, NbSigma=5.63):
        """
        Renvoie le prix de l'option en fonction du mode (léger ou complet).
        
        Args:
            light_mode (bool): Indique si le mode léger est activé.
            factor (float): Facteur d'échelle pour deltaT (en mode léger).
            threshold (float): Seuil pour les calculs de probabilités (en mode complet).
            NbSigma (float): Nombre d'écarts-types pour la génération des voisins (en mode léger).
        
        Returns:
            float: Prix de l'option calculé.
        """
        if light_mode:
            self.light_build_tree(factor=factor, NbSigma=NbSigma)
            return self.root.option_price
        else:
            self.build_tree(factor=factor, threshold=threshold)
            return self.calculate_option_price()

    def error_interval(self, price, verbose=False):
        """
        Calcule l'intervalle d'erreur pour le prix donné.
        
        Args:
            price (float): Prix de l'option.
            verbose (bool): Indique si le message détaillé doit être affiché.
        
        Returns:
            tuple: Bornes inférieure et supérieure de l'intervalle d'erreur.
        """
        gap = (3 * self.market.S0 * math.sqrt(math.exp(self.market.sigma ** 2 * self.option.T) - 1)) / (8 * math.sqrt(2 * math.pi) * self.N * self.market.sigma ** 2)
        
        ratio = gap / self.market.S0

        lower_bound = price * (1 - ratio)
        upper_bound = price * (1 + ratio)
        
        if verbose:
            print(f"{round(price, 5)} ∈ {{{round(lower_bound, 5)}; {round(upper_bound, 5)}}}")
        else:
            return lower_bound, upper_bound

    ########################### Visualisation ###########################

    def traverse_tree(self, node, G, pos):
        """
        Parcourt récursivement l'arbre pour ajouter les nœuds et les liens au graphe.
        
        Args:
            node (Node): Le nœud actuel à parcourir.
            G (networkx.Graph): Le graphe où ajouter les nœuds et les arêtes.
            pos (dict): Dictionnaire des positions des nœuds pour l'affichage.
        """
        if node is None or node in G:
            return

        G.add_node(node)
        # Pour une progression de gauche à droite, x correspond au deltaT cumulé, y correspond à node.value
        cumulative_deltaT = sum(self.deltaT_array[:node.step])
        pos[node] = (cumulative_deltaT, node.value)

        # Liste des voisins
        neighbors = [
            node.forward_up_neighbor,
            node.forward_mid_neighbor,
            node.forward_down_neighbor
        ]

        for neighbor in neighbors:
            if neighbor is not None:
                # Parcourir le voisin
                self.traverse_tree(neighbor, G, pos)
                # Ajouter l'arête après avoir ajouté le voisin
                G.add_edge(node, neighbor)

    def visualize_tree(self, show=True):
        """
        Visualise l'arbre trinomial en utilisant Plotly.
        
        Args:
            show (bool): Indique si la figure doit être affichée.
        
        Returns:
            go.Figure: Figure Plotly contenant la visualisation de l'arbre.
        """
        G = nx.DiGraph()
        pos = {}
        self.traverse_tree(self.root, G, pos)

        # Préparer les données pour Plotly
        edge_x = []
        edge_y = []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color='black'),
            hoverinfo='none',
            mode='lines'
        )

        node_x = []
        node_y = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        # Définir une taille fixe pour les nœuds, ajustée en fonction de N
        node_size = max(5, 100 / self.N)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            hoverinfo='none',
            marker=dict(
                color='lightblue',
                size=node_size,
                line_width=0))

        # Créer la figure
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(
                                title='cumulated DeltaT',
                                showgrid=False,
                                zeroline=False,
                                showticklabels=True),
                            yaxis=dict(
                                title='Node Value',
                                showgrid=False,
                                zeroline=False,
                                showticklabels=True)
                        )
                        )

        if show:
            fig.show()
        else:
            return fig
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Classes.Node import Node
import plotly.graph_objs as go
import networkx as nx



class Tree:
    def __init__(self, market, option, N):
        self.market = market
        self.option = option
        self.N = N

    def get_deltaT(self, factor=0):
        exp_factors = np.exp(np.linspace(0, factor, self.N))  # Exponential scaling factors
        self.deltaT_array = exp_factors / np.sum(exp_factors) * self.option.T  # Normalize to make sure the sum equals T

    def dividend_value(self, step):
        if self.dividend_step == step:
            return self.market.dividend
        return 0.0

    def build_tree(self, factor=0, threshold=0):

        self.root = Node(self.market.S0, 0, self)
        self.root.cum_prob = 1.0
        self.number_of_nodes = 0

        self.threshold = threshold
            
        self.get_deltaT(factor)
        self.dividend_step = math.ceil((self.market.ex_div_date / self.option.T) * self.N)

        trunc = self.root

        for _ in range(0, self.N, 1): 

            trunc.create_forward_neighbors()

            trunc.generate_upper_neighbors()
            trunc.generate_lower_neighbors()

            trunc = trunc.forward_mid_neighbor

        self.last_trunc = trunc

    def compute_payoff(self):

        last_node = trunc = self.last_trunc

        trunc.option_price = trunc.tree.option.payoff(last_node.value)

        while last_node.up_neighbor is not None:
            last_node = last_node.up_neighbor
            last_node.option_price = last_node.tree.option.payoff(last_node.value)

        last_node = trunc

        while last_node.down_neighbor is not None:
            last_node = last_node.down_neighbor
            last_node.option_price = last_node.tree.option.payoff(last_node.value)

        return trunc
    
    def backpropagation(self):

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

        self.compute_payoff()

        self.backpropagation()

        return self.root.option_price
    
########################### Light Version ###########################

    def light_build_tree(self, factor=0, NbSigma=5.63):

        self.root = Node(self.market.S0, 0, self)
        self.root.cum_prob = 1.0
        self.number_of_nodes = 0
            
        self.get_deltaT(factor)
        self.dividend_step = math.ceil((self.market.ex_div_date / self.option.T) * self.N)

        trunc = self.root
        
        for _ in range(0, self.N, 1): 

            trunc.light_forward_neighbor()

            trunc = trunc.forward_mid_neighbor

        for i in range(self.N, 0, -1): 

            trunc.light_generating_neighbors(NbSigma)
 
            if trunc.step != self.N:

                trunc.light_manage_forward_for_trunc()

            trunc = trunc.backward_neighbor

        trunc.light_manage_forward_for_trunc()

########################### Utilitaire ###########################

    def get_number_of_nodes(self):

        """if self.number_of_nodes != 0:
            return self.number_of_nodes"""

        self.number_of_nodes = 0

        trunc = self.root

        for _ in range(0, self.N + 1):
            
            current_node = trunc

            while current_node is not None:
                self.number_of_nodes += 1
                current_node = current_node.down_neighbor

            current_node = trunc 
            
            while current_node is not None:
                self.number_of_nodes += 1
                current_node = current_node.up_neighbor

            trunc = trunc.forward_mid_neighbor

        return self.number_of_nodes

    def get_number_of_nodes_per_step(self):
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

        if light_mode:
            self.light_build_tree(factor=factor, NbSigma=NbSigma)
            return self.root.option_price
        else:
            self.build_tree(factor=factor, threshold=threshold)
            return self.calculate_option_price()

    def error_interval(self, price, verbose=False):
        gap = (3 * self.S0 * math.sqrt(math.exp(self.sigma ** 2 * self.T) - 1)) / (8 * math.sqrt(2 * math.pi) * self.N * self.sigma ** 2)
        
        ratio = gap / self.S0

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

        Parameters:
        - node: le nœud actuel
        - G: le graphe NetworkX
        - pos: positions des nœuds pour l'affichage
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

    # Ancienne méthode de visualisation (Ne prenait pas bien en compte les connexions)
    """def save_tree_to_dataframe(self):
    
            max_columns = self.N + 1 
            max_rows = 2 * self.N + 1 
            extra_rows = int(math.ceil(self.N * 0.05))
            df_mid = pd.DataFrame(index=range(max_rows + extra_rows), columns=range(max_columns))
            df_up = pd.DataFrame(index=range(max_rows + extra_rows), columns=range(max_columns))
            df_down = pd.DataFrame(index=range(max_rows + extra_rows), columns=range(max_columns))

            trunc = self.root

            for step in range(0, self.N + 1):
                
                column = step 
                current_row_df1 = self.N 
                current_row_df2 = current_row_df3 = step 
                current_node = trunc

                while current_node is not None:
                    df_mid.iloc[current_row_df1, column] = current_node.value
                    df_up.iloc[current_row_df2, column] = current_node.value
                    df_down.iloc[current_row_df3, column] = current_node.value

                    current_row_df1 += 1
                    current_row_df2 += 1
                    current_row_df3 -= 1

                    current_node = current_node.down_neighbor

                current_row_df1 = self.N  
                current_row_df2 = current_row_df3 = step 
                current_node = trunc 
                
                while current_node is not None:
                    df_mid.iloc[current_row_df1, column] = current_node.value
                    df_up.iloc[current_row_df2, column] = current_node.value
                    df_down.iloc[current_row_df3, column] = current_node.value

                    current_row_df1 -= 1
                    current_row_df2 -= 1
                    current_row_df3 += 1
                    
                    current_node = current_node.up_neighbor

                trunc = trunc.forward_mid_neighbor
                
            self.number_of_nodes = df_mid.count().sum()

            time_steps = np.cumsum(np.insert(self.deltaT_array, 0, 0))
            
            df_mid.columns = df_down.columns = df_up.columns = time_steps

            return df_mid.T, df_up.T, df_down.T
        
        def visualize_tree(self, style='ko-'):

            line_width = max(0.5, 4.0 / self.N)  
            marker_size = max(2, 10.0 / (self.N / 5))

            plt.figure(figsize=(min(20, max(self.N * 0.4, 14)), min(10, max(self.N * 0.2, 7))))  

            df_mid, df_up, df_down = self.save_tree_to_dataframe()

            plt.plot(df_down, style, linewidth=line_width, markersize=marker_size)
            plt.plot(df_up, style, linewidth=line_width, markersize=marker_size)
            plt.plot(df_mid, style, linewidth=line_width, markersize=marker_size)

            # Met des points bleue légèrement plus gros sur les mid de l'arbre(CaD la colonne avec la plus de valeurs non nulles)
            plt.plot(df_mid[df_mid.count().idxmax()],'o', linewidth=line_width, markersize=marker_size+3, color='blue')

            # Set plot labels
            plt.xlabel("Maturity (Cumulative)")
            plt.ylabel("Stock Price")
            # Add a title
            plt.title(f"Trinomial Tree for {self.option.type} option with {self.option.style} style - {self.N} steps")

            return plt

        def visualize_tree_plotly(self):
            # Définir une couleur commune pour toutes les lignes et la taille des lignes et des marqueurs en fonction de N
            common_color = 'grey'
            line_width = max(0.5, 4.0 / self.N)  # Ajuster la taille des lignes en fonction de N
            marker_size = max(2, 10.0 / (self.N / 5))  # Ajuster la taille des marqueurs en fonction de N

            # Obtenir les DataFrames pour l'arbre
            df_mid, df_up, df_down = self.save_tree_to_dataframe()

            # Création de la figure
            fig = go.Figure()

            # Tracer df_mid
            for i in range(len(df_mid.T)):  # Transpose (T) pour itérer sur les lignes
                fig.add_trace(go.Scatter(
                    x=df_mid.T.columns,
                    y=df_mid.T.iloc[i],
                    mode='lines+markers',
                    line=dict(color=common_color, width=line_width),
                    marker=dict(size=marker_size),  # Taille des marqueurs ajustée
                    showlegend=False
                ))

            # Tracer df_up
            for i in range(len(df_up.T)):
                fig.add_trace(go.Scatter(
                    x=df_up.T.columns,
                    y=df_up.T.iloc[i],
                    mode='lines+markers',
                    line=dict(color=common_color, width=line_width),
                    marker=dict(size=marker_size),  # Taille des marqueurs ajustée
                    showlegend=False
                ))

            # Tracer df_down
            for i in range(len(df_down.T)):
                fig.add_trace(go.Scatter(
                    x=df_down.T.columns,
                    y=df_down.T.iloc[i],
                    mode='lines+markers',
                    line=dict(color=common_color, width=line_width),  
                    marker=dict(size=marker_size),  # Taille des marqueurs ajustée
                    showlegend=False
                ))

            # Met des points bleus légèrement plus gros sur les mid de l'arbre (colonne avec la plus de valeurs non nulles)
            fig.add_trace(go.Scatter(
                x=df_mid[df_mid.count().idxmax()].index,
                y=df_mid[df_mid.count().idxmax()],
                mode='markers',
                marker=dict(color='blue', size=marker_size * 2),  # Taille légèrement plus grande pour ces marqueurs
                showlegend=False
            ))

            # Ajouter une mise en page pour le titre et les axes
            fig.update_layout(
                title=f"Trinomial Tree for {self.option.type} option with {self.option.style} style - {self.N} steps",
                xaxis_title="Nœuds",
                yaxis_title="Prix de l'option",
                showlegend=False
            )

            return fig"""

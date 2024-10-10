import plotly.graph_objects as go
from Classes.Tree import Tree
from Classes.Market import Market
from Classes.Option import Option
import numpy as np
import pandas as pd
from Classes.BlackScholes import BlackScholes
import psutil

class Metrics:
    def __init__(self, tree_model):
        self.tree_model = tree_model
        self.tree_model.dividend = 0
        self.bsm_model = BlackScholes(tree_model.market, tree_model.option)

    def get_metrics(self, light_mode=False, NbSigma=5.63):
        if light_mode:
            starting_time = pd.Timestamp.now()
            price = self.tree_model.get_option_price(light_mode=True, NbSigma=NbSigma)
            ending_time = pd.Timestamp.now()

            building_time = 0
            pricing_time = (ending_time - starting_time).total_seconds()

            number_of_nodes = 1

            building_time_by_node = 0
            pricing_time_by_node = pricing_time

            memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # en Mo

        else:
            starting_time = pd.Timestamp.now()
            self.tree_model.build_tree()  # Assurez-vous que cette méthode est définie dans TrinomialTreeModel
            ending_time = pd.Timestamp.now()

            building_time = (ending_time - starting_time).total_seconds()

            memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # en Mo

            starting_time = pd.Timestamp.now()
            #price = self.tree_model.calculate_option_price()
            price = self.tree_model.calculate_option_price()
            ending_time = pd.Timestamp.now()

            pricing_time = (ending_time - starting_time).total_seconds()

            number_of_nodes = self.tree_model.get_number_of_nodes()

            building_time_by_node = (building_time) / number_of_nodes
            pricing_time_by_node = (pricing_time) / number_of_nodes

        return building_time, pricing_time, building_time_by_node, pricing_time_by_node, number_of_nodes, price, memory_usage

    def compute_and_show_metrics(self, N_values, light_mode=False, show=True):

        building_times = []
        pricing_times = []
        building_times_by_node = []
        pricing_times_by_node = []
        number_of_nodes_list = []
        prices = []
        differences = []
        execution_times = []
        times_by_node = []
        memory_usages = []
        
        for N in N_values:
            self.tree_model.N = N

            building_time, pricing_time, building_time_by_node, pricing_time_by_node, number_of_nodes, price, memory_usage = self.get_metrics(light_mode)

            building_times.append(building_time)
            pricing_times.append(pricing_time)
            building_times_by_node.append(building_time_by_node)
            pricing_times_by_node.append(pricing_time_by_node)
            number_of_nodes_list.append(number_of_nodes)
            prices.append(price)
            execution_times.append(building_time + pricing_time)
            times_by_node.append(building_time_by_node + pricing_time_by_node)
            memory_usages.append(memory_usage)

        prices = np.array(prices)
        execution_times = np.array(execution_times)

        # Utiliser le prix du modèle Black-Scholes
        bsm_price = self.bsm_model.option_price()

        # Calcul de la différence entre le modèle Trinomial et Black-Scholes, multipliée par N
        differences = []
        for i, price in enumerate(prices):
            diff = (price - bsm_price) * N_values[i]
            differences.append(diff)

        # Graph 1: Execution time with curves for building time, pricing time, and total execution time
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=list(N_values), y=building_times, mode='lines+markers', name='Building Time'))
        fig1.add_trace(go.Scatter(x=list(N_values), y=pricing_times, mode='lines+markers', name='Pricing Time'))
        fig1.add_trace(go.Scatter(x=list(N_values), y=execution_times, mode='lines+markers', name='Total Execution Time'))
        fig1.update_layout(
            title="Execution Time as a Function of the Number of Steps N",
            xaxis_title="N (number of steps in the trinomial tree)",
            yaxis_title="Time (seconds)",
            legend_title="Time",
            template="plotly_white"
        )

        # Graph 2: Convergence of trinomial model towards Black-Scholes
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=list(N_values), y=prices, mode='lines+markers', name='Trinomial Tree'))
        fig2.add_trace(go.Scatter(x=list(N_values), y=[bsm_price]*len(N_values), mode='lines', name='Black-Scholes'))
        fig2.update_layout(
            title="Convergence of Trinomial Model to Black-Scholes",
            xaxis_title="N (number of steps in the trinomial tree)",
            yaxis_title="Option Price",
            legend_title="Models",
            template="plotly_white"
        )

        # Graph 3: (Trinomial Value – Black-Scholes Value) x N
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=list(N_values), y=differences, mode='lines+markers', name='(Trinomial Value - Black-Scholes Value) x N'))
        fig3.update_layout(
            title="(Trinomial Value – Black-Scholes Value) x N",
            xaxis_title="N (number of steps in the trinomial tree)",
            yaxis_title="Difference x N",
            template="plotly_white"
        )

        # Graph 4: Memory usage as a function of N
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=list(N_values), y=memory_usages, mode='lines+markers', name='Memory Usage'))
        fig4.update_layout(
            title="Memory Usage as a Function of the Number of Steps N",
            xaxis_title="N (number of steps in the trinomial tree)",
            yaxis_title="Memory Usage (MB)",
            legend_title="Memory",
            template="plotly_white"
        )

        # Graph 5: Number of nodes for each N
        fig5 = go.Figure()
        if not light_mode:
            fig5.add_trace(go.Scatter(x=list(N_values), y=number_of_nodes_list, mode='lines+markers', name='Number of Nodes'))
            fig5.update_layout(
                title="Number of Nodes as a Function of the Number of Steps N",
                xaxis_title="N (number of steps in the trinomial tree)",
                yaxis_title="Number of Nodes",
                template="plotly_white"
            )

        # Graph 6: Time per node for each N
        fig6 = go.Figure()
        if not light_mode:
            fig6.add_trace(go.Scatter(x=list(N_values), y=times_by_node, mode='lines+markers', name='Total Time per Node'))
            fig6.add_trace(go.Scatter(x=list(N_values), y=building_times_by_node, mode='lines+markers', name='Building Time per Node'))
            fig6.add_trace(go.Scatter(x=list(N_values), y=pricing_times_by_node, mode='lines+markers', name='Pricing Time per Node'))
            fig6.update_layout(
                title="Time per Node as a Function of the Number of Steps N",
                xaxis_title="N (number of steps in the trinomial tree)",
                yaxis_title="Time per Node (seconds)",
                legend_title="Time per Node",
                template="plotly_white"
            )


        if show:
            fig1.show()
            fig2.show()
            fig3.show()
            fig4.show()
            fig5.show()
            fig6.show()
        else:
            return fig1, fig2, fig3, fig4, fig5, fig6
                
                    
    def compute_and_show_sigma_convergence(self, Sigma_values, N_values, show=True):
        convergence_prices = {sigma: [] for sigma in Sigma_values}  # Dictionnaire pour stocker les prix pour chaque sigma
        execution_times = {sigma: [] for sigma in Sigma_values}  # Dictionnaire pour stocker les temps d'exécution pour chaque sigma
        
        original_N = self.tree_model.N

        # Parcourir toutes les valeurs de N
        for N in N_values:
            self.tree_model.N = N

            for sigma in Sigma_values:
                # Mesurer le temps de calcul et obtenir le prix de l'option
                start_time = pd.Timestamp.now()
                price = self.tree_model.get_option_price(light_mode=True, NbSigma=sigma)
                end_time = pd.Timestamp.now()

                # Calculer le temps d'exécution
                exec_time = (end_time - start_time).total_seconds()

                # Stocker les prix et les temps pour ce N et ce sigma
                convergence_prices[sigma].append(price)
                execution_times[sigma].append(exec_time)

        # Restaurer la valeur originale de N
        self.tree_model.N = original_N

        # Obtenir le prix du modèle Black-Scholes pour comparaison
        bsm_price = self.bsm_model.option_price()

        # Graph 1: Convergence pour différents N et sigma
        fig1 = go.Figure()
        for sigma in Sigma_values:
            fig1.add_trace(go.Scatter(x=list(N_values), y=convergence_prices[sigma], mode='lines+markers', name=f'Sigma={sigma}'))
        fig1.add_trace(go.Scatter(x=list(N_values), y=[bsm_price]*len(N_values), mode='lines', name='Black-Scholes'))
        fig1.update_layout(
            title="Convergence of Trinomial Model to Black-Scholes for Different Sigma and N",
            xaxis_title="N (number of steps in the trinomial tree)",
            yaxis_title="Option Price",
            legend_title="Sigma Values",
            template="plotly_white"
        )

        # Graph 2: Temps d'exécution pour chaque combinaison de N et sigma
        fig2 = go.Figure()
        for sigma in Sigma_values:
            fig2.add_trace(go.Scatter(x=list(N_values), y=execution_times[sigma], mode='lines+markers', name=f'Sigma={sigma}'))
        fig2.update_layout(
            title="Execution Time for Different Sigma and N Values",
            xaxis_title="N (number of steps in the trinomial tree)",
            yaxis_title="Execution Time (seconds)",
            legend_title="Sigma Values",
            template="plotly_white"
        )

        if show:
            fig1.show()
            fig2.show()
        else:
            return fig1, fig2
        


    def compute_and_show_factor_convergence(self, Factor_values, N_values, show=True):
        convergence_prices = {factor: [] for factor in Factor_values}  # Dictionnaire pour stocker les prix pour chaque sigma
        
        original_N = self.tree_model.N

        # Parcourir toutes les valeurs de N
        for N in N_values:
            self.tree_model.N = N

            for factor in Factor_values:
                # Obtenir le prix de l'option
                price = self.tree_model.get_option_price(light_mode=True, factor=factor)

                # Stocker les prix et les temps pour ce N et ce sigma
                convergence_prices[factor].append(price)

        # Restaurer la valeur originale de N
        self.tree_model.N = original_N

        # Obtenir le prix du modèle Black-Scholes pour comparaison
        bsm_price = self.bsm_model.option_price()

        # Graph 1: Convergence pour différents N et sigma
        fig1 = go.Figure()
        for factor in Factor_values:
            fig1.add_trace(go.Scatter(x=list(N_values), y=convergence_prices[factor], mode='lines+markers', name=f'Factor={factor}'))
        fig1.add_trace(go.Scatter(x=list(N_values), y=[bsm_price]*len(N_values), mode='lines', name='Black-Scholes'))
        fig1.update_layout(
            title="Convergence of Trinomial Model to Black-Scholes for Different Factor and N",
            xaxis_title="N (number of steps in the trinomial tree)",
            yaxis_title="Option Price",
            legend_title="Sigma Values",
            template="plotly_white"
        )

        if show:
            fig1.show()
        else:
            return fig1
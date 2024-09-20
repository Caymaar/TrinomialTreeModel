import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Classes.Node import Node
    
class Tree:
    def __init__(self, market, option, N):
        self.market = market
        self.option = option
        self.N = N
        self.root = Node(market.S0, 0, self)
        self.root.cum_prob = 1.0
        self.number_of_nodes = 0

    def get_deltaT(self, factor=0):
        exp_factors = np.exp(np.linspace(0, factor, self.N))  # Exponential scaling factors
        self.deltaT_array = exp_factors / np.sum(exp_factors) * self.option.T  # Normalize to make sure the sum equals T

    def dividend_value(self, step):
        if self.dividend_step == step:
            return self.market.dividend
        return 0.0

    def build_tree(self, factor=0, threshold=0):

        if self.root.forward_mid_neighbor is not None:
            self.root = Node(self.market.S0, 0, self)
            self.root.cum_prob = 1.0

        self.threshold = threshold
            
        self.get_deltaT(factor)
        self.dividend_step = math.ceil(self.market.ex_div_date * self.N)

        trunc = self.root

        for step in range(0, self.N, 1): 

            trunc.create_forward_nodes()

            trunc.generate_upper_neighbors()
            trunc.generate_lower_neighbors()

            trunc = trunc.forward_mid_neighbor

        self.last_trunc = trunc 

    def compute_payoff(self):

        last_node = trunc = self.last_trunc

        trunc.payoff()

        while last_node.up_neighbor is not None:
            last_node = last_node.up_neighbor
            last_node.payoff()

        last_node = trunc

        while last_node.down_neighbor is not None:
            last_node = last_node.down_neighbor
            last_node.payoff()

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
    
    def save_tree_to_dataframe(self):
 
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

        return plt

    def get_number_of_nodes(self):

        if self.number_of_nodes != 0:
            return self.number_of_nodes

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
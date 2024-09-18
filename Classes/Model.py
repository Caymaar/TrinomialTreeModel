import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sps
from Classes.Market import Market
from Classes.Option import Option
from Classes.Node import Node

class BlackScholesModel(Market, Option):

    def __init__(self, S0, rate, sigma, K, T, option_type, option_style):
        Market.__init__(self, S0, rate, sigma)
        Option.__init__(self, K, T, option_type, option_style)

    def option_price(self):
        d1 = (np.log(self.S0 / self.K) + (self.rate + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        if self.option_type == "call":
            option_price = self.S0 * sps.norm.cdf(d1) - self.K * np.exp(-self.rate * self.T) * sps.norm.cdf(d2)
        else:
            option_price = self.K * np.exp(-self.rate * self.T) * sps.norm.cdf(-d2) - self.S0 * sps.norm.cdf(-d1)

        return option_price
    
class TrinomialTreeModel(Market, Option):
    
    def __init__(self, S0, rate, sigma, K, T, option_type, option_style, N, dividend=0.0, ex_div_date=None):
        Market.__init__(self, S0, rate, sigma, dividend, ex_div_date)
        Option.__init__(self, K, T, option_type, option_style)
        self.N = N
        self.root = Node(S0, 0, self)
        self.root.cum_prob = 1.0

    def get_deltaT(self, factor=0):
        exp_factors = np.exp(np.linspace(0, factor, self.N))  # Exponential scaling factors
        self.deltaT_array = exp_factors / np.sum(exp_factors) * self.T  # Normalize to make sure the sum equals T

    def dividend_value(self, step):
        if self.dividend_step == step:
            return self.dividend
        return 0.0

    def build_tree(self, factor=0, threshold=0):

        if self.root.forward_mid_neighbor is not None:
            self.root = Node(self.S0, 0, self)
            self.root.cum_prob = 1.0

        self.threshold = threshold
            
        self.get_deltaT(factor)
        self.dividend_step = math.ceil(self.ex_div_date * self.N)

        trunc = self.root

        for _ in range(0, self.N, 1): 

            trunc.create_forward_nodes()

            trunc.generate_upper_neighbors()
            trunc.generate_lower_neighbors()

            trunc = trunc.forward_mid_neighbor
                
    def compute_payoff_and_return_last_trunc(self):

        trunc = self.root

        while trunc.forward_mid_neighbor is not None:
            trunc = trunc.forward_mid_neighbor

        last_node = trunc

        trunc.payoff()

        while last_node.up_neighbor is not None:
            last_node = last_node.up_neighbor
            last_node.payoff()

        last_node = trunc

        while last_node.down_neighbor is not None:
            last_node = last_node.down_neighbor
            last_node.payoff()

        return trunc
    
    def back_propagation_pricing(self, last_trunc):

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

        last_trunc = self.compute_payoff_and_return_last_trunc()

        self.back_propagation_pricing(last_trunc)

        return self.root.option_price
    
    def save_tree_to_dataframe(self):
 
        max_columns = self.N + 1 
        max_rows = 2 * self.N + 1 
        extra_rows = int(math.ceil(self.N * 0.05))
        df1 = pd.DataFrame(index=range(max_rows + extra_rows), columns=range(max_columns))
        df2 = pd.DataFrame(index=range(max_rows + extra_rows), columns=range(max_columns))
        df3 = pd.DataFrame(index=range(max_rows + extra_rows), columns=range(max_columns))

        trunc = self.root

        for step in range(0, self.N + 1):
            
            column = step 
            current_row_df1 = self.N 
            current_row_df2 = current_row_df3 = step 
            current_node = trunc

            while current_node is not None:
                df1.iloc[current_row_df1, column] = current_node.value
                df2.iloc[current_row_df2, column] = current_node.value
                df3.iloc[current_row_df3, column] = current_node.value

                current_row_df1 += 1
                current_row_df2 += 1
                current_row_df3 -= 1

                current_node = current_node.down_neighbor

            current_row_df1 = self.N  
            current_row_df2 = current_row_df3 = step 
            current_node = trunc 
            
            while current_node is not None:
                df1.iloc[current_row_df1, column] = current_node.value
                df2.iloc[current_row_df2, column] = current_node.value
                df3.iloc[current_row_df3, column] = current_node.value

                current_row_df1 -= 1
                current_row_df2 -= 1
                current_row_df3 += 1
                
                current_node = current_node.up_neighbor

            trunc = trunc.forward_mid_neighbor
            
        self.number_of_nodes = df1.count().sum()

        return df1, df2, df3
    
    def visualize_tree(self, style='ko-'):

        # Calculate the number of steps in the tree
        N = self.N

        # Calculate cumulative time steps for the x-axis based on self.deltaT_array
        time_steps = np.cumsum(np.insert(self.deltaT_array, 0, 0))
        
        # Dynamically adjust line width and marker size based on the number of steps
        line_width = max(0.5, 4.0 / N)  # Decrease line width with more steps
        marker_size = max(2, 10.0 / (N / 5))  # Decrease marker size with more steps

        # Dynamically adjust figure size based on the number of steps
        plt.figure(figsize=(min(20, max(N * 0.4, 14)), min(10, max(N * 0.2, 7))))  # Width and height increase proportionally to N, but capped for readability

        df_mid, df_up, df_down = self.save_tree_to_dataframe()
        
        # Rename the columns for the mid DataFrame with time steps
        df_mid.columns = time_steps
        df_down.columns = time_steps
        df_up.columns = time_steps

        # Plot the tree using the style specified, with dynamic line width and marker size
        plt.plot(df_down.transpose(), style, linewidth=line_width, markersize=marker_size)  # Plot for one orientation
        plt.plot(df_up.transpose(), style, linewidth=line_width, markersize=marker_size)  # Plot for the reversed orientation
        plt.plot(df_mid.transpose(), style, linewidth=line_width, markersize=marker_size)  # Plot for the shifted orientation

        # Met des points bleue légèrement plus gros sur les mid de l'arbre(CaD la colonne avec la plus de valeurs non nulles)
        plt.plot(df_mid.T[df_mid.T.count().idxmax()],'o', linewidth=line_width, markersize=marker_size+3, color='blue')

        # Set plot labels
        plt.xlabel("Maturity (Cumulative)")
        plt.ylabel("Stock Price")

        return plt



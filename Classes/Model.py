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

    def calculate_option_price(self):
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

    def get_deltaT(self, factor=0):
        exp_factors = np.exp(np.linspace(0, factor, self.N))  # Exponential scaling factors
        self.deltaT_array = exp_factors / np.sum(exp_factors) * self.T  # Normalize to make sure the sum equals T

    def dividend_value(self, step):
        if self.dividend_step == step:
            return self.dividend
        return 0.0

    def compute_probability(self, node, step):
        deltaT = self.deltaT_array[step]
        alpha = math.exp(self.sigma * math.sqrt(3 * deltaT))

        esp = node.value * math.exp(self.rate * deltaT) - self.dividend_value(step)
        var = node.value ** (2) * math.exp(2 * self.rate * deltaT) * (math.exp(self.sigma ** 2 * deltaT) - 1)

        forward = node.value * math.exp(self.rate * deltaT) - self.dividend_value(step)
        p_d = (forward ** (-2) * (var + esp ** 2) - 1 - (alpha + 1) * (esp / forward - 1)) / ((1 - alpha) * (alpha ** (-2) - 1))
        p_u = (esp / forward - 1 - (1 / alpha - 1) * p_d) / (alpha - 1)
        p_m = 1 - p_u - p_d

        return p_d, p_u, p_m
    
    def compute_values(self, node, step):
        deltaT = self.deltaT_array[step]
        alpha = math.exp(self.sigma * math.sqrt(3 * deltaT))

        u_value = (node.value * math.exp(self.rate * deltaT) - self.dividend_value(step)) * alpha
        d_value = (node.value * math.exp(self.rate * deltaT) - self.dividend_value(step)) / alpha
        m_value = node.value * math.exp(self.rate * deltaT) - self.dividend_value(step)

        return d_value, u_value, m_value
    
    def build_tree(self, factor=0):

        self.get_deltaT(factor)
        self.dividend_step = math.ceil(self.ex_div_date * self.N)

        # Initialisation de l'arbre trinomial
        self.tree = [[None for _ in range(2 * N + 1)] for N in range(self.N + 1)]

        # Définir la racine de l'arbre (niveau 0)
        self.tree[0][0] = Node(self.S0, prob=1.0, cum_prob=1.0)

        # Construire l'arbre trinomial
        for step in range(self.N):  # Parcours de chaque niveau

            for i in range(2 * step + 1):  # Parcours de chaque nœud au niveau
                
                if self.tree[step][i] is not None:  # Si le nœud existe

                    current_node = self.tree[step][i]

                    # Calcul des probabilités pour chaque direction
                    p_d, p_u, p_m = self.compute_probability(current_node, step)

                    d_forward, u_forward, m_forward = self.compute_values(current_node, step)

                    # DOWN
                    if self.tree[step + 1][i] is None:
                        down_node = Node(d_forward, prob=p_d, cum_prob=p_d * current_node.cum_prob)
                        self.tree[step + 1][i] = down_node
                    else:
                        down_node = self.tree[step + 1][i]
                        down_node.cum_prob += p_d * current_node.cum_prob

                    # MID
                    if self.tree[step + 1][i + 1] is None:
                        mid_node = Node(m_forward, prob=p_m, cum_prob=p_m * current_node.cum_prob)                        
                        self.tree[step + 1][i + 1] = mid_node
                    else:
                        mid_node = self.tree[step + 1][i + 1]
                        mid_node.cum_prob += p_m * current_node.cum_prob
                        mid_node.value = m_forward
                
                    # UP
                    if self.tree[step + 1][i + 2] is None:
                        up_node = Node(u_forward, prob=p_u, cum_prob=p_u * current_node.cum_prob)
                        self.tree[step + 1][i + 2] = up_node
                    else:
                        up_node = self.tree[step + 1][i + 2]
                        up_node.cum_prob += p_u * current_node.cum_prob

                    # Lier les enfants au parent
                    current_node.children = [(down_node, p_d), (mid_node, p_m), (up_node, p_u)]

        # Calculer les valeurs terminales de l'option
        for i in range(2 * self.N + 1):
            self.tree[self.N][i].payoff(self.K, self.option_type)

    def calculate_option_price(self):
        """
        Calculate the option price using backward induction for both European and American styles.
        """
        # Calcul du facteur d'actualisation
        discount = np.exp(-self.rate * (self.T / self.N))
        
        # Parcours inverse de l'arbre (induction inverse)
        for j in range(self.N - 1, -1, -1):
            for i in range(2 * j + 1):
                current_node = self.tree[j][i]
            
                # Calculer le prix de l'option comme la somme pondérée des prix des enfants (prix théorique)
                option_value = discount * sum(child.option_price * prob for child, prob in current_node.children)

                if self.option_style == "american":
                    # On calcul le payoff à cette date, et on prend le max entre le payoff et le prix théorique de l'option
                    current_node.payoff(self.K, self.option_type)
                    current_node.option_price = max(option_value, current_node.option_price)
                else:  # Option européenne
                    # Utiliser uniquement le prix théorique de l'option pour les options européennes
                    current_node.option_price = option_value
        
        # Retourner le prix de l'option au nœud racine
        return self.tree[0][0].option_price

    def visualize_tree(self, style='ko-', threshold=0.0):
        """
        Plots the trinomial tree (lattice) for stock prices using matplotlib from a trinomial tree.
        Values with cumulative probabilities below a threshold are covered in white.

        Parameters
        ----------
        tree : list of lists
            A list containing lists of nodes, representing the trinomial tree steps.
            
        style : str
                Matplotlib style for plotting lines and markers. Default is 'ko-'.
        
        threshold : float
                    The threshold value for cumulative probability below which nodes are not plotted.
        """
        
        # Calculate the number of steps in the tree
        N = len(self.tree)

        # Calculate cumulative time steps for the x-axis based on self.deltaT_array
        time_steps = np.cumsum(np.insert(self.deltaT_array, 0, 0))
        
        # Dynamically adjust line width and marker size based on the number of steps
        line_width = max(0.5, 4.0 / N)  # Decrease line width with more steps
        marker_size = max(2, 10.0 / (N / 5))  # Decrease marker size with more steps

        # Dynamically adjust figure size based on the number of steps
        plt.figure(figsize=(min(20, max(N * 0.4, 14)), min(10, max(N * 0.2, 7))))  # Width and height increase proportionally to N, but capped for readability
        
        # Dictionary to hold values for plotting
        vals_down_node = {}
        vals_up_node = {}

        # Collect stock prices and cumulative probabilities from the trinomial tree nodes for each step
        for i, step in enumerate(self.tree):
            # Extract stock prices or values for the current step
            vals_down_node[i] = [[node.value, node.cum_prob] for node in step]  # Use 'node.value' or 'node.stock_price' based on your node definition
            vals_up_node[i] = list(reversed(vals_down_node[i]))  # For reverse plotting, we reverse the list
        
        # Filter values based on the threshold and convert collected data into Pandas DataFrames for easier plotting
        dict_down = {k: pd.Series([np.nan if sublist[1] < threshold else sublist[0] for sublist in v]) for k, v in vals_down_node.items()}
        dict_up = {k: pd.Series([np.nan if sublist[1] < threshold else sublist[0] for sublist in v]) for k, v in vals_up_node.items()}

        # Create DataFrames for plotting
        df_down = pd.DataFrame(dict_down)
        df_up = pd.DataFrame(dict_up)
        df_mid = df_up.copy()
        for i in range(len(df_mid.columns)-1, -1, -1):
            df_mid[i] = df_mid[i].shift(-(i - len(df_mid.columns) + 1))
        
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

        # Display the plot
        return plt
    
    def get_number_of_nodes(self):
        """
        Returns the total number of nodes in the trinomial tree.
        """
        return sum(1 for step in self.tree for node in step if node is not None)
    
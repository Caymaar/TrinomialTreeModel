from Classes.BlackScholes import BlackScholes
import plotly.graph_objects as go

class OneDimensionalDerivative:
    def __init__(self, function, shift = 1e-4):
        self.function = function
        self.shift = shift

    def first_derivative(self, x):
        return (self.function(x + self.shift) - self.function(x - self.shift)) / (2 * self.shift)

    def second_derivative(self, x):
        return (self.function(x + self.shift) - 2 * self.function(x) + self.function(x - self.shift)) / (self.shift ** 2)

class Greeks:
    def __init__(self, tree_model, shift = 1e-4):
        self.tree_model = tree_model
        self.bs_model = BlackScholes(tree_model.market, tree_model.option)
        self.shift = shift

    def option_price_given_S0(self, S0):
        original_S0 = self.tree_model.market.S0
        self.tree_model.market.S0 = S0
        price = self.tree_model.get_option_price(light_mode=True)
        self.tree_model.market.S0 = original_S0
        return price

    def calculate_delta(self):
        derivative = OneDimensionalDerivative(lambda S0: self.option_price_given_S0(S0), self.shift)
        delta = derivative.first_derivative(self.tree_model.market.S0)
        return delta

    def calculate_gamma(self):
        original_shift = self.shift
        self.shift = 3
        derivative = OneDimensionalDerivative(lambda S0: self.option_price_given_S0(S0), self.shift)
        gamma = derivative.second_derivative(self.tree_model.market.S0)
        self.shift = original_shift
        return gamma

    def option_price_given_T(self, T):
        original_T = self.tree_model.option.T
        self.tree_model.option.T = T
        price = self.tree_model.get_option_price(light_mode=True)
        self.tree_model.option.T = original_T
        return price

    def calculate_theta(self):
        derivative = OneDimensionalDerivative(lambda T: self.option_price_given_T(T), self.shift)
        theta = - derivative.first_derivative(self.tree_model.option.T)
        return theta

    def option_price_given_sigma(self, sigma):
        original_sigma = self.tree_model.market.sigma
        self.tree_model.market.sigma = sigma
        price = self.tree_model.get_option_price(light_mode=True)
        self.tree_model.market.sigma = original_sigma
        return price

    def calculate_vega(self):
        derivative = OneDimensionalDerivative(lambda sigma: self.option_price_given_sigma(sigma), self.shift)
        vega = derivative.first_derivative(self.tree_model.market.sigma)
        return vega

    def option_price_given_rate(self, rate):
        original_rate = self.tree_model.market.rate
        self.tree_model.market.rate = rate
        price = self.tree_model.get_option_price(light_mode=True)
        self.tree_model.market.rate = original_rate
        return price

    def calculate_rho(self):
        derivative = OneDimensionalDerivative(lambda rate: self.option_price_given_rate(rate), self.shift)
        rho = derivative.first_derivative(self.tree_model.market.rate)
        return rho

    def get_greeks(self):
        delta = self.calculate_delta()
        gamma = self.calculate_gamma()
        theta = self.calculate_theta()
        vega = self.calculate_vega()
        rho = self.calculate_rho()
        return delta, gamma, theta, vega, rho
    
    def get_greeks_bs(self):
        delta = self.bs_model.delta()
        gamma = self.bs_model.gamma()
        theta = self.bs_model.theta()
        vega = self.bs_model.vega()
        rho = self.bs_model.rho()
        return delta, gamma, theta, vega, rho
    
    def compute_and_show_greeks(self, S0_values, show=True):
        deltas = []
        gammas = []
        thetas = []
        vegas = []
        rhos = []

        bs_deltas = []
        bs_gammas = []
        bs_thetas = []
        bs_vegas = []
        bs_rhos = []

        # Stocker les valeurs originales pour les restaurer après le calcul
        original_S0 = self.tree_model.market.S0
        original_bs_S0 = self.bs_model.market.S0

        for S0 in S0_values:
            # Mettre à jour S0 pour le modèle de l'arbre trinomial
            self.tree_model.market.S0 = S0
            delta, gamma, theta, vega, rho = self.get_greeks()
            deltas.append(delta)
            gammas.append(gamma)
            thetas.append(theta)
            vegas.append(vega)
            rhos.append(rho)

            # Mettre à jour S0 pour le modèle de Black-Scholes
            self.bs_model.market.S0 = S0
            bs_delta, bs_gamma, bs_theta, bs_vega, bs_rho = self.get_greeks_bs()
            bs_deltas.append(bs_delta)
            bs_gammas.append(bs_gamma)
            bs_thetas.append(bs_theta)
            bs_vegas.append(bs_vega)
            bs_rhos.append(bs_rho)

        # Restaurer les valeurs originales de S0
        self.tree_model.market.S0 = original_S0
        self.bs_model.market.S0 = original_bs_S0

        # Figure 1: Delta
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=list(S0_values), y=deltas, mode='lines+markers', name='Delta - Tree'))
        fig1.add_trace(go.Scatter(x=list(S0_values), y=bs_deltas, mode='lines+markers', name='Delta - Black-Scholes'))
        fig1.update_layout(
            title="Delta as a Function of the Underlying Asset Price",
            xaxis_title="Underlying Asset Price",
            yaxis_title="Delta",
            template="plotly_white"
        )

        # Figure 2: Gamma
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=list(S0_values), y=gammas, mode='lines+markers', name='Gamma - Tree'))
        fig2.add_trace(go.Scatter(x=list(S0_values), y=bs_gammas, mode='lines+markers', name='Gamma - Black-Scholes'))
        fig2.update_layout(
            title="Gamma as a Function of the Underlying Asset Price",
            xaxis_title="Underlying Asset Price",
            yaxis_title="Gamma",
            template="plotly_white"
        )

        # Figure 3: Theta
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=list(S0_values), y=thetas, mode='lines+markers', name='Theta - Tree'))
        fig3.add_trace(go.Scatter(x=list(S0_values), y=bs_thetas, mode='lines+markers', name='Theta - Black-Scholes'))
        fig3.update_layout(
            title="Theta as a Function of the Underlying Asset Price",
            xaxis_title="Underlying Asset Price",
            yaxis_title="Theta",
            template="plotly_white"
        )

        # Figure 4: Vega
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=list(S0_values), y=vegas, mode='lines+markers', name='Vega - Tree'))
        fig4.add_trace(go.Scatter(x=list(S0_values), y=bs_vegas, mode='lines+markers', name='Vega - Black-Scholes'))
        fig4.update_layout(
            title="Vega as a Function of the Underlying Asset Price",
            xaxis_title="Underlying Asset Price",
            yaxis_title="Vega",
            template="plotly_white"
        )

        # Figure 5: Rho
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=list(S0_values), y=rhos, mode='lines+markers', name='Rho - Tree'))
        fig5.add_trace(go.Scatter(x=list(S0_values), y=bs_rhos, mode='lines+markers', name='Rho - Black-Scholes'))
        fig5.update_layout(
            title="Rho as a Function of the Underlying Asset Price",
            xaxis_title="Underlying Asset Price",
            yaxis_title="Rho",
            template="plotly_white"
        )

        if show:
            fig1.show()
            fig2.show()
            fig3.show()
            fig4.show()
            fig5.show()
        else:
            return fig1, fig2, fig3, fig4, fig5
from Classes.BlackScholes import BlackScholes
import plotly.graph_objects as go

class OneDimensionalDerivative:
    """
    Classe pour calculer les dérivées d'une fonction à une dimension.
    
    Attributes:
        function: La fonction dont on souhaite calculer les dérivées.
        shift: Le petit incrément utilisé pour le calcul des dérivées.
    """

    def __init__(self, function, shift=1e-4):
        """
        Initialise la classe avec la fonction et le décalage.
        
        Args:
            function (callable): La fonction dont on souhaite calculer les dérivées.
            shift (float): Le décalage utilisé pour les calculs de dérivée.
        """
        self.function = function
        self.shift = shift

    def first_derivative(self, x):
        """
        Calcule la première dérivée de la fonction en un point donné.

        Args:
            x (float): Le point auquel la dérivée est calculée.

        Returns:
            float: La valeur de la première dérivée.
        """
        # Calcul des valeurs de la fonction pour x + shift et x - shift
        test1 = self.function(x + self.shift)
        test2 = self.function(x - self.shift)
        
        # Calcul de la dérivée
        return (test1 - test2) / (2 * self.shift)

    def second_derivative(self, x):
        """
        Calcule la seconde dérivée de la fonction en un point donné.

        Args:
            x (float): Le point auquel la seconde dérivée est calculée.

        Returns:
            float: La valeur de la seconde dérivée.
        """
        return (self.function(x + self.shift) - 2 * self.function(x) + self.function(x - self.shift)) / (self.shift ** 2)

class Greeks:
    """
    Classe pour calculer les 'Greeks' d'une option à l'aide d'un modèle d'arbre et du modèle Black-Scholes.

    Attributes:
        tree_model: Un modèle d'arbre trinomial pour le pricing d'options.
        bs_model: Un modèle Black-Scholes pour le pricing d'options.
        shift: Le petit incrément utilisé pour le calcul des dérivées.
    """

    def __init__(self, tree_model, shift=1e-4):
        """
        Initialise la classe avec le modèle d'arbre et le décalage.

        Args:
            tree_model (Tree): Le modèle d'arbre pour le pricing des options.
            shift (float): Le décalage utilisé pour les calculs de dérivées.
        """
        self.tree_model = tree_model
        self.bs_model = BlackScholes(tree_model.market, tree_model.option)
        self.shift = shift

    def option_price_given_S0(self, S0):
        """
        Calcule le prix de l'option en fonction du prix de l'actif sous-jacent S0.

        Args:
            S0 (float): Prix de l'actif sous-jacent.

        Returns:
            float: Prix de l'option correspondant au S0 donné.
        """
        original_S0 = self.tree_model.market.S0
        self.tree_model.market.S0 = S0
        price = self.tree_model.get_option_price(light_mode=False, factor=0, threshold=1e-10)
        self.tree_model.market.S0 = original_S0
        return price

    def calculate_delta(self):
        """
        Calcule la sensibilité (delta) de l'option par rapport au prix de l'actif sous-jacent.

        Returns:
            float: Valeur de delta.
        """
        derivative = OneDimensionalDerivative(lambda S0: self.option_price_given_S0(S0), self.shift)
        return derivative.first_derivative(self.tree_model.market.S0)

    def calculate_gamma(self):
        """
        Calcule la convexité (gamma) de l'option.

        Returns:
            float: Valeur de gamma.
        """
        original_shift = self.shift
        self.shift = 3  # Changement de décalage pour le calcul de gamma
        derivative = OneDimensionalDerivative(lambda S0: self.option_price_given_S0(S0), self.shift)
        gamma = derivative.second_derivative(self.tree_model.market.S0)
        self.shift = original_shift  # Restauration du décalage original
        return gamma

    def option_price_given_T(self, T):
        """
        Calcule le prix de l'option en fonction de la maturité T.

        Args:
            T (float): Maturité de l'option.

        Returns:
            float: Prix de l'option correspondant à T donné.
        """
        original_T = self.tree_model.option.T
        self.tree_model.option.T = T
        price = self.tree_model.get_option_price(light_mode=False, factor=0, threshold=1e-10)
        self.tree_model.option.T = original_T
        return price

    def calculate_theta(self):
        """
        Calcule la sensibilité (theta) de l'option par rapport à la maturité.

        Returns:
            float: Valeur de theta, ajustée pour une période de 365 jours.
        """
        derivative = OneDimensionalDerivative(lambda T: self.option_price_given_T(T), self.shift)
        theta = -derivative.first_derivative(self.tree_model.option.T)
        return theta / 365

    def option_price_given_sigma(self, sigma):
        """
        Calcule le prix de l'option en fonction de la volatilité sigma.

        Args:
            sigma (float): Volatilité de l'actif sous-jacent.

        Returns:
            float: Prix de l'option correspondant à sigma donné.
        """
        original_sigma = self.tree_model.market.sigma
        self.tree_model.market.sigma = sigma
        price = self.tree_model.get_option_price(light_mode=False, factor=0, threshold=1e-10)
        self.tree_model.market.sigma = original_sigma
        return price

    def calculate_vega(self):
        """
        Calcule la sensibilité (vega) de l'option par rapport à la volatilité.

        Returns:
            float: Valeur de vega, ajustée par un facteur de 100.
        """
        derivative = OneDimensionalDerivative(lambda sigma: self.option_price_given_sigma(sigma), self.shift)
        vega = derivative.first_derivative(self.tree_model.market.sigma)
        return vega / 100

    def option_price_given_rate(self, rate):
        """
        Calcule le prix de l'option en fonction du taux d'intérêt.

        Args:
            rate (float): Taux d'intérêt.

        Returns:
            float: Prix de l'option correspondant au taux donné.
        """
        original_rate = self.tree_model.market.rate
        self.tree_model.market.rate = rate
        price = self.tree_model.get_option_price(light_mode=False, factor=0, threshold=1e-10)
        self.tree_model.market.rate = original_rate
        return price

    def calculate_rho(self):
        """
        Calcule la sensibilité (rho) de l'option par rapport au taux d'intérêt.

        Returns:
            float: Valeur de rho, ajustée par un facteur de 100.
        """
        derivative = OneDimensionalDerivative(lambda rate: self.option_price_given_rate(rate), self.shift)
        rho = derivative.first_derivative(self.tree_model.market.rate)
        return rho / 100

    def get_greeks(self):
        """
        Calcule tous les 'Greeks' (delta, gamma, theta, vega, rho) pour l'option à partir du modèle d'arbre.

        Returns:
            tuple: Valeurs de delta, gamma, theta, vega et rho.
        """
        delta = self.calculate_delta()
        gamma = self.calculate_gamma()
        theta = self.calculate_theta()
        vega = self.calculate_vega()
        rho = self.calculate_rho()
        return delta, gamma, theta, vega, rho
    
    def get_greeks_bs(self):
        """
        Calcule tous les 'Greeks' pour l'option à partir du modèle Black-Scholes.

        Returns:
            tuple: Valeurs de delta, gamma, theta, vega et rho.
        """
        delta = self.bs_model.delta()
        gamma = self.bs_model.gamma()
        theta = self.bs_model.theta()
        vega = self.bs_model.vega()
        rho = self.bs_model.rho()
        return delta, gamma, theta, vega, rho
    
    def compute_and_show_greeks(self, S0_values, show=True):
        """
        Calcule et affiche les 'Greeks' pour une série de valeurs de S0.
        
        Args:
            S0_values (list): Liste des valeurs de l'actif sous-jacent.
            show (bool): Indique si les graphiques doivent être affichés.
        
        Returns:
            tuple: Figures contenant les graphiques pour chaque Greek si show est False.
        """
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

        # Création des graphiques pour chaque Greek
        figures = []
        
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
        figures.append(fig1)

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
        figures.append(fig2)

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
        figures.append(fig3)

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
        figures.append(fig4)

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
        figures.append(fig5)

        if show:
            for fig in figures:
                fig.show()
        else:
            return figures
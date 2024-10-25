import numpy as np
from scipy.stats import norm

class BlackScholes:
    """
    Classe pour le modèle de Black-Scholes.

    Attributes:
        market (Market): Instance du marché associé à l'option.
        option (Option): Instance de l'option à évaluer.
    """

    def __init__(self, market, option):
        """
        Initialise le modèle de Black-Scholes avec un marché et une option.

        Args:
            market (Market): Instance du marché.
            option (Option): Instance de l'option.
        """
        self.market = market
        self.option = option

    def d1(self):
        """
        Calcule le terme d1 utilisé dans les formules Black-Scholes.

        Returns:
            float: Valeur du terme d1.
        """
        return (np.log(self.market.S0 / self.option.K) + (self.market.rate + 0.5 * self.market.sigma ** 2) * self.option.T) / (self.market.sigma * np.sqrt(self.option.T))
    
    def d2(self):
        """
        Calcule le terme d2 utilisé dans les formules Black-Scholes.

        Returns:
            float: Valeur du terme d2.
        """
        return self.d1() - self.market.sigma * np.sqrt(self.option.T)
    
    def option_price(self):
        """
        Calcule le prix de l'option selon le modèle de Black-Scholes.

        Returns:
            float: Prix de l'option.
        """
        if self.option.type == "call":
            return self.market.S0 * norm.cdf(self.d1()) - self.option.K * np.exp(-self.market.rate * self.option.T) * norm.cdf(self.d2())
        else:  # Option put
            return self.option.K * np.exp(-self.market.rate * self.option.T) * norm.cdf(-self.d2()) - self.market.S0 * norm.cdf(-self.d1())

    def delta(self):
        """
        Calcule le delta de l'option.

        Returns:
            float: Delta de l'option.
        """
        if self.option.type == "call":
            return norm.cdf(self.d1())
        else:  # Option put
            return norm.cdf(self.d1()) - 1

    def gamma(self):
        """
        Calcule le gamma de l'option.

        Returns:
            float: Gamma de l'option.
        """
        return norm.pdf(self.d1()) / (self.market.S0 * self.market.sigma * np.sqrt(self.option.T))

    def theta(self):
        """
        Calcule le theta de l'option.

        Returns:
            float: Theta de l'option, en termes de variation de prix par jour.
        """
        term1 = - (self.market.S0 * norm.pdf(self.d1()) * self.market.sigma) / (2 * np.sqrt(self.option.T))
        if self.option.type == "call":
            term2 = self.market.rate * self.option.K * np.exp(-self.market.rate * self.option.T) * norm.cdf(self.d2())
            return (term1 - term2) / 365
        else:  # Option put
            term2 = self.market.rate * self.option.K * np.exp(-self.market.rate * self.option.T) * norm.cdf(-self.d2())
            return (term1 + term2) / 365

    def vega(self):
        """
        Calcule le vega de l'option.

        Returns:
            float: Vega de l'option.
        """
        return (self.market.S0 * norm.pdf(self.d1()) * np.sqrt(self.option.T)) / 100

    def rho(self):
        """
        Calcule le rho de l'option.

        Returns:
            float: Rho de l'option.
        """
        if self.option.type == "call":
            return (self.option.K * self.option.T * np.exp(-self.market.rate * self.option.T) * norm.cdf(self.d2())) / 100
        else:  # Option put
            return (-self.option.K * self.option.T * np.exp(-self.market.rate * self.option.T) * norm.cdf(-self.d2())) / 100
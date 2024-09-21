import numpy as np
from scipy import stats as sps

class BlackScholes:

    def __init__(self, market, option):
        self.market = market
        self.option = option

    def option_price(self):

        # Calcul des d1 et d2 avec le sous-jacent ajusté pour les dividendes
        d1 = (np.log(self.market.S0 / self.option.K) + (self.market.rate + 0.5 * self.market.sigma ** 2) * self.option.T) / (self.market.sigma * np.sqrt(self.option.T))
        d2 = d1 - self.market.sigma * np.sqrt(self.option.T)

        # Calcul du prix de l'option
        if self.option.type == "call":
            option_price = self.market.S0 * sps.norm.cdf(d1) - self.option.K * np.exp(-self.market.rate * self.option.T) * sps.norm.cdf(d2)
        else:  # Option put
            option_price = self.option.K * np.exp(-self.market.rate * self.option.T) * sps.norm.cdf(-d2) - self.market.S0 * sps.norm.cdf(-d1)

        return option_price
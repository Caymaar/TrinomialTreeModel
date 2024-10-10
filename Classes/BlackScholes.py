import numpy as np
from scipy.stats import norm

class BlackScholes:

    def __init__(self, market, option):
        self.market = market
        self.option = option

    def d1(self):
        return (np.log(self.market.S0 / self.option.K) + (self.market.rate + 0.5 * self.market.sigma ** 2) * self.option.T) / (self.market.sigma * np.sqrt(self.option.T))
    
    def d2(self):
        return self.d1() - self.market.sigma * np.sqrt(self.option.T)
    
    def option_price(self):
        if self.option.type == "call":
            return self.market.S0 * norm.cdf(self.d1()) - self.option.K * np.exp(-self.market.rate * self.option.T) * norm.cdf(self.d2())
        else:  # Option put
            return self.option.K * np.exp(-self.market.rate * self.option.T) * norm.cdf(-self.d2()) - self.market.S0 * norm.cdf(-self.d1())

    def delta(self):
        if self.option.type == "call":
            return norm.cdf(self.d1())
        else:
            return norm.cdf(self.d1()) - 1

    def gamma(self):
        return norm.pdf(self.d1()) / (self.market.S0 * self.market.sigma * np.sqrt(self.option.T))

    def theta(self):
        term1 = - (self.market.S0 * norm.pdf(self.d1()) * self.market.sigma) / (2 * np.sqrt(self.option.T))
        if self.option.type == "call":
            term2 = self.market.rate * self.option.K * np.exp(-self.market.rate * self.option.T) * norm.cdf(self.d2())
            return term1 - term2
        else:
            term2 = self.market.rate * self.option.K * np.exp(-self.market.rate * self.option.T) * norm.cdf(-self.d2())
            return term1 + term2

    def vega(self):
        return self.market.S0 * norm.pdf(self.d1()) * np.sqrt(self.option.T)

    def rho(self):
        if self.option.type == "call":
            return self.option.K * self.option.T * np.exp(-self.market.rate * self.option.T) * norm.cdf(self.d2())
        else:
            return -self.option.K * self.option.T * np.exp(-self.market.rate * self.option.T) * norm.cdf(-self.d2())

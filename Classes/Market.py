class Market:
    """
    Classe représentant un marché financier.

    Attributes:
        S0 (float): Le prix initial du sous-jacent.
        rate (float): Le taux d'intérêt sans risque.
        sigma (float): La volatilité du sous-jacent.
        dividend (float): Le dividende du sous-jacent.
        ex_div_date (datetime, optional): La date d'ex-dividende.
    """

    def __init__(self, S0, rate, sigma, dividend=0.0, ex_div_date=None):
        """
        Initialise le marché avec ses paramètres.

        Args:
            S0 (float): Le prix initial du sous-jacent.
            rate (float): Le taux d'intérêt sans risque.
            sigma (float): La volatilité du sous-jacent.
            dividend (float, optional): Le dividende du sous-jacent.
            ex_div_date (datetime, optional): La date d'ex-dividende.
        """
        self.S0 = S0
        self.rate = rate
        self.sigma = sigma
        self.dividend = dividend
        self.ex_div_date = ex_div_date
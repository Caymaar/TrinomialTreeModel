class Market:
    def __init__(self, S0, rate, sigma, dividend=0.0, ex_div_date=None):
        self.S0 = S0
        self.rate = rate
        self.sigma = sigma
        self.dividend = dividend
        self.ex_div_date = ex_div_date
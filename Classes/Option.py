class Option:
    def __init__(self, K, T, type, style):
        self.K = K
        self.T = T
        self.type = type
        self.style = style

    def payoff(self, value):
        if self.type == "call":
            return max(value - self.K, 0)
        else:
            return max(self.K - value, 0)
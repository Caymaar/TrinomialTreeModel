class Node:
    def __init__(self, value, prob=0.0, cum_prob=0.0):
        self.value = value  # Valeur de l'actif sous-jacent
        self.prob = prob  # Probabilité de ce mouvement à partir du parent
        self.cum_prob = cum_prob  # Probabilité cumulée d'atteindre ce nœud
        self.option_price = 0.0  # Prix de l'option au nœud
        self.children = []  # Liste des enfants du nœud
        self.parent = None  # Parent du nœud
    
    def payoff(self, K, option_type):
        if option_type == "call":
            self.option_price = max(self.value - K, 0)
        else:
            self.option_price = max(K - self.value, 0)

    #def calculate_prob(self, parent_prob):

    def __repr__(self):
        return f"Node(value={self.value:.2f}, option_price={self.option_price:.5f}, prob={self.prob:.5f}, cum_prob={self.cum_prob:.5f})"

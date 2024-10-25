class Option:
    """
    Classe représentant une option financière.

    Attributes:
        K (float): Le prix d'exercice de l'option.
        T (float): Le temps jusqu'à l'échéance en années.
        type (str): Type d'option, soit "call" soit "put".
        style (str): Style d'option (non utilisé dans les méthodes actuelles).
    """

    def __init__(self, K, T, type, style):
        """
        Initialise une option avec ses paramètres.

        Args:
            K (float): Le prix d'exercice de l'option.
            T (float): Le temps jusqu'à l'échéance.
            type (str): Type d'option ("call" ou "put").
            style (str): Style d'option (non utilisé dans les méthodes actuelles).
        """
        self.K = K
        self.T = T
        self.type = type
        self.style = style

    def payoff(self, value):
        """
        Calcule le payoff de l'option à l'échéance.

        Args:
            value (float): La valeur du sous-jacent à l'échéance.

        Returns:
            float: Le payoff de l'option.
        """
        if self.type == "call":
            return max(value - self.K, 0)
        else:  # Put option
            return max(self.K - value, 0)
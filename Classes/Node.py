import math

class Node:
    """
    Classe représentant un nœud dans un arbre trinomial pour le pricing des options.
    
    Attributes:
        value: La valeur du nœud.
        step: L'étape (ou niveau) dans l'arbre.
        tree: Une référence à l'arbre dans lequel le nœud se trouve.
        option_price: Le prix de l'option associé à ce nœud.
        cum_prob: Probabilité cumulée.
        up_neighbor, down_neighbor, backward_neighbor: Références aux nœuds voisins dans l'arbre.
        forward_up_neighbor, forward_mid_neighbor, forward_down_neighbor: Références aux voisins en avant.
        prob_forward_up_neighbor, prob_forward_mid_neighbor, prob_forward_down_neighbor: Probabilités associées aux voisins en avant.
    """

    def __init__(self, value, step, tree):
        """
        Initialise un nœud avec une valeur, un pas et une référence à l'arbre.
        
        Args:
            value (float): Valeur du nœud.
            step (int): Étape du nœud dans l'arbre.
            tree (Tree): Référence à l'arbre dans lequel se trouve le nœud.
        """
        self.value = value
        self.step = step

        self.tree = tree
        self.option_price = 0.0  # Prix de l'option, initialisé à zéro.

        self.cum_prob = 0.0  # Initialisation de la probabilité cumulée.

        # Initialisation des voisins à None.
        self.up_neighbor = None
        self.down_neighbor = None
        self.backward_neighbor = None

        self.forward_up_neighbor = None
        self.forward_mid_neighbor = None
        self.forward_down_neighbor = None

        self.prob_forward_up_neighbor = None
        self.prob_forward_mid_neighbor = None
        self.prob_forward_down_neighbor = None

    ########################### Générations ###########################

    def create_forward_neighbors(self):
        """
        Crée les nœuds voisins en avant (up, mid, down) à partir de ce nœud.
        """
        # Récupération de la valeur de deltaT et alpha pour ce step
        deltaT, alpha = self.get_deltaT_alpha()

        # Création du Forward Mid
        self.forward_mid_neighbor = Node(
            self.value * math.exp(self.tree.market.rate * deltaT) - self.tree.dividend_value(self.step + 1),
            self.step + 1, self.tree
        )
        self.forward_mid_neighbor.backward_neighbor = self

        # Création du Forward Up
        self.forward_up_neighbor = Node(self.forward_mid_neighbor.value * alpha, self.step + 1, self.tree)
        self.forward_mid_neighbor.up_neighbor = self.forward_up_neighbor
        self.forward_up_neighbor.down_neighbor = self.forward_mid_neighbor
        
        # Création du Forward Down
        self.forward_down_neighbor = Node(self.forward_mid_neighbor.value / alpha, self.step + 1, self.tree)
        self.forward_mid_neighbor.down_neighbor = self.forward_down_neighbor
        self.forward_down_neighbor.up_neighbor = self.forward_mid_neighbor

        # Calcul des probabilités
        self.compute_probabilities()

    def generate_upper_neighbors(self):
        """
        Génère les voisins supérieurs du nœud actuel.
        """
        deltaT, alpha = self.get_deltaT_alpha()

        # Stockage du tronc pour restaurer la référence à ce nœud
        trunc = self

        # Initialisation de la variable Forward pour naviguer vers le haut
        actual_forward_up_node = self.forward_up_neighbor

        while self.up_neighbor is not None:
            # Création du voisin supérieur
            actual_forward_up_node.up_neighbor = Node(actual_forward_up_node.value * alpha, self.step + 1, self.tree)
            actual_forward_up_node.up_neighbor.down_neighbor = actual_forward_up_node
            
            supposed_mid_for_up_node_value = self.up_neighbor.value * math.exp(self.tree.market.rate * deltaT) - self.tree.dividend_value(self.step + 1)

            upper_bound = actual_forward_up_node.value * (1 + alpha) / 2
            lower_bound = actual_forward_up_node.value * (1 + 1 / alpha) / 2

            # Vérification si la valeur supposée est dans les limites
            if supposed_mid_for_up_node_value < upper_bound and supposed_mid_for_up_node_value > lower_bound:
                self.up_neighbor.forward_up_neighbor = actual_forward_up_node.up_neighbor
                self.up_neighbor.forward_mid_neighbor = actual_forward_up_node
                self.up_neighbor.forward_down_neighbor = actual_forward_up_node.down_neighbor
                
                self.up_neighbor.compute_probabilities()

                # Monomial si la probabilité cumulée est en dessous du seuil et s'il n'y a pas de voisin supérieur
                if self.up_neighbor.forward_up_neighbor.cum_prob < self.tree.threshold and self.up_neighbor.up_neighbor is None:
                    self.up_neighbor.monomial()
                    actual_forward_up_node.up_neighbor = None

                self = self.up_neighbor

            actual_forward_up_node = actual_forward_up_node.up_neighbor

        # Restauration du tronc
        self = trunc

    def generate_lower_neighbors(self):
        """
        Génère les voisins inférieurs du nœud actuel.
        """
        deltaT, alpha = self.get_deltaT_alpha()

        trunc = self

        actual_forward_down_node = self.forward_down_neighbor

        while self.down_neighbor is not None:
            actual_forward_down_node.down_neighbor = Node(actual_forward_down_node.value / alpha, self.step + 1, self.tree)
            actual_forward_down_node.down_neighbor.up_neighbor = actual_forward_down_node
            
            supposed_mid_for_down_node_value = self.down_neighbor.value * math.exp(self.tree.market.rate * deltaT) - self.tree.dividend_value(self.step + 1)

            upper_bound = actual_forward_down_node.value * (1 + alpha) / 2
            lower_bound = actual_forward_down_node.value * (1 + 1 / alpha) / 2

            # Vérification des conditions pour créer le voisin inférieur
            if (supposed_mid_for_down_node_value <= upper_bound and supposed_mid_for_down_node_value >= lower_bound) or supposed_mid_for_down_node_value <= 0:
                self.down_neighbor.forward_up_neighbor = actual_forward_down_node.up_neighbor
                self.down_neighbor.forward_mid_neighbor = actual_forward_down_node
                self.down_neighbor.forward_down_neighbor = actual_forward_down_node.down_neighbor
                
                self.down_neighbor.compute_probabilities()
                
                # Monomial si la probabilité cumulée est en dessous du seuil et s'il n'y a pas de voisin inférieur
                if self.down_neighbor.forward_down_neighbor.cum_prob < self.tree.threshold and self.down_neighbor.down_neighbor is None:
                    self.down_neighbor.monomial()
                    actual_forward_down_node.down_neighbor = None

                self = self.down_neighbor

            actual_forward_down_node = actual_forward_down_node.down_neighbor

        self = trunc
        
    def monomial(self):
        """
        Transforme le nœud en un monomial en supprimant ses voisins en avant et en fixant la probabilité à 1.
        """
        self.forward_down_neighbor = None
        self.forward_up_neighbor = None
        self.prob_forward_mid_neighbor = 1.0

    ########################### Calculs ###########################

    def compute_option_price(self):
        """
        Calcule le prix de l'option pour ce nœud en tenant compte des prix de ses voisins.
        """
        discount = math.exp(-self.tree.market.rate * self.tree.deltaT_array[self.step])
        option_value = 0.0

        # Somme des valeurs d'option des voisins pondérées par leurs probabilités
        if self.forward_up_neighbor is not None:
            option_value += self.forward_up_neighbor.option_price * self.prob_forward_up_neighbor

        if self.forward_mid_neighbor is not None:
            option_value += self.forward_mid_neighbor.option_price * self.prob_forward_mid_neighbor

        if self.forward_down_neighbor is not None:
            option_value += self.forward_down_neighbor.option_price * self.prob_forward_down_neighbor

        option_value *= discount

        # Pour les options de style américain, on compare avec le payoff
        if self.tree.option.style == "american":
            self.option_price = max(option_value, self.tree.option.payoff(self.value))
        else:
            self.option_price = option_value

    def compute_probabilities(self):
        """
        Calcule les probabilités pour les voisins en avant en fonction de la valeur du nœud.
        """
        deltaT, alpha = self.get_deltaT_alpha()

        esp = self.value * math.exp(self.tree.market.rate * deltaT) - self.tree.dividend_value(self.step + 1)
        var = self.value ** (2) * math.exp(2 * self.tree.market.rate * deltaT) * (math.exp(self.tree.market.sigma ** 2 * deltaT) - 1)

        # Calcul des probabilités pour les voisins
        down_prob = (self.forward_mid_neighbor.value ** (-2) * (var + esp ** 2) - 1 - (alpha + 1) * (esp / self.forward_mid_neighbor.value - 1)) / ((1 - alpha) * (alpha ** (-2) - 1))
        up_prob = (esp / self.forward_mid_neighbor.value - 1 - (1 / alpha - 1) * down_prob) / (alpha - 1)
        mid_prob = 1 - up_prob - down_prob

        # Attribution des probabilités calculées
        self.prob_forward_down_neighbor = down_prob
        self.prob_forward_up_neighbor = up_prob
        self.prob_forward_mid_neighbor = mid_prob

        # Mise à jour des probabilités cumulées pour les voisins
        if self.forward_up_neighbor is not None:
            self.forward_up_neighbor.cum_prob += self.cum_prob * self.prob_forward_up_neighbor

        if self.forward_mid_neighbor is not None:
            self.forward_mid_neighbor.cum_prob += self.cum_prob * self.prob_forward_mid_neighbor

        if self.forward_down_neighbor is not None:
            self.forward_down_neighbor.cum_prob += self.cum_prob * self.prob_forward_down_neighbor

    def get_deltaT_alpha(self):
        """
        Récupère deltaT et alpha pour le nœud actuel.
        
        Raises:
            IndexError: Si l'étape actuelle dépasse la taille du tableau deltaT.
        
        Returns:
            tuple: (deltaT, alpha)
        """
        if self.step >= len(self.tree.deltaT_array):
            raise IndexError(f"Index {self.step} is out of bounds for deltaT_array with size {len(self.tree.deltaT_array)}")
        
        deltaT = self.tree.deltaT_array[self.step]
        alpha = math.exp(self.tree.market.sigma * math.sqrt(3 * deltaT))
        return deltaT, alpha

    ########################### Dunders ###########################

    def __repr__(self):
        """
        Représentation du nœud sous forme de chaîne pour le débogage.
        
        Returns:
            str: Représentation du nœud.
        """
        prob_up = f"{self.prob_forward_up_neighbor:.5f}" if self.prob_forward_up_neighbor is not None else ""
        prob_mid = f"{self.prob_forward_mid_neighbor:.5f}" if self.prob_forward_mid_neighbor is not None else ""
        prob_down = f"{self.prob_forward_down_neighbor:.5f}" if self.prob_forward_down_neighbor is not None else ""
        
        return f"v={self.value:.5f}, step={self.step}, op={self.option_price:.5f}, up={prob_up}, mid={prob_mid}, down={prob_down}"
    
    def __hash__(self):
        """
        Fonction de hachage pour le nœud. Utile pour la création du graphique.
        
        Returns:
            int: Valeur de hachage du nœud.
        """
        return hash((self.value, self.step))

    def __eq__(self, other):
        """
        Vérifie l'égalité entre deux nœuds. Utile pour la création du graphique.
        
        Args:
            other (Node): Autre nœud à comparer.
        
        Returns:
            bool: True si les nœuds sont égaux, sinon False.
        """
        return isinstance(other, Node) and self.value == other.value and self.step == other.step

    def __str__(self):
        """
        Représentation du nœud sous forme de chaîne.
        
        Returns:
            str: Représentation lisible du nœud.
        """
        return f"v={self.value:.5f}, step={self.step}, op={self.option_price:.5f}"
    
    ############################## Version Light ####################################

    def light_forward_neighbor(self):
        """
        Crée un voisin en avant (mid) dans la version légère de l'arbre.
        """
        deltaT, _ = self.get_deltaT_alpha()

        self.forward_mid_neighbor = Node(self.value * math.exp(self.tree.market.rate * deltaT) - self.tree.dividend_value(self.step + 1), self.step + 1, self.tree)
        self.forward_mid_neighbor.backward_neighbor = self
    
    def light_generating_neighbors(self, NbSigma=5.63):
        """
        Génère les voisins en avant dans la version légère de l'arbre.

        Args:
            NbSigma (float): Facteur de scalabilité pour la génération des voisins.
        """
        self.step -= 1  # Décrémentation de l'étape pour le calcul
        deltaT, alpha = self.get_deltaT_alpha()
        self.step += 1  # Restauration de l'étape

        trunc = self
        current_up_node = self
        current_down_node = self

        if trunc.step != self.tree.N:
            actual_forward_up_node = self.forward_mid_neighbor.up_neighbor
            actual_forward_down_node = self.forward_mid_neighbor.down_neighbor

        k_up = k_down = min(self.step + 1, math.ceil(NbSigma * math.sqrt((self.step + 1) / 3)))
        i = j = 1
        if trunc.step == self.tree.N:
            trunc.option_price = trunc.tree.option.payoff(trunc.value)

        while i < k_up:
            current_up_node.up_neighbor = Node(current_up_node.value * alpha, current_up_node.step, current_up_node.tree)
            current_up_node.up_neighbor.down_neighbor = current_up_node

            if trunc.step == self.tree.N:
                current_up_node.up_neighbor.option_price = current_up_node.up_neighbor.tree.option.payoff(current_up_node.up_neighbor.value)
            else:
                if current_up_node.up_neighbor is not None:
                    expected_mid_value_for_up = current_up_node.up_neighbor.value * math.exp(current_up_node.tree.market.rate * deltaT) - current_up_node.tree.dividend_value(self.step + 1)
                    if expected_mid_value_for_up < actual_forward_up_node.value * (1 + alpha) / 2 and expected_mid_value_for_up > actual_forward_up_node.value * (1 + 1 / alpha) / 2:
                        current_up_node.update_forward_neighbors(direction='up')
                        current_up_node.up_neighbor.compute_probabilities()
                        current_up_node.up_neighbor.compute_option_price()
                        current_up_node.up_neighbor.delete_old_nodes()
                    else:
                        k_up += 1
                    
                    if actual_forward_up_node.up_neighbor is None:
                        break
                    actual_forward_up_node = actual_forward_up_node.up_neighbor

            current_up_node = current_up_node.up_neighbor
            i += 1

        while j < k_down:
            current_down_node.down_neighbor = Node(current_down_node.value / alpha, current_down_node.step, current_down_node.tree)
            current_down_node.down_neighbor.up_neighbor = current_down_node

            if trunc.step == self.tree.N:
                current_down_node.down_neighbor.option_price = current_down_node.down_neighbor.tree.option.payoff(current_down_node.down_neighbor.value)
            else:
                if current_down_node.down_neighbor is not None:
                    expected_mid_value_for_down = current_down_node.down_neighbor.value * math.exp(current_up_node.tree.market.rate * deltaT) - current_down_node.tree.dividend_value(self.step + 1)
                    if expected_mid_value_for_down <= actual_forward_down_node.value * (1 + alpha) / 2 and expected_mid_value_for_down >= actual_forward_down_node.value * (1 + 1 / alpha) / 2:
                        current_down_node.update_forward_neighbors(direction='down')
                        current_down_node.down_neighbor.compute_probabilities()
                        current_down_node.down_neighbor.compute_option_price()
                        current_down_node.down_neighbor.delete_old_nodes()
                    else:
                        k_down += 1
                    
                    if actual_forward_down_node.down_neighbor is None:
                        break
                    actual_forward_down_node = actual_forward_down_node.down_neighbor

            current_down_node = current_down_node.down_neighbor
            j += 1
        self = trunc
       
    def update_forward_neighbors(self, direction):
        """
        Met à jour les références des voisins en avant selon la direction spécifiée.

        Args:
            direction (str): La direction à mettre à jour, 'up' ou 'down'.
        """
        if direction == 'up':
            self.up_neighbor.forward_down_neighbor = self.forward_mid_neighbor
            self.up_neighbor.forward_mid_neighbor = self.forward_mid_neighbor.up_neighbor
            if self.forward_mid_neighbor.up_neighbor is not None:
                self.up_neighbor.forward_up_neighbor = self.forward_mid_neighbor.up_neighbor.up_neighbor
            
        elif direction == 'down':
            if self.forward_mid_neighbor.down_neighbor is not None:
                self.down_neighbor.forward_down_neighbor = self.forward_mid_neighbor.down_neighbor.down_neighbor
            self.down_neighbor.forward_mid_neighbor = self.forward_mid_neighbor.down_neighbor
            self.down_neighbor.forward_up_neighbor = self.forward_mid_neighbor
        
    def delete_old_nodes(self):
        """
        Supprime les anciens nœuds pour libérer de la mémoire.
        """
        if self.forward_mid_neighbor is not None:
            self.forward_mid_neighbor.forward_up_neighbor = None
            self.forward_mid_neighbor.forward_mid_neighbor = None
            self.forward_mid_neighbor.forward_down_neighbor = None

        if self.forward_up_neighbor is not None:
            self.forward_up_neighbor.forward_up_neighbor = None
            self.forward_up_neighbor.forward_mid_neighbor = None
            self.forward_up_neighbor.forward_down_neighbor = None

        if self.forward_down_neighbor is not None:
            self.forward_down_neighbor.forward_up_neighbor = None
            self.forward_down_neighbor.forward_mid_neighbor = None
            self.forward_down_neighbor.forward_down_neighbor = None

    def light_manage_forward_for_trunc(self):
        """
        Gère les voisins en avant pour le nœud de tronc dans la version légère.
        """
        self.forward_up_neighbor = self.forward_mid_neighbor.up_neighbor
        self.forward_down_neighbor = self.forward_mid_neighbor.down_neighbor

        self.compute_probabilities()
        self.compute_option_price()
        self.delete_old_nodes()
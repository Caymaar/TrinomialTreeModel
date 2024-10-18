import math

class Node:
    def __init__(self, value, step, tree):
        self.value = value
        self.step = step

        self.tree = tree
        self.option_price = 0.0

        self.cum_prob = 0.0

        self.up_neighbor = None
        self.down_neighbor = None
        self.backward_neighbor = None

        self.forward_up_neighbor = None
        self.forward_mid_neighbor = None
        self.forward_down_neighbor = None

        self.prob_forward_up_neighbor = None
        self.prob_forward_mid_neighbor = None
        self.prob_forward_down_neighbor = None

    def create_forward_neighbors(self):

        deltaT, alpha = self.get_deltaT_alpha()

        self.forward_mid_neighbor = Node(self.value * math.exp(self.tree.market.rate * deltaT) - self.tree.dividend_value(self.step), self.step + 1, self.tree)
        self.forward_mid_neighbor.backward_neighbor = self

        self.forward_up_neighbor = Node(self.forward_mid_neighbor.value * alpha, self.step + 1, self.tree)
        self.forward_mid_neighbor.up_neighbor = self.forward_up_neighbor
        self.forward_up_neighbor.down_neighbor = self.forward_mid_neighbor
        
        self.forward_down_neighbor = Node(self.forward_mid_neighbor.value / alpha, self.step + 1, self.tree)
        self.forward_mid_neighbor.down_neighbor = self.forward_down_neighbor
        self.forward_down_neighbor.up_neighbor = self.forward_mid_neighbor

        self.compute_probabilities()
        
    def monomial(self):
        self.forward_down_neighbor = None
        self.forward_up_neighbor = None
        self.prob_forward_mid_neighbor = 1.0

    def generate_upper_neighbors(self):

        deltaT, alpha = self.get_deltaT_alpha()

        trunc = self

        actual_forward_up_node = self.forward_up_neighbor

        while self.up_neighbor is not None:

            actual_forward_up_node.up_neighbor = Node(actual_forward_up_node.value * alpha, self.step + 1, self.tree)
            actual_forward_up_node.up_neighbor.down_neighbor = actual_forward_up_node
            
            supposed_mid_for_up_node_value = self.up_neighbor.value * math.exp(self.tree.market.rate * deltaT) - self.tree.dividend_value(self.step)

            if supposed_mid_for_up_node_value < actual_forward_up_node.value * (1 + alpha) / 2 and supposed_mid_for_up_node_value > actual_forward_up_node.value * (1 + 1 / alpha) / 2:

                self.up_neighbor.forward_up_neighbor = actual_forward_up_node.up_neighbor
                self.up_neighbor.forward_mid_neighbor = actual_forward_up_node
                self.up_neighbor.forward_down_neighbor = actual_forward_up_node.down_neighbor
                
                self.up_neighbor.compute_probabilities()

                if self.up_neighbor.forward_up_neighbor.cum_prob < self.tree.threshold and self.up_neighbor.up_neighbor is None:#) or i >= k:   

                    self.up_neighbor.monomial()

                    actual_forward_up_node.up_neighbor = None

                self = self.up_neighbor

            actual_forward_up_node = actual_forward_up_node.up_neighbor

        self = trunc

    def generate_lower_neighbors(self):

        deltaT, alpha = self.get_deltaT_alpha()

        trunc = self

        actual_forward_down_node = self.forward_down_neighbor

        while self.down_neighbor is not None:
            
            actual_forward_down_node.down_neighbor = Node(actual_forward_down_node.value / alpha, self.step + 1, self.tree)
            actual_forward_down_node.down_neighbor.up_neighbor = actual_forward_down_node
            
            supposed_mid_for_down_node_value = self.down_neighbor.value * math.exp(self.tree.market.rate * deltaT) - self.tree.dividend_value(self.step)

            if supposed_mid_for_down_node_value <= actual_forward_down_node.value * (1 + alpha) / 2 and supposed_mid_for_down_node_value >= actual_forward_down_node.value * (1 + 1 / alpha) / 2:
                
                self.down_neighbor.forward_up_neighbor = actual_forward_down_node.up_neighbor
                self.down_neighbor.forward_mid_neighbor = actual_forward_down_node
                self.down_neighbor.forward_down_neighbor = actual_forward_down_node.down_neighbor
                
                self.down_neighbor.compute_probabilities()
                
                if self.down_neighbor.forward_down_neighbor.cum_prob < self.tree.threshold and self.down_neighbor.down_neighbor is None:

                    self.down_neighbor.monomial()

                    actual_forward_down_node.down_neighbor = None

                self = self.down_neighbor

            actual_forward_down_node = actual_forward_down_node.down_neighbor

        self = trunc

    def compute_option_price(self):

        discount = math.exp(-self.tree.market.rate * self.tree.deltaT_array[self.step])

        option_value = 0.0

        if self.forward_up_neighbor is not None:
            option_value += self.forward_up_neighbor.option_price * self.prob_forward_up_neighbor

        if self.forward_mid_neighbor is not None:
            option_value += self.forward_mid_neighbor.option_price * self.prob_forward_mid_neighbor

        if self.forward_down_neighbor is not None:
            option_value += self.forward_down_neighbor.option_price * self.prob_forward_down_neighbor

        option_value *= discount

        if self.tree.option.style == "american":
            #self.payoff()
            self.option_price = self.tree.option.payoff(self.value)
            self.option_price = max(option_value, self.option_price)
        else:
            self.option_price = option_value

    def compute_probabilities(self):

        deltaT, alpha = self.get_deltaT_alpha()

        esp = self.value * math.exp(self.tree.market.rate * deltaT) - self.tree.dividend_value(self.step)
        var = self.value ** (2) * math.exp(2 * self.tree.market.rate * deltaT) * (math.exp(self.tree.market.sigma ** 2 * deltaT) - 1)

        down_prob = (self.forward_mid_neighbor.value ** (-2) * (var + esp ** 2) - 1 - (alpha + 1) * (esp / self.forward_mid_neighbor.value - 1)) / ((1 - alpha) * (alpha ** (-2) - 1))
        up_prob = (esp / self.forward_mid_neighbor.value - 1 - (1 / alpha - 1) * down_prob) / (alpha - 1)
        mid_prob = 1 - up_prob - down_prob

        self.prob_forward_down_neighbor = down_prob
        self.prob_forward_up_neighbor = up_prob
        self.prob_forward_mid_neighbor = mid_prob

        if self.forward_up_neighbor is not None:
            self.forward_up_neighbor.cum_prob += self.cum_prob * self.prob_forward_up_neighbor

        if self.forward_mid_neighbor is not None:
            self.forward_mid_neighbor.cum_prob += self.cum_prob * self.prob_forward_mid_neighbor

        if self.forward_down_neighbor is not None:
            self.forward_down_neighbor.cum_prob += self.cum_prob * self.prob_forward_down_neighbor

    def get_deltaT_alpha(self):
        if self.step >= len(self.tree.deltaT_array):
            raise IndexError(f"Index {self.step} is out of bounds for deltaT_array with size {len(self.tree.deltaT_array)}")
        
        deltaT = self.tree.deltaT_array[self.step]
        alpha = math.exp(self.tree.market.sigma * math.sqrt(3 * deltaT))
        return deltaT, alpha

    def __repr__(self):
        prob_up = f"{self.prob_forward_up_neighbor:.5f}" if self.prob_forward_up_neighbor is not None else ""
        prob_mid = f"{self.prob_forward_mid_neighbor:.5f}" if self.prob_forward_mid_neighbor is not None else ""
        prob_down = f"{self.prob_forward_down_neighbor:.5f}" if self.prob_forward_down_neighbor is not None else ""
        
        return f"v={self.value:.5f}, step={self.step}, op={self.option_price:.5f}, up={prob_up}, mid={prob_mid}, down={prob_down}"
    
    def __hash__(self):
        return hash((self.value, self.step))

    def __eq__(self, other):
        return isinstance(other, Node) and self.value == other.value and self.step == other.step

    def __str__(self):
        return f"v={self.value:.5f}, step={self.step}, op={self.option_price:.5f}"
    ############################## LIGHT VERSION ####################################

    def light_forward_neighbor(self):

        deltaT, _ = self.get_deltaT_alpha()

        self.forward_mid_neighbor = Node(self.value * math.exp(self.tree.market.rate * deltaT) - self.tree.dividend_value(self.step), self.step + 1, self.tree)
        self.forward_mid_neighbor.backward_neighbor = self

    def light_generating_neighbors(self, NbSigma=5.63):

        self.step -= 1
        deltaT, alpha = self.get_deltaT_alpha()
        self.step += 1

        trunc = self
        current_up_node = self
        current_down_node = self

        if trunc.step != self.tree.N:
            actual_forward_up_node = self.forward_mid_neighbor.up_neighbor
            actual_forward_down_node = self.forward_mid_neighbor.down_neighbor

        k = min(self.step + 1, math.ceil(NbSigma * math.sqrt((self.step + 1) / 3)))

        if trunc.step == self.tree.N:
            trunc.option_price = trunc.tree.option.payoff(trunc.value)
        
        for _ in range(1, k):

            current_up_node.up_neighbor = Node(current_up_node.value * alpha, current_up_node.step, current_up_node.tree)
            current_up_node.up_neighbor.down_neighbor = current_up_node

            current_down_node.down_neighbor = Node(current_down_node.value / alpha, current_down_node.step, current_down_node.tree)
            current_down_node.down_neighbor.up_neighbor = current_down_node
            
            if trunc.step == self.tree.N:
    
                current_up_node.up_neighbor.option_price = current_up_node.up_neighbor.tree.option.payoff(current_up_node.up_neighbor.value)
                current_down_node.down_neighbor.option_price = current_down_node.down_neighbor.tree.option.payoff(current_down_node.down_neighbor.value)

            elif current_up_node.up_neighbor is not None:
                
                expected_mid_value_for_up = current_up_node.up_neighbor.value * math.exp(current_up_node.tree.market.rate * deltaT) - current_up_node.tree.dividend_value(self.step)
                if expected_mid_value_for_up < actual_forward_up_node.value * (1 + alpha) / 2 and expected_mid_value_for_up > actual_forward_up_node.value * (1 + 1 / alpha) / 2:

                    current_up_node.update_forward_neighbors(direction='up')

                    current_up_node.up_neighbor.compute_probabilities()
                    current_up_node.up_neighbor.compute_option_price()
                    current_up_node.up_neighbor.delete_old_nodes()

                actual_forward_up_node = actual_forward_up_node.up_neighbor

            elif current_down_node.down_neighbor is not None:
                    
                expected_mid_value_for_down = current_down_node.down_neighbor.value * math.exp(current_up_node.tree.market.rate * deltaT) - current_down_node.tree.dividend_value(self.step)
                if expected_mid_value_for_down <= actual_forward_down_node.value * (1 + alpha) / 2 and expected_mid_value_for_down >= actual_forward_down_node.value * (1 + 1 / alpha) / 2:
 
                    current_down_node.update_forward_neighbors(direction='down')

                    current_down_node.down_neighbor.compute_probabilities()
                    current_down_node.down_neighbor.compute_option_price()
                    current_down_node.down_neighbor.delete_old_nodes()
                
                actual_forward_down_node = actual_forward_down_node.down_neighbor

            current_up_node = current_up_node.up_neighbor
            current_down_node = current_down_node.down_neighbor

        self = trunc
        
    def update_forward_neighbors(self, direction):
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

        self.forward_up_neighbor = self.forward_mid_neighbor.up_neighbor
        self.forward_down_neighbor = self.forward_mid_neighbor.down_neighbor

        self.compute_probabilities()
        self.compute_option_price()
        self.delete_old_nodes()

    """
    def light_up_neighbors(self):
        self.step -= 1
        deltaT, alpha = self.get_deltaT_alpha()
        self.step += 1
        trunc = self

        k = min(self.step + 1, math.ceil(4 * math.sqrt(self.step + 1 / 3)))
         # Arrondir à la valeur supérieure et convertir en entier
        print(k)

        for _ in range(1, k, 1):

            self.up_neighbor = Node(self.value * alpha, self.step, self.tree)
            self.up_neighbor.down_neighbor = self

            if self.up_neighbor.step == self.tree.N and self.up_neighbor is not None:

                self.option_price = self.tree.option.payoff(self.value)
                self.up_neighbor.option_price = self.up_neighbor.tree.option.payoff(self.up_neighbor.value)
            
            elif self.up_neighbor is not None:

                self.update_forward_neighbors(direction='up')

                self.up_neighbor.compute_probabilities()
                self.up_neighbor.compute_option_price()

            self = self.up_neighbor

        self = trunc

    def light_lower_neighbors(self):
        self.step -= 1
        deltaT, alpha = self.get_deltaT_alpha()
        self.step += 1
        trunc = self

        k = min(self.step + 1, math.ceil(4 * math.sqrt(self.step + 1 / 3)))
        
        for _ in range(1, k, 1):
            
            self.down_neighbor = Node(self.value / alpha, self.step, self.tree)
            self.down_neighbor.up_neighbor = self
            
            if self.down_neighbor.step == self.tree.N and self.down_neighbor is not None:

                self.down_neighbor.option_price = self.down_neighbor.tree.option.payoff(self.down_neighbor.value)

            elif self.down_neighbor is not None:

                expected_mid_value = self.down_neighbor.value * math.exp(self.tree.market.rate * deltaT) - self.tree.dividend_value(self.step)
                if expected_mid_value <= self.forward_mid_neighbor.down_neighbor.value * (1 + alpha) / 2 and expected_mid_value >= self.forward_mid_neighbor.down_neighbor.value * (1 + 1 / alpha) / 2:
 
                    self.update_forward_neighbors(direction='down')

                    self.down_neighbor.compute_probabilities()
                    self.down_neighbor.compute_option_price()



            self = self.down_neighbor

        self = trunc
    """
        
    """
    def payoff(self):

        if self.tree.option.type == "call":
            self.option_price = max(self.value - self.tree.option.K, 0)
        else:
            self.option_price = max(self.tree.option.K - self.value, 0)"""
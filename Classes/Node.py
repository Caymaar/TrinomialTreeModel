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

    def create_forward_nodes(self):

        deltaT, alpha = self.get_deltaT_alpha()

        self.forward_mid_neighbor = Node(self.value * math.exp(self.tree.market.rate * deltaT) - self.tree.dividend_value(self.step), self.step + 1, self.tree)
        self.forward_mid_neighbor.backward_neighbor = self

        self.forward_up_neighbor = Node(self.forward_mid_neighbor.value * alpha, self.step + 1, self.tree)
        self.forward_mid_neighbor.up_neighbor = self.forward_up_neighbor
        self.forward_mid_neighbor.up_neighbor.down_neighbor = self.forward_mid_neighbor
        
        self.forward_down_neighbor = Node(self.forward_mid_neighbor.value / alpha, self.step + 1, self.tree)
        self.forward_mid_neighbor.down_neighbor = self.forward_down_neighbor
        self.forward_mid_neighbor.down_neighbor.up_neighbor = self.forward_mid_neighbor

        self.compute_probabilities()

    def generate_upper_neighbors(self):

        deltaT, alpha = self.get_deltaT_alpha()

        trunc = self

        actual_forward_up_node = self.forward_up_neighbor

        while self.up_neighbor is not None:

            actual_forward_up_node.up_neighbor = Node(actual_forward_up_node.value * alpha, self.step + 1, self.tree)
            actual_forward_up_node.up_neighbor.down_neighbor = actual_forward_up_node
            
            supposed_mid_for_up_node_value = self.up_neighbor.value * math.exp(self.tree.market.rate * deltaT) - self.tree.dividend_value(self.step)

            if supposed_mid_for_up_node_value < actual_forward_up_node.value * (1 + alpha) / 2 and supposed_mid_for_up_node_value > actual_forward_up_node.value * (1 + 1 / alpha) / 2:

                self.up_neighbor.forward_down_neighbor = actual_forward_up_node.down_neighbor
                self.up_neighbor.forward_mid_neighbor = actual_forward_up_node
                self.up_neighbor.forward_up_neighbor = actual_forward_up_node.up_neighbor

                self.up_neighbor.compute_probabilities()

                if self.up_neighbor.forward_up_neighbor.cum_prob < self.tree.threshold:
                    self.up_neighbor.forward_up_neighbor = None
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
                
                self.down_neighbor.forward_down_neighbor = actual_forward_down_node.down_neighbor
                self.down_neighbor.forward_mid_neighbor = actual_forward_down_node
                self.down_neighbor.forward_up_neighbor = actual_forward_down_node.up_neighbor
                
                self.down_neighbor.compute_probabilities()
                
                if self.down_neighbor.forward_down_neighbor.cum_prob < self.tree.threshold:
                    self.down_neighbor.forward_down_neighbor = None
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
            self.payoff()
            self.option_price = max(option_value, self.option_price)
        else:
            self.option_price = option_value
        
    def payoff(self):

        if self.tree.option.type == "call":
            self.option_price = max(self.value - self.tree.option.K, 0)
        else:
            self.option_price = max(self.tree.option.K - self.value, 0)

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

        deltaT = self.tree.deltaT_array[self.step]
        alpha = math.exp(self.tree.market.sigma * math.sqrt(3 * deltaT))
        return deltaT, alpha

    def __repr__(self):
        return f"value={self.value:.2f}, step={self.step}, option_price={self.option_price:.2f}, cum_prob={self.cum_prob:.5f}"

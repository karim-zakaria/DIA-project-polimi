import numpy as np

np.random.seed(41148)


"""
This environment assumes:
      -Stationary enivronment
      -Sequential customer arrival i.e. return reward for each customer based on class and price not whole round reward
"""


class SequentialArrivalEnvironment:
    def __init__(self, margin1, margin2, conv_rate1, conv_rate2):
        self.margin1 = margin1
        self.margin2 = margin2
        self.conv_rate1 = conv_rate1
        self.conv_rate2 = conv_rate2

    def sub_round_1(self, customer_class, price_candidate):
        """sample conversion based on customer class and price candidate and return received reward
            used for sampling rewards for product 1"""
        conv_sample = np.random.binomial(1, self.conv_rate1[customer_class, price_candidate])
        reward = self.margin1[price_candidate] * conv_sample
        return reward
# customer_class, 0, chosen_promo
    def sub_round_2(self, customer_class, price_candidate, promo):
        """sample conversion based on customer class, price candidate and promo and return received reward
            used for sampling rewards for product 2"""
        conv_sample = np.random.binomial(1, self.conv_rate2[customer_class, price_candidate, promo])
        reward = self.margin2[price_candidate, promo] * conv_sample
        return reward

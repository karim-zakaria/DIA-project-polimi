import numpy as np
from pricing.enviroment.Sequential_Arrival_Environment import SequentialArrivalEnvironment
np.random.seed(41148)



"""
This environment assumes:
      -Non Stationary enivronment
      -Sequential customer arrival i.e. return reward for each customer based on class and price not whole round reward
"""


class NonStationarySequentialEnvironment(SequentialArrivalEnvironment):
    def __init__(self, margin1, margin2, conv_rate1, conv_rate2, horizon, n_phases):
        super().__init__(margin1, margin2, conv_rate1, conv_rate2)
        self.t = 0
        self.horizon = horizon
        self.n_phases = n_phases
        self.phase_length = int(horizon/n_phases)

    def sub_round_1(self, customer_class, price_candidate):
        """sample conversion based on customer class and price candidate and return received reward
            used for sampling rewards for product 1"""
        current_phase = min(int(self.t / self.phase_length), self.n_phases-1)
        conv_sample = np.random.binomial(1, self.conv_rate1[current_phase, customer_class, price_candidate])
        reward = self.margin1[price_candidate] * conv_sample
        return reward

    def sub_round_2(self, customer_class, price_candidate, promo):
        """sample conversion based on customer class, price candidate and promo and return received reward
            used for sampling rewards for product 2"""
        current_phase = min(int(self.t / self.phase_length), self.n_phases-1)
        conv_sample = np.random.binomial(1, self.conv_rate2[current_phase, customer_class, price_candidate, promo])
        reward = self.margin2[price_candidate, promo] * conv_sample
        return reward


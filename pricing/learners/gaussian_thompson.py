# Explanation: https://towardsdatascience.com/thompson-sampling-fc28817eacb8

from pricing.learners.learner import Learner
import numpy as np


class Gaussian_Thompson(Learner):

    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.gaussian_parameters = np.ones((n_arms, 2)) * 50000
        self.gaussian_parameters[:, 1] = np.ones(n_arms) * 0.00001

    def pull_arm(self):
        idx = np.argmax(
            np.random.randn(self.n_arms) / np.sqrt(self.gaussian_parameters[:, 1]) + self.gaussian_parameters[:, 0])
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.gaussian_parameters[pulled_arm, 0] = ((self.gaussian_parameters[pulled_arm, 1] * self.gaussian_parameters[
            pulled_arm, 0]) + sum(self.rewards_per_arm[pulled_arm])) / (self.gaussian_parameters[pulled_arm, 1] + len(
            self.rewards_per_arm[pulled_arm]))
        self.gaussian_parameters[pulled_arm, 1] = self.gaussian_parameters[pulled_arm, 1] + 1

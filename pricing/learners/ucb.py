import numpy as np
from scipy.optimize import linear_sum_assignment

from pricing.learners.learner import Learner
from matching.cumsum import CUMSUM


class UCB(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.array([np.inf] * n_arms)

    def pull_arm(self):
        return np.argmax(self.empirical_means + self.confidence)

    def update(self, pull_arm, reward):
        self.t += 1
        self.empirical_means[pull_arm] = (self.empirical_means[pull_arm] * (self.t - 1) + reward) / self.t
        for a in range(self.n_arms):
            n_samples = len(self.rewards_per_arm[a])
            self.confidence[a] = (2 * np.log(self.t) / n_samples) ** 0.5 if n_samples > 0 else np.inf
        self.update_observations(pull_arm, reward)


class SW_UCB(UCB):
    def __init__(self, n_arms, window_size):
        super(SW_UCB, self).__init__(n_arms)
        self.window_size = window_size

    def pull_arm(self):
        for a in range(self.n_arms):
            self.clear_outside_window_rewards(a)
        return np.argmax(self.empirical_means + self.confidence)

    def update(self, pulled_arm, reward):
        self.update_observations(pulled_arm, reward)
        total_samples = 0
        for a in range(self.n_arms):
            total_samples += len(self.rewards_per_arm[a])
        n_samples = len(self.rewards_per_arm[pulled_arm])
        self.empirical_means[pulled_arm] = np.mean(np.array([r[0] for r in self.rewards_per_arm[pulled_arm]])) \
            if n_samples > 0 else 0
        self.confidence[pulled_arm] = (2 * np.log(min(self.t, self.window_size)) / n_samples) ** 0.5 if n_samples > 0 else np.inf

    # Remove rewards that are outside of the window for a given arm. Since the array is ordered by the timestamp and
    # we execute this function every round, we only need to check the timestamp of the first element
    def clear_outside_window_rewards(self, arm):
        if len(self.rewards_per_arm[arm]) == 0:
            return
        min_timestamp = self.t - self.window_size
        temp_reward_list = []
        for r in self.rewards_per_arm[arm]:
            if r[1] >= min_timestamp:
                temp_reward_list.append(r)
        self.rewards_per_arm[arm] = temp_reward_list

        total_samples = 0
        for a in range(self.n_arms):
            total_samples += len(self.rewards_per_arm[a])

        n_samples = len(self.rewards_per_arm[arm])
        self.empirical_means[arm] = np.mean(np.array([r[0] for r in self.rewards_per_arm[arm]])) if n_samples > 0 else 0
        self.confidence[arm] = (2 * np.log(min(self.t, self.window_size)) / n_samples) ** 0.5 if n_samples > 0 else np.inf

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append((reward, self.t))  # add a timestamp when an arm has been pulled
        self.collected_rewards = np.append(self.collected_rewards, reward)

class CUMSUM_UCB(UCB):
    def __init__(self, n_arms, M=100, eps=0.05, h=20, alpha=0.01):
        super().__init__(n_arms)
        self.change_detection = [CUMSUM(M, eps, h) for _ in range(n_arms)]
        self.valid_rewards_per_arms = [[] for _ in range(n_arms)]
        self.detections = [[] for _ in range(n_arms)]
        self.alpha = alpha

    def pull_arm(self):
        if np.random.binomial(1, 1 - self.alpha):
            return np.argmax(self.empirical_means + self.confidence)
        else:
            return np.random.randint(self.n_arms)

    def update(self, pulled_arm, reward):
        self.t += 1
        if self.change_detection[pulled_arm].update(reward):
            self.detections[pulled_arm].append(self.t)
            self.valid_rewards_per_arms[pulled_arm] = []
            self.change_detection[pulled_arm].reset()
        self.update_observations(pulled_arm, reward)
        self.empirical_means[pulled_arm] = np.mean(self.valid_rewards_per_arms[pulled_arm])
        total_valid_samples = sum([len(x) for x in self.valid_rewards_per_arms])

        for a in range(self.n_arms):
            n_samples = len(self.valid_rewards_per_arms[a])
            self.confidence[a] = (2 * np.log(total_valid_samples) / n_samples) ** 0.5 if n_samples > 0 else np.inf

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.valid_rewards_per_arms[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)



class Matching_UCB(UCB):
    def __init__(self, n_arms, n_rows, n_cols, col_promo):
        super().__init__(n_arms)
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.col_promo = col_promo
        assert n_arms == n_cols * n_rows

    def pull_arms(self, promos_remaining):
        upper_conf = self.empirical_means + self.confidence
        upper_conf[np.isinf(upper_conf)] = 1e3
        cost_matrix = -upper_conf.reshape(self.n_rows, self.n_cols)

        # if there are no more promotion of a certain type we set the cost of that column as high as possible
        for c in range(self.n_cols):
            if self.col_promo[c] != 0:
                if promos_remaining[self.col_promo[c]-1] == 0:
                    for r in range(self.n_rows):
                        cost_matrix[r][c] = np.inf

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return row_ind, col_ind

    def pull_arms_2(self, promos_remaining): #The only difference is that this function also returns the value of the objective function
        upper_conf = self.empirical_means + self.confidence
        upper_conf[np.isinf(upper_conf)] = 1e3
        cost_matrix = -upper_conf.reshape(self.n_rows, self.n_cols)

        # if there are no more promotion of a certain type we set the cost of that column as high as possible
        for x in range(len(promos_remaining)):
            if promos_remaining[x] == 0:
                for y in range(self.n_rows):
                    cost_matrix[y][x+self.n_rows] = np.inf # self.n_rows bcs we want to start from the first real promo

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return row_ind, col_ind, cost_matrix[row_ind, col_ind].sum()

    def update(self, pulled_arms, rewards):
        self.t += 1
        pulled_arms_flat = np.ravel_multi_index(pulled_arms, (self.n_rows, self.n_cols))

        for pulled_arm, reward in zip(pulled_arms_flat, rewards):
            self.update_observations(pulled_arm, reward)
            self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * (self.t - 1) + reward) / self.t

        for a in range(self.n_arms):
            n_samples = len(self.rewards_per_arm[a])
            self.confidence[a] = (2 * np.log(self.t) / n_samples) ** 0.5 if n_samples > 0 else np.inf

    def update_one(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * (self.t - 1) + reward) / self.t
        for a in range(self.n_arms):
            n_samples = len(self.rewards_per_arm[a])
            self.confidence[a] = (2 * np.log(self.t) / n_samples) ** 0.5 if n_samples > 0 else np.inf


class SW_Matching_UCB(Matching_UCB):
    def __init__(self, n_arms, n_rows, n_cols, col_promo, window_size):
        super().__init__(n_arms, n_rows, n_cols, col_promo)
        self.window_size = window_size

    def pull_arms(self, promos_remaining):
        for a in range(self.n_arms):
            self.clear_outside_window_rewards(a)
        upper_conf = self.empirical_means + self.confidence
        upper_conf[np.isinf(upper_conf)] = 1e3
        cost_matrix = -upper_conf.reshape(self.n_rows, self.n_cols)

        # if there are no more promotion of a certain type we set the cost of that column as high as possible
        for c in range(self.n_cols):
            if self.col_promo[c] != 0:
                if promos_remaining[self.col_promo[c]-1] == 0:
                    for r in range(self.n_rows):
                        cost_matrix[r][c] = np.inf

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return row_ind, col_ind

    def update_one(self, pulled_arm, reward):
        self.update_observations(pulled_arm, reward)
        total_samples = 0
        for a in range(self.n_arms):
            total_samples += len(self.rewards_per_arm[a])
        n_samples = len(self.rewards_per_arm[pulled_arm])
        self.empirical_means[pulled_arm] = np.mean(np.array([r[0] for r in self.rewards_per_arm[pulled_arm]])) \
            if n_samples > 0 else 0
        self.confidence[pulled_arm] = (2 * np.log(min(self.t, self.window_size)) / n_samples) ** 0.5 if n_samples > 0 else np.inf

    # Remove rewards that are outside of the window for a given arm. Since the array is ordered by the timestamp and
    # we execute this function every round, we only need to check the timestamp of the first element
    def clear_outside_window_rewards(self, arm):
        if len(self.rewards_per_arm[arm]) == 0:
            return
        min_timestamp = self.t - self.window_size
        temp_reward_list = []
        for r in self.rewards_per_arm[arm]:
            if r[1] >= min_timestamp:
                temp_reward_list.append(r)
        self.rewards_per_arm[arm] = temp_reward_list

        total_samples = 0
        for a in range(self.n_arms):
            total_samples += len(self.rewards_per_arm[a])

        n_samples = len(self.rewards_per_arm[arm])
        self.empirical_means[arm] = np.mean(np.array([r[0] for r in self.rewards_per_arm[arm]])) if n_samples > 0 else 0
        self.confidence[arm] = (2 * np.log(min(self.t, self.window_size)) / n_samples) ** 0.5 if n_samples > 0 else np.inf

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append((reward, self.t))  # add a timestamp when an arm has been pulled
        self.collected_rewards = np.append(self.collected_rewards, reward)


class CUMSUM_Matching_UCB(Matching_UCB):
    def __init__(self, n_arms, n_rows, n_cols, col_promo, M=100, eps=0.05, h=20, alpha=0.01):
        super().__init__(n_arms, n_rows, n_cols, col_promo)
        self.change_detection = [CUMSUM(M, eps, h) for _ in range(n_arms)]
        self.valid_rewards_per_arms = [[] for _ in range(n_arms)]
        self.detections = [[] for _ in range(n_arms)]
        self.alpha = alpha

    def pull_arm(self):
        if np.random.binomial(1, 1 - self.alpha):
            upper_conf = self.empirical_means + self.confidence
            upper_conf[np.isinf(upper_conf)] = 1e3
            row_ind, col_ind = linear_sum_assignment(-upper_conf.reshape(self.n_rows, self.n_cols))
            return row_ind, col_ind
        else:
            costs_random = np.random.randint(0, 10, size=(self.n_rows, self.n_cols))
            return linear_sum_assignment(costs_random)

    def update(self, pulled_arms, rewards):
        self.t += 1
        pulled_arms_flat = np.ravel_multi_index(pulled_arms, (self.n_rows, self.n_cols))

        for pulled_arm, reward in zip(pulled_arms_flat, rewards):
            if self.change_detection[pulled_arm].update(reward):
                self.detections[pulled_arm].append(self.t)
                self.valid_rewards_per_arms[pulled_arm] = []
                self.change_detection[pulled_arm].reset()
            self.update_observations(pulled_arm, reward)
            self.empirical_means[pulled_arm] = np.mean(self.valid_rewards_per_arms[pulled_arm])
        total_valid_samples = sum([len(x) for x in self.valid_rewards_per_arms])

        for a in range(self.n_arms):
            n_samples = len(self.valid_rewards_per_arms[a])
            self.confidence[a] = (2 * np.log(total_valid_samples) / n_samples) ** 0.5 if n_samples > 0 else np.inf

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.valid_rewards_per_arms[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)

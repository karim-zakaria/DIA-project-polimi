import random

import numpy as np
import matplotlib.pyplot as plt

from random import randrange
from pricing.learners.ucb import UCB, Matching_UCB
from pricing.enviroment.Sequential_Arrival_Environment import SequentialArrivalEnvironment

np.random.seed(41148)
random.seed(41148)

# time horizon
T = 365

# number of customers of each class
N_CLASSES = 4
n_class = np.array([200, 150, 50, 100])

# respective costs of the two products
COST1 = 550
COST2 = 50

# price candidates for each product
N_PRICES = 7
price1 = np.array([600, 700, 800, 900, 1000, 1100, 1200])

price2 = np.array([70, 75, 80, 85, 90, 95, 100])

# profit margin for each price candidate for product 1
margin1 = np.array([50, 150, 250, 350, 450, 550, 650]) / COST1

# profit margin for price candidate (axis 0) and promo (axis 1)
margin2 = np.array([[20, 13, 6, 2.5],
                    [25, 17.5, 10, 6.25],
                    [30, 22, 14, 10],
                    [35, 26.5, 18, 13.75],
                    [40, 31, 22, 17.5],
                    [45, 35.5, 26, 21.25],
                    [50, 40, 30, 25]]) / COST2

# conversion of each class for each price candidate of product 1
conv_1 = np.array([[0.45, 0.6, 0.57, 0.52, 0.37, 0.15, 0.08],
                   [0.5, 0.55, 0.51, 0.47, 0.42, 0.35, 0.21],
                   [0.45, 0.42, 0.35, 0.27, 0.14, 0.1, 0.05],
                   [0.65, 0.7, 0.67, 0.55, 0.3, 0.21, 0.11]])

# conversion of each class for each price candidate (axis 0) and promo (axis 1) of product 2
conv_2 = np.array([[[0.57, 0.6, 0.67, 0.69],
                    [0.55, 0.58, 0.65, 0.67],
                    [0.53, 0.56, 0.6, 0.65],
                    [0.48, 0.55, 0.58, 0.6],
                    [0.46, 0.53, 0.56, 0.58],
                    [0.43, 0.48, 0.55, 0.56],
                    [0.32, 0.46, 0.53, 0.55]],
                   [[0.38, 0.4, 0.49, 0.53],
                    [0.35, 0.39, 0.43, 0.49],
                    [0.31, 0.37, 0.4, 0.43],
                    [0.25, 0.35, 0.39, 0.4],
                    [0.22, 0.31, 0.37, 0.39],
                    [0.2, 0.25, 0.35, 0.37],
                    [0.18, 0.22, 0.31, 0.35]],
                   [[0.17, 0.24, 0.37, 0.39],
                    [0.12, 0.2, 0.3, 0.36],
                    [0.08, 0.15, 0.23, 0.3],
                    [0.06, 0.12, 0.2, 0.23],
                    [0.06, 0.08, 0.15, 0.2],
                    [0.05, 0.06, 0.12, 0.16],
                    [0.04, 0.06, 0.08, 0.12]],
                   [[0.49, 0.53, 0.65, 0.68],
                    [0.35, 0.5, 0.55, 0.65],
                    [0.29, 0.46, 0.52, 0.55],
                    [0.25, 0.34, 0.5, 0.52],
                    [0.21, 0.29, 0.46, 0.5],
                    [0.18, 0.25, 0.34, 0.46],
                    [0.15, 0.21, 0.26, 0.35]]])

# two alternative settings for number of promos of each class
N_PROMOS = 4
promo_setting_1 = np.array([0.3, 0.15, 0.25])
promo_setting_2 = np.array([0.25, 0.35, 0.2])

promo_setting = promo_setting_1

promo_assignment = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0]])

col_promo = [0, 0, 0, 0, 1, 2, 3]

n_arms1 = int(conv_1.size / N_CLASSES)  # for disaggregate model n_arms1 = conv_1.size
n_arms2 = int(conv_2.size / N_CLASSES / N_PROMOS)  # for disaggregate model n_arms2 = conv_2.size


def random_positive_choice(iterable):
    """random choice of index from non-zero elements of iterable"""
    iterable = np.array(iterable)
    indices = np.argwhere(iterable > 0).reshape(-1)
    index = np.random.choice(indices)

    return index


# Task 5
def main():
    # ENVIRONMENT DEFINITION
    environment = SequentialArrivalEnvironment(margin1=margin1, margin2=margin2, conv_rate1=conv_1, conv_rate2=conv_2)

    # initialize variable for average of daily customers per day to calculate number of promos
    empirical_customer_amount = 300

    # LEARNER DEFINITION
    learner1 = UCB(N_CLASSES)
    extra_promos = N_CLASSES - 1  # we create additional copies of p0 as a hack for the linear sum assignment
    all_promos = N_PROMOS + extra_promos
    learner2 = Matching_UCB(all_promos * N_CLASSES, N_CLASSES, all_promos, col_promo)

    def expected_value_of_reward(pulled_arm1, pulled_arm2, current_customer_class, curr_promo):
        """calculate and return expected value of reward for arm choices based on conversion rates"""
        reward = margin1[pulled_arm1] * conv_1[current_customer_class, pulled_arm1] * COST1
        if pulled_arm2 != -1:
            reward += margin2[pulled_arm2, curr_promo] * conv_1[current_customer_class, pulled_arm1] \
                      * conv_2[current_customer_class, pulled_arm2, curr_promo] * COST2
        return reward

    #
    # START LEARNING PROCESS
    #
    print("---- Start Learning Process ----")

    rewards1 = []
    rewards2 = []
    expected_rewards = []
    clairvoyant_expected_rewards = []
    arms1 = []
    arms2 = []
    for i in range(T):
        print(f"  Progress: {i}/{T} days", end="\r") if i % 10 == 0 else False
        # sample number of customer for each class and truncate at 0 to avoid negative
        round_class_num = np.random.normal(n_class, 10)
        round_class_num = [int(n) if n >= 0 else 0 for n in round_class_num]
        daily_customer_amount = sum(round_class_num)

        daily_promos = [n * empirical_customer_amount for n in promo_setting]

        # initialize variables for accumulating round rewards
        round_reward1 = 0
        round_reward2 = 0
        round_expected_reward = 0
        round_clairvoyant_expected = 0
        for c in range(daily_customer_amount):
            # simulate customer arrival by random choice of class which has customers remaining for the day
            customer_class = random_positive_choice(iterable=round_class_num)
            round_class_num[customer_class] -= 1
            chosen_promo = 0

            # pull price 1 arm, observe rewards
            arm1 = learner1.pull_arm()
            reward1 = environment.sub_round_1(customer_class, 0)  # The second parameter is 0 due to fixed prices

            # pull price 2 arm if positive reward and update else reward2 = 0
            if reward1 > 0:
                row_ind, col_ind = learner2.pull_arms(daily_promos)
                chosen_promo = col_ind[customer_class] - extra_promos

                if chosen_promo < 0:
                    chosen_promo = 0
                if chosen_promo > 0:
                    daily_promos[chosen_promo - 1] -= 1  # daily_promos [#p1 #p2 #p3], chosen_promo [p0 p1 p2 p3]

                reward2 = environment.sub_round_2(customer_class, 0,
                                                  chosen_promo)  # The second parameter is 0 due to fixed prices.  chosen_promo+1 since [p0 p1 p2 p3]
                arm2 = customer_class * all_promos + chosen_promo
                if chosen_promo == 0:
                    #  Update all arms that correspond to P0 for a given customer_class
                    for promo in range(extra_promos + 1):
                        learner2.update_one(customer_class * all_promos + promo, reward2)
                else:
                    learner2.update_one(arm2, reward2)
            else:
                reward2 = 0
                arm2 = -1

            # update learner 1
            learner1.update(arm1, reward1 + reward2)

            arms1.append(arm1)
            arms2.append(arm2)

            # add rewards to cumulative sums of round rewards and calculate expected rewards
            round_reward1 += reward1 * COST1
            round_reward2 += reward2 * COST2
            round_expected_reward += expected_value_of_reward(0, 0, customer_class, chosen_promo)  # first and
            # second parameters are 0 due to fixed prices
            round_clairvoyant_expected += np.max([expected_value_of_reward(0, 0, customer_class, p)
                                                   for p in range(N_PROMOS)])

        # append round rewards to lists of rewards
        rewards1.append(round_reward1)
        rewards2.append(round_reward2)
        expected_rewards.append(round_expected_reward)
        clairvoyant_expected_rewards.append(round_clairvoyant_expected)

        # update empirical number of customers per day
        empirical_customer_amount = (empirical_customer_amount * i + daily_customer_amount)/(i + 1)

    rewards1 = np.array(rewards1)
    rewards2 = np.array(rewards2)
    expected_rewards = np.array(expected_rewards)
    clairvoyant_expected_rewards = np.array(clairvoyant_expected_rewards)
    rewards = rewards1 + rewards2

    print(f"  Progress: {T}/{T} days")

    #
    # LEARNING RESULTS
    #
    print()
    print('LEARNING RESULTS')
    print()
    print(f'Total profit collected from product 1: {np.sum(rewards1)}')
    print(f'Total profit collected from product 2: {np.sum(rewards2)}')

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(np.cumsum(rewards1), label='Product 1')
    ax1.legend(loc='lower right')
    ax2.plot(np.cumsum(rewards2), label='Product 2')
    ax2.legend(loc='lower right')
    fig.suptitle('Cumulative Rewards from each product')

    plt.show()

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    plt.plot(moving_average(rewards, 10))
    plt.plot(np.ones(len(rewards))*np.mean(rewards),label = "Average Daily Rewards", color='tab:blue', linestyle=(0,(5,10)))
    plt.title('10-day moving average of rewards collected')
    plt.show()

    #
    # REGRETS
    #
    # regrets are calculated in terms of the expected value of the reward for each pulled arm

    print()
    print(f'Total expected regret: {np.sum(np.subtract(clairvoyant_expected_rewards, expected_rewards))}')

    plt.plot(moving_average(expected_rewards, 10), label='Expected rewards for UCB')
    plt.plot(np.ones(len(expected_rewards))*np.mean(expected_rewards), color='tab:blue', linestyle=(0,(5,10)))
    plt.plot(moving_average(clairvoyant_expected_rewards, 10), label='Expected rewards for Clairvoyant Algorithm',
             color='r')
    plt.plot(np.ones(len(clairvoyant_expected_rewards))*np.mean(clairvoyant_expected_rewards), color='r', linestyle=(0,(5,10)))
    plt.legend(loc='lower right')
    plt.title('10-day moving average of expected rewards of each algorithm')
    plt.show()


if __name__ == '__main__':
    main()

"""
- Point 7 in assignment
- Solution of pricing anf matching problem
- Daily customer number drawn from gaussian dist and not known
- Learning process applied for each arriving customer each day
- For each arriving customer class drawn at random according to class distribution
- Promo level distribution as percentage of total constant (2 settings available)
- Number of available promos calculated using variable that holds average total number of customers
- Non Stationary Environment -> Sliding Window Learners
"""

import random
import numpy as np
import matplotlib.pyplot as plt

from pricing.learners.ucb import SW_UCB, SW_Matching_UCB
from pricing.enviroment.Non_Stationary_Sequential_Environment import NonStationarySequentialEnvironment

np.random.seed(41148)
random.seed(41148)

# time horizon
T = 40
N_PHASES = 2
PHASE_LENGTH = int(T / N_PHASES)


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
                    [50, 40, 30, 25]])/ COST2

# conversion of each class for each price candidate of product 1
conv_1 = np.array([[[0.45, 0.6, 0.57, 0.52, 0.37, 0.15, 0.08],
                    [0.5, 0.55, 0.51, 0.47, 0.42, 0.35, 0.21],
                    [0.45, 0.42, 0.35, 0.27, 0.14, 0.1, 0.05],
                    [0.65, 0.7, 0.67, 0.55, 0.3, 0.21, 0.11]],

                   [[0.45, 0.6, 0.57, 0.52, 0.37, 0.15, 0.08],
                    [0.5, 0.55, 0.51, 0.47, 0.42, 0.35, 0.21],
                    [0.45, 0.42, 0.35, 0.27, 0.14, 0.1, 0.05],
                    [0.65, 0.7, 0.67, 0.55, 0.3, 0.21, 0.11]]])
conv_1[0] = np.clip(conv_1[1,0] - 0.2, 0.05, 1.)
conv_1[1] = np.clip(conv_1[1,1] - 0.2, 0.05, 1.)

# conversion of each class for each price candidate (axis 0) and promo (axis 1) of product 2
conv_2 = np.array([[[[0.57, 0.6, 0.67, 0.69],
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
                     [0.15, 0.21, 0.26, 0.35]]],

                   [[[0.57, 0.6, 0.67, 0.69],
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
                     [0.15, 0.21, 0.26, 0.35]]]])
conv_2[1,2] = np.clip(conv_2[1,2] - 0.2, 0.05, 1.)
conv_2[1,3] = np.clip(conv_2[1,3] - 0.2, 0.05, 1.)

# two alternative settings for number of promos of each class
N_PROMOS = 4
promo_setting_1 = np.array([0.3, 0.15, 0.25])
promo_setting_2 = np.array([0.25, 0.35, 0.2])

promo_setting = promo_setting_2

promo_assignment = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0]])

col_promo = [0, 0, 0, 0, 1, 2, 3]


def random_positive_choice(iterable):
    """random choice of index from non-zero elements of iterable"""
    iterable = np.array(iterable)
    indices = np.argwhere(iterable > 0).reshape(-1)
    index = np.random.choice(indices)

    return index


def expected_value_of_reward(pulled_arm1, pulled_arm2, current_customer_class, curr_promo, curr_phase):
    """calculate and return expected value of reward for arm choices based on conversion rates"""

    # we need to take into account the proportion of promo codes available impose by the promo setting
    # in the expected conversion rate and margin (promo 0 is no promo)
    if curr_promo != 0:
        conv_promo = promo_setting[curr_promo - 1] * conv_2[curr_phase, current_customer_class, pulled_arm2, curr_promo] \
                     + (1 - promo_setting[curr_promo - 1]) * conv_2[curr_phase, current_customer_class, pulled_arm2, 0]
        margin_promo = promo_setting[curr_promo - 1] * margin2[pulled_arm2, curr_promo] \
                       + (1 - promo_setting[curr_promo - 1]) * margin2[pulled_arm2, 0]
    else:
        conv_promo = conv_2[curr_phase, current_customer_class, pulled_arm2, curr_promo]
        margin_promo = margin2[pulled_arm2, curr_promo]

    reward = margin1[pulled_arm1] * conv_1[curr_phase, current_customer_class, pulled_arm1] * COST1
    if pulled_arm2 != -1:
        reward += margin_promo * conv_1[curr_phase, current_customer_class, pulled_arm1] \
                  * conv_promo * COST2
    return reward


# Task 7
def main():
    # ENVIRONMENT DEFINITION
    environment = NonStationarySequentialEnvironment(margin1=margin1, margin2=margin2, conv_rate1=conv_1,
                                                     conv_rate2=conv_2, horizon=T, n_phases=N_PHASES)

    # initialize variable for average of daily customers per day to calculate number of promos
    empirical_customer_amount = 500
    window_size = np.sqrt(T)*empirical_customer_amount

    # LEARNER DEFINITION
    learner1 = [SW_UCB(N_PRICES, window_size) for n in range(N_CLASSES)]
    extra_promos = N_CLASSES - 1  # we create additional copies of p0 as a hack for the linear sum assignment
    all_promos = N_PROMOS + extra_promos
    learner2 = SW_Matching_UCB(all_promos * N_CLASSES * N_PRICES, N_CLASSES, all_promos * N_PRICES,
                               col_promo * N_PRICES, window_size)

    arm_pull_count_1 = np.zeros((N_PHASES, N_CLASSES, N_PRICES))
    arm_pull_count_m = np.zeros((N_PHASES, N_CLASSES, N_PROMOS))
    arm_pull_count_2 = np.zeros((N_PHASES, N_CLASSES, N_PRICES))

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
        print('\r', "Progress: {}/{} days".format(i, T), end=" ") if i % 10 == 0 else False
        # sample number of customer for each class and truncate at 0 to avoid negative
        round_class_num = np.random.normal(n_class, 10)
        round_class_num = [int(n) if n >= 0 else 0 for n in round_class_num]
        daily_customer_amount = sum(round_class_num)

        daily_promos = [n * empirical_customer_amount for n in promo_setting]

        current_phase = min(int(i / PHASE_LENGTH), N_PHASES - 1)
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
            arm1 = learner1[customer_class].pull_arm()
            reward1 = environment.sub_round_1(customer_class, arm1)  # The second parameter is 0 due to fixed prices
            arm_pull_count_1[current_phase, customer_class, arm1] += 1

            # pull price 2 arm if positive reward and update else reward2 = 0
            if reward1 > 0:

                # costs = [learner.pull_arms_2(daily_promos)[2] for learner in learner2]

                row_ind, col_ind = learner2.pull_arms(daily_promos)
                chosen_promo = col_ind[customer_class] % N_PRICES - extra_promos
                chosen_price_2 = col_ind[customer_class] // all_promos
                if chosen_promo < 0:
                    chosen_promo = 0
                if chosen_promo > 0:
                    daily_promos[chosen_promo - 1] -= 1  # daily_promos [#p1 #p2 #p3], chosen_promo [p0 p1 p2 p3]

                # arm3 = learner3.pull_arm()
                reward2 = environment.sub_round_2(customer_class, chosen_price_2,
                                                  chosen_promo)
                # The second parameter is 0 due to fixed prices.  chosen_promo+1 since [p0 p1 p2 p3]
                # learner3.update(arm3, reward1 + reward2)
                arm2 = customer_class * all_promos * N_PRICES + chosen_price_2 * all_promos + chosen_promo

                arm_pull_count_m[current_phase, customer_class, chosen_promo] += 1
                arm_pull_count_2[current_phase, customer_class, chosen_price_2] += 1

                if chosen_promo == 0:
                    #  Update all arms that correspond to P0 for a given customer_class
                    for promo in range(extra_promos + 1):
                        temp_arm2 = customer_class * all_promos * N_PRICES + chosen_price_2 * all_promos + promo
                        learner2.update_one(temp_arm2, reward2)
                else:
                    learner2.update_one(arm2, reward2)
            else:
                reward2 = 0
                arm2 = -1
                chosen_price_2 = -1

            # update learner 1
            learner1[customer_class].update(arm1, reward1 + reward2)

            arms1.append(arm1)
            arms2.append(arm2)
            # arms3.append(arm3)

            # add rewards to cumulative sums of round rewards and calculate expected rewards
            round_reward1 += reward1 * COST1
            round_reward2 += reward2 * COST2
            round_expected_reward += expected_value_of_reward(arm1, chosen_price_2, customer_class, chosen_promo,
                                                              current_phase)  # first and
            # second parameters are 0 due to fixed prices
            round_clairvoyant_expected += np.max([[[expected_value_of_reward(i, j, customer_class, p, current_phase)
                                                    for i in range(N_PRICES)]
                                                   for j in range(N_PRICES)]
                                                  for p in range(N_PROMOS)])

            for l in learner1:
                l.t += 1
            learner2.t += 1

        # append round rewards to lists of rewards
        rewards1.append(round_reward1)
        rewards2.append(round_reward2)
        expected_rewards.append(round_expected_reward)
        clairvoyant_expected_rewards.append(round_clairvoyant_expected)

        # update empirical number of customers per day
        empirical_customer_amount = (empirical_customer_amount * i + daily_customer_amount) / (i + 1)
        window_size = np.sqrt(T) * empirical_customer_amount
        for l in learner1:
            l.window_size = window_size
        learner2.window_size = window_size

        environment.t += 1

    rewards1 = np.array(rewards1)
    rewards2 = np.array(rewards2)
    expected_rewards = np.array(expected_rewards)
    clairvoyant_expected_rewards = np.array(clairvoyant_expected_rewards)
    rewards = rewards1 + rewards2

    print("\r", "Progress: {}/{} days".format(T, T))

    #
    # LEARNING RESULTS
    #
    print()
    print('LEARNING RESULTS RESULTS')
    print()
    for phase in range(N_PHASES):
        print("Phase {} results:".format(phase+1))
        for c in range(N_CLASSES):
            p1 = price1[np.argmax(arm_pull_count_1[phase][c][:])]
            p2 = price2[np.argmax(arm_pull_count_2[phase][c][:])]
            print("class {} learners converged to price {} for product 1 and price {} for 2 ".format(c+1, p1, p2))
        print("With the following shows the number of times a class was assigned each promo level:")
        print(arm_pull_count_m[phase][:][:])
    print()
    print(f'Total profit collected from product 1: {np.sum(rewards1)}')
    print(f'Total profit collected from product 2: {np.sum(rewards2)}')

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(np.cumsum(rewards1), label='Product 1')
    ax1.axvline(x=PHASE_LENGTH,c='r',linestyle='--', label="Phase Change")
    ax1.legend(loc='lower right')
    ax2.plot(np.cumsum(rewards2), label='Product 2')
    ax2.axvline(x=PHASE_LENGTH,c='r',linestyle='--', label="Phase Change")
    ax2.legend(loc='lower right')
    fig.suptitle('Cumulative Rewards from each product')
    plt.show()

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    plt.plot(moving_average(rewards, 10))
    plt.title('10-day moving average of rewards collected')
    plt.axvline(x=PHASE_LENGTH-10,c='r',linestyle='--', label="Phase Change")
    plt.legend(loc='lower right')
    plt.show()

    #
    # REGRETS
    #
    # regrets are calculated in terms of the expected value of the reward for each pulled arm

    print()
    print(f'Total expected regret: {np.sum(np.subtract(clairvoyant_expected_rewards, expected_rewards))}')

    plt.plot(moving_average(expected_rewards, 10), label='Expected rewards for Maching UCB')
    plt.plot(moving_average(clairvoyant_expected_rewards, 10), label='Expected rewards for Clairvoyant Algorithm',
             color='r')
    plt.axvline(x=PHASE_LENGTH-10,c='r',linestyle='--', label="Phase Change")
    plt.legend(loc='lower right')
    plt.title('10-day moving average of expected rewards of each algorithm')
    plt.ylim(500,max(clairvoyant_expected_rewards))
    plt.show()


if __name__ == '__main__':
    main()

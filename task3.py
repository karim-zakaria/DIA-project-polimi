import numpy as np
import matplotlib.pyplot as plt

from pricing.enviroment.Enviroment import Enviroment
from pricing.enviroment.Non_Stationary_Enviroment import Non_Stationary_Enviroment

from pricing.learners.thompson import SW_Thompson
from pricing.learners.ucb import SW_UCB
from pricing.learners.thompson import Thompson
from pricing.learners.ucb import UCB


np.random.seed(0)

# Task 3
def main():
    #
    # FIXED PROBLEM PARAMETERS
    #
    T = 365
    n_classes = 4
    #arms apply to price 1
    n_arms = 4
    #since promos only apply to price 2, we consider that we only have one promo (no promo)
    n_promos = 1
    #Uncomment for non stationary
    # n_phases = 4
    
    #
    # MARGINS
    #
    margin1 = np.array([1.0, 0.95, 0.90, 0.85])/1000
    margin2 = np.array([.5])/1000
    
    #
    # NUMBER OF COSTUMERS
    #
    #Comment for non stationary
    #Number of customers of class i
    n_customers = (100*np.random.rand(n_classes)).astype(int)
    #Uncomment for non stationary
    # #Number of customers of class i at phase j
    # n_customers = (100*np.random.rand(n_classes, n_phases)).astype(int)

    #
    # CONVERSION RATE PRODUCT 1
    #
    #Comment for non stationary
    #Convertion rate of the first item, from class i at price j.
    conv_rate1 = np.random.rand(n_classes, n_arms)
    #Uncomment for non stationary
    # #Convertion rate of the first item, from class i at price j and phase k.
    # conv_rate1 = np.random.rand(n_classes, n_arms, n_phases)

    #
    # CONVERSION RATE PRODUCT 2
    #
    #Comment for non stationary
    #Convertion rate of the second item, from class i at original price j, discount from promo k
    conv_rate2 = np.random.rand(n_classes, n_arms, n_promos)
    #Uncomment for non stationary
    # #Convertion rate of the second item, from class i at original price j, discount from promo k and phase l
    # conv_rate2 = np.random.rand(n_classes, n_arms, n_promos, n_phases)
    # FOR TASK 3
    #we need to make it so conversion rates for price 2 for each of the arms are the same
    # for i in range(1,n_arms):
    #     conv_rate2[:,i,:] = conv_rate2[:,0,:]

    #
    # PROMOTIONS FOR PRODUCT 2
    #
    promo_assig = np.random.rand(n_classes, n_promos)
    promo_assig = promo_assig / promo_assig.sum(axis=1)[:, np.newaxis]

    #
    # ENVIRONMENT DEFINITION
    #
    #Comment for non stationary
    st_env1 = Enviroment(n_arms, n_customers, margin1, margin2, conv_rate1, conv_rate2, promo_assig)
    st_env2 = Enviroment(n_arms, n_customers, margin1, margin2, conv_rate1, conv_rate2, promo_assig)
    #Uncomment for non stationary
    # nst_env1 = Non_Stationary_Enviroment(n_arms, n_customers, margin1, margin2, conv_rate1, conv_rate2, promo_assig, T, n_phases)
    # nst_env2 = Non_Stationary_Enviroment(n_arms, n_customers, margin1, margin2, conv_rate1, conv_rate2, promo_assig, T, n_phases)

    #
    # LEARNER DEFINITION
    #
    #Comment for non stationary
    learner1 = Thompson(n_arms)
    learner2 = UCB(n_arms)
    #Uncomment for non stationary
    # #window size of around 20 samples
    # learner1 = SW_Thompson(n_arms, 2*np.sqrt(365).astype(int))
    # learner2 = SW_UCB(n_arms, 2*np.sqrt(365).astype(int))

    #
    # START LEARNING PROCESS
    #
    rewards1 = []
    rewards2 = []
    for i in range(T):
        arm1 = learner1.pull_arm()
        arm2 = learner2.pull_arm()

        #Comment for non stationary
        reward1 = st_env1.round(arm1)
        reward2 = st_env2.round(arm2)

        #Uncomment for non stationary
        # reward1 = nst_env1.round(arm1)
        # reward2 = nst_env2.round(arm2)

        learner1.update(arm1,reward1)
        learner2.update(arm2,reward2)

        rewards1.append(reward1)
        rewards2.append(reward2)

    #
    # LEARNING RESULTS RESULTS
    #
    print(f'Total margin collected by UCB: {np.sum(rewards2)}')
    print(f'Total margin collected by Thompson Sampling: {np.sum(rewards1)}')

    plt.plot(np.cumsum(rewards1), label = 'Thomson Sampling')
    plt.plot(np.cumsum(rewards2), label = 'UCB1')
    plt.legend(loc = 'lower right')
    plt.title('Cumulative Rewards collected by both learners')
    plt.show()

    def moving_average(x,w):
        return np.convolve(x, np.ones(w), 'valid') / w
    plt.plot(moving_average(rewards1,7), label = 'Thomson Sampling')
    plt.plot(moving_average(rewards2,7), label = 'UCB1')
    plt.legend(loc = 'upper right')
    plt.title('Weekly moving average of both learners')
    plt.show()

    #regrets

if __name__ == '__main__':
    main()
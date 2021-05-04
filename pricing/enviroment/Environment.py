import numpy as np

if __name__ == '__main__':
    np.random.seed(0)

"""
This enviroment assumes:
      - Price of the second item fixed
      - Convertion rates known and fixed in time
      - Assignment of promos fixed
      - Numbers of customers per day of each class known and fixed in time
"""


class Enviroment():
    def __init__(self, n_arms, n_customers, margin1, margin2, conv_rate1, conv_rate2, promo_assig):
        self.n_arms = n_arms
        self.n_customers = n_customers
        self.margin1 = margin1
        self.margin2 = margin2
        self.conv_rate1 = conv_rate1
        self.conv_rate2 = conv_rate2
        self.promo_assig = promo_assig
        self.n_promos = len(margin2)

    def round(self, pulled_arm):
        reward = 0
        for cust_class in range(len(self.n_customers)):
            buyers = np.random.binomial(self.n_customers[cust_class], self.conv_rate1[cust_class, pulled_arm])
            # divide number of buyers by number of custumers to work with fractions and keep rewards bounded in [0,1]
            reward += self.margin1[pulled_arm] * buyers / self.n_customers[cust_class]

            for promo in range(self.n_promos):
                buyers2 = np.random.binomial(buyers * self.promo_assig[cust_class, promo],
                                             self.conv_rate2[cust_class, pulled_arm, promo])
                # divide buyers2 by maximum number of buyers to keep rewards bounded
                reward += self.margin2[promo] * buyers2 / (buyers * self.promo_assig[cust_class, promo])

        return reward


def main():
    # Each arm is a different price for the first item
    T = 365
    n_arms = 3
    n_arms = int(np.ceil((np.log10(T) * T) ** 0.25))

    # Number of classes of customers
    n_classes = 4
    n_promos = 5

    # Margin of product 1 at different prices
    # Vector of positive floats
    margin1 = 50 * np.random.rand(n_arms)

    # Margin of product 2 at different promos
    # Vector of positive floats
    margin2 = 50 * np.random.rand(n_promos)

    # Number of customers of each class
    # Vector of integers
    n_customers = (100 * np.random.rand(n_classes)).astype(int)

    # Convertion rate of the first item, from class i at price j.
    # Matrix of probabilities
    conv_rate1 = np.random.rand(n_classes, n_arms)

    # Convertion rate of the second item, from class i at original price j and discount from promo k
    # Matrix of probabilities
    conv_rate2 = np.random.rand(n_classes, n_arms, n_promos)

    # Percentage of people from class i that get the promotion j
    promo_assig = np.random.rand(n_classes, n_promos)

    # Normalization so the sum of each row is equal to 1
    # Matrix of probabilities
    promo_assig = promo_assig / promo_assig.sum(axis=1)[:, np.newaxis]

    env = Enviroment(n_arms, n_customers, margin1, margin2, conv_rate1, conv_rate2, promo_assig)

    env.round(0), env.round(1), env.round(2)


if __name__ == '__main__':
    main()

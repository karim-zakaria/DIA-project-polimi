import numpy as np

if __name__ == '__main__':
    np.random.seed(0)
    from Enviroment import Enviroment
else:
    from pricing.enviroment.Enviroment import Enviroment

"""
This enviroment assumes:
      - Price of the second item fixed
      - Convertion rates known and fixed in time
      - Assignment of promos fixed
      - Numbers of customers per day of each class Gaussian and fixed in time
"""

class Gaussian_Enviroment(Enviroment):
  def __init__(self, n_arms, n_customers, margin1, margin2, conv_rate1, conv_rate2, promo_assig, cust_var):
    super().__init__(n_arms, n_customers, margin1, margin2, conv_rate1, conv_rate2, promo_assig)
    self.cust_var = cust_var

  def round(self, pulled_arm):
    reward = 0
    for cust_class in range(len(self.n_customers)):
      customers = (np.random.randn()*np.sqrt(self.cust_var[cust_class]) + self.n_customers[cust_class]).astype(int)
      if customers < 0:
        customers=0
      buyers = np.random.binomial(customers, self.conv_rate1[cust_class, pulled_arm])
      reward += self.margin1[pulled_arm]*buyers

      for promo in range(self.n_promos):
        buyers2 = np.random.binomial(buyers*self.promo_assig[cust_class, promo], self.conv_rate2[cust_class, pulled_arm, promo])
        reward += self.margin2[promo]*buyers2
        
    return reward
    
def main():
    # Each arm is a different price for the first item
    T = 365
    n_arms = 3
    n_arms = int(np.ceil((np.log10(T)*T)**0.25))

    #Number of classes of customers
    n_classes = 4
    n_promos = 5

    # Margin of product 1 at different prices
    # Vector of positive floats
    margin1 = 50*np.random.rand(n_arms)

    # Margin of product 2 at different promos
    # Vector of positive floats
    margin2 = 50*np.random.rand(n_promos)

    # Number of customers of each class
    # Vector of integers
    n_customers = (100*np.random.rand(n_classes)).astype(int)
    
    # Variance of the number of customers of each class
    # Vector of floats
    cust_var = 5*np.random.rand(n_classes)

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

    g_env = Gaussian_Enviroment(n_arms, n_customers, margin1, margin2, conv_rate1, conv_rate2, promo_assig, cust_var)

    g_env.round(0), g_env.round(1), g_env.round(2)

if __name__ == '__main__':
    main()

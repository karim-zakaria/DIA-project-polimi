import numpy as np

if __name__ == '__main__':
    np.random.seed(0)
    from Enviroment import Enviroment
else:
    from pricing.enviroment.Enviroment import Enviroment

"""
This enviroment assumes:
      - Price of the second item fixed
      - Convertion rates known and change with time
      - Assignment of promos fixed
      - Numbers of customers per day of each class known and change with time
"""

class Non_Stationary_Enviroment(Enviroment):
  def __init__(self, n_arms, n_customers, margin1, margin2, conv_rate1, conv_rate2, promo_assig, horizon, n_phases):
    super().__init__(n_arms, n_customers, margin1, margin2, conv_rate1, conv_rate2, promo_assig)
    self.t = 0
    self.n_phases = n_phases
    self.phase_size = horizon/n_phases

  def round(self, pulled_arm):
    current_phase = int(self.t / self.phase_size)
    reward = 0
    for cust_class in range(len(self.n_customers)):
      buyers = np.random.binomial(self.n_customers[cust_class, current_phase], self.conv_rate1[cust_class, pulled_arm, current_phase])
      reward += buyers*self.margin1[pulled_arm]

      for promo in range(self.n_promos):
        buyers2 = np.random.binomial(buyers*self.promo_assig[cust_class, promo], self.conv_rate2[cust_class, pulled_arm, promo, current_phase])
        reward += self.margin2[promo]*buyers2
        
    self.t += 1
    return reward
  

def main():
    T = 365
    n_arms = 3
    n_arms = int(np.ceil((np.log10(T)*T)**0.25))
    n_classes = 4
    n_promos = 5
    n_phases = 6

    margin1 = 50*np.random.rand(n_arms)
    margin2 = 50*np.random.rand(n_promos)

    #Number of customers of class i at phase j
    n_customers = (100*np.random.rand(n_classes, n_phases)).astype(int)

    #Convertion rate of the first item, from class i at price j and phase k.
    conv_rate1 = np.random.rand(n_classes, n_arms, n_phases)

    #Convertion rate of the second item, from class i at original price j, discount from promo k and phase l
    conv_rate2 = np.random.rand(n_classes, n_arms, n_promos, n_phases)

    promo_assig = np.random.rand(n_classes, n_promos)
    promo_assig = promo_assig / promo_assig.sum(axis=1)[:, np.newaxis]

    nst_env = Non_Stationary_Enviroment(n_arms, n_customers, margin1, margin2, conv_rate1, conv_rate2, promo_assig, T, n_phases)

    for i in range(10):
      print(nst_env.round(0))
      print(nst_env.t)

if __name__ == '__main__':
    main()


from Learner import Learner
import numpy as np
from scipy.optimize import linear_sum_assignment
from matching.cumsum import CUMSUM

class UCB(Learner):
  def __init__(self, n_arms)
    super().__init__(n_arms)
    self.empirical_means=np.zeros(n_arms)
    self.confidence=np.array([np.inf]*n_arms)

  def pull_arm(self):
    upper_conf=self.empirical_means+self.confidence
    return np.random.choice(np.where(upper_conf==upper_conf.max())[0])

  def update(self, pull_arm, reward):
      self.t+=1
      self.empirical_means[pull_arm]=(self.empirical_means[pull_arm]*(self.t-1)+reward)/self.t
      for a in range(self.n_arms):
        n_samples=len(self.rewards_per_arm[a])
        self.confidence[a]=(2*np.log(self.t)/n_samples)**0.5 if n_samples>0 else np.inf
      self.update_observations(pull_arm, reward)


class Matching_UCB(UCB):
  def __init__(self,n_arms,n_rows,n_cols):
    super().__init__(n_arms)
    self.n_rows=n_rows
    self.n_cols=n_cols
    assert n_arms==n_cols*n_rows

  def pull_arms(self):
    upper_conf=self.empirical_means+self.confidence
    upper_conf[np.isinf(upper_conf)]=1e3
    row_ind, col_ind=linear_sum_assignment(-upper_conf.reshape(self.n_rows, self.n_cols))
    return (row_ind, col_ind)

  def update(self, pulled_arms, rewards):
    self.t+=1
    pulled_arms_flat= np.ravel_multi_index(pulled_arms,(self.n_rows, self.n_cols))
    
    for pulled_arm, reward in zip(pulled_arms_flat, rewards):
      self.update_observations(pulled_arm, reward)
      self.empirical_means[pulled_arm]=(self.empirical_means[pulled_arm]*(self.t-1)+reward)/self.t

    for a in range(self.n_arms):
      n_samples=len(self.rewards_per_arm[a])
      self.confidence[a]=(2*np.log(self.t)/n_samples)**0.5 if n_samples>0 else np.inf


class CUMSUM_Matching_UCB(Matching_UCB):
  def __init__(self, n_arms, n_rows, n_cols, M=100, eps=0.05, h=20, alpha=0.01):
    super().__init__(n_arms, n_rows, n_cols)
    self.change_detection=[CUMSUM(M, eps, h) for _ in range(n_arms)]
    self.valid_rewards_per_arms=[[] for _ in range(n_arms)]
    self.detections= [[] for _ in range(n_arms)]
    self.alpha=alpha

  def pull_arm(self):
    if np.random.binomial(1,1-self.alpha):
      upper_conf=self.empirical_means+self.condfidence
      upper_conf[np.isinf(upper_conf)]=1e3
      row_ind, col_ind= linear_sum_assignment(-upper_conf.reshape(self.n_rows, self.n_cols))
      return row_ind, col_ind
    else:
      costs_random= np.random.randint(0, 10, size-(self.n_rows, self.n_cols))
      return linear_sum_assignment(costs_random)

  def update(self, pulled_arms, rewards):
    self.t+=1
    pulled_arms_flat= np.ravel_multi_index(pulled_arms,(self.n_rows, self.n_cols))

    for pulled_arm, reward in zip(pulled_arms_flat, rewards):
      if self.change_detection[pulled_arm].update(reward):
        self.detections[pulled_arm].append(self.t)
        self.valid_rewards_per_arms[pulled_arm]=[]
        self.change_detection[pulled_arm].reset()
      self.update_observations(pulled_arm, reward)
      self.empirical_means[pulled_arm]=np.mean(self.valid_rewards_per_arms[pulled_arm])
    total_valid_samples=sum([len(x) for x in self.valid_rewards_per_arm])

    for a in range(self.n_arms):
      n_samples=len(self.valid_rewards_per_arm[a])
      self.confidence[a]=(2*np.log(total_valid_samples)/n_samples)**0.5 if n_samples>0 else np.inf

  def update_observations(self, pulled_arm, reward):
    self.rewards_per_arm[pulled_arm].append(reward)
    self.valid_rewards_per_arm[pulled_arm].append(reward)
    self.collected_rewards=np.append(self.collected_rewards, reward)

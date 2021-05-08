# -*- coding: utf-8 -*-
"""task0.ipynb

Brute force solution of pricing of product 1 and pricing+promo matching of product 2
"""

from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd

# number of customers of each class
N_CLASSES = 4
n_class=np.array([200,150,50,100])

# price candidates for each product
N_PRICES=7
price_1 = np.array([600, 700, 800, 900, 1000, 1100, 1200])

price_2 = np.array([70, 75, 80, 85, 90, 95, 100])

# profit margin for each price candidate for product 1
margin_1 = np.array([50, 150, 250, 350, 450, 550, 650])

# profit margin for price candidate (axis 0) and promo (axis 1)
margin_2 = np.array([[20, 13, 6, 2.5],
                    [25, 17.5, 10, 6.25],
                    [30, 22, 14, 10],
                    [35, 26.5, 18, 13.75],
                    [40, 31, 22, 17.5],
                    [45, 35.5, 26, 21.25],
                    [50, 40, 30, 25]])

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
promo_dist_1 = np.array([150, 150, 75, 125])
promo_dist_2 = np.array([100, 125, 175, 100])
promo_dist=promo_dist_1

PROMO_BATCH_SIZE = 25 # size of promo batches that can be assigned to one of the classes

# generate two axis of the matching cost matrix with classes on axis 0 and promos on axis 1
matching_axis_0 = np.array([])
matching_axis_1 = np.array([])

for c in range(N_CLASSES):
  for _ in range(int(n_class[c]/PROMO_BATCH_SIZE)):
    matching_axis_0 = np.append(matching_axis_0, c)

for p in range(len(promo_dist)):
  for _ in range(int(promo_dist[p]/PROMO_BATCH_SIZE)):
    matching_axis_1 = np.append(matching_axis_1, p)

# compute optimal product 1 candidate for each class based on revenue per unit
product_1_revenue = [np.multiply(conv,margin_1) for conv in conv_1]
best_product_1_candidate = np.argmax(product_1_revenue, axis=1) #best candidate index for each class
best_product_1_price = price_1[best_product_1_candidate] #best price for each class
print(best_product_1_price)


# compute best product 2 candidate for each class and best promo assignment
results={"class 1 price":[],
         "class 2 price":[],
         "class 3 price":[],
         "class 4 price":[],
         "promo assignment":[],
         "reward":[]} 
results=pd.DataFrame.from_dict(results) # dataframe to hold each setting and corresponding reward

# brute force loop over all possible price candidate for each class
for pc1 in range(N_PRICES):
  for pc2 in range(N_PRICES):
    for pc3 in range(N_PRICES):
      for pc4 in range(N_PRICES):
        cost_matrix = np.zeros((len(matching_axis_0), len(matching_axis_1)))
        for p in range(len(matching_axis_1)):
          promo_batch = int(matching_axis_1[p])
          cost_matrix[matching_axis_0==0, p] = conv_1[0, best_product_1_candidate[0]]*conv_2[0, pc1, promo_batch]*margin_2[pc1, promo_batch]*n_class[0]
          cost_matrix[matching_axis_0==1, p] = conv_1[1, best_product_1_candidate[1]]*conv_2[1, pc2, promo_batch]*margin_2[pc2, promo_batch]*n_class[1]
          cost_matrix[matching_axis_0==2, p] = conv_1[2, best_product_1_candidate[2]]*conv_2[2, pc3, promo_batch]*margin_2[pc3, promo_batch]*n_class[2]
          cost_matrix[matching_axis_0==3, p] = conv_1[3, best_product_1_candidate[3]]*conv_2[3, pc4, promo_batch]*margin_2[pc4, promo_batch]*n_class[3]

        row_ind, col_ind = linear_sum_assignment(-cost_matrix) # solve for optimal assignment given price setting
        reward = cost_matrix[row_ind, col_ind].sum()
        assignment = np.zeros_like(cost_matrix)
        assignment[row_ind,col_ind] = 1
        results=results.append({"class 1 price":price_2[pc1],
                                "class 2 price":price_2[pc2],
                                "class 3 price":price_2[pc3],
                                "class 4 price":price_2[pc4],
                                "promo assignment":assignment,
                                "reward":reward},ignore_index=True) # append setting and rward to dataframe

results=results.sort_values(by=["reward"], ascending=False) #sort by reward
print(results.iloc[0]) #print optimal setting
print(results.iloc[0]['promo assignment']) #print optimal promo assignment
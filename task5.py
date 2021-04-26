import numpy as np
import pandas as pd
from hungarian_algorithm import algorithm
from scipy.optimize import linear_sum_assignment
from munkres import Munkres

from matching.hungarian_algorithm import solve

n_classes = 4
n_promos = 4
fraction = 0.6

customers = [250, 250, 250, 250]
promotion_size = sum(customers) * fraction

margin2 = [20, 13, 6, 2.5]
conv_rate2 = [[0.57, 0.6, 0.67, 0.69], [0.38, 0.4, 0.49, 0.53],
              [0.17, 0.24, 0.37, 0.39], [0.49, 0.53, 0.65, 0.68]]

m = np.zeros((n_classes, n_promos))

for x in range(n_classes):
    for y in range(n_promos):
        m[x][y] = margin2[y] * conv_rate2[x][y]

# row_ind, col_ind = linear_sum_assignment(m)
# print(col_ind)
# print("------------------------")

# m = m.astype(int)
# print(m)
# matching = solve(m)
# print(matching)

cost_matrix = []
max_val = m.max()
for row in m:
    cost_row = []
    for col in row:
        cost_row += [max_val - col]
    cost_matrix += [cost_row]

print(m)
hungarian = Munkres()
indexes = hungarian.compute(cost_matrix)
indexes = [x[1] for x in indexes]

# Promotions found by the algorithm
print(indexes)


# Since the number of promotions are limited, they are assigned to the customer with respect to the reward in
# descending order
reward_class_i = []
class_i = []
for i in range(0, len(indexes)):
    reward_class_i.append(m[i, indexes[i]])
    class_i.append(i)

table = list(zip(class_i, indexes, reward_class_i, customers))
table = sorted(table, reverse=True, key=lambda x:x[2])

promotions = promotion_size
promo_assigned = []

# If the promotion for a class is set to Promotion_0, meaning no promotion applied, the promotion assigned
# for that class is then set to zero.
for x in table:
    if not x[1] == 0:
        promo_assigned.append(min(x[3], int(promotions)))
        promotions -= min(x[3], int(promotions))
    else:
        promo_assigned.append(0)

# A data frame presenting data for each class.
df = pd.DataFrame(table, columns=['Class', 'Promotion', 'Reward', 'Customers in class'])
df['Assigned promotions'] = promo_assigned
df = df.set_index('Class')
df = df.sort_values('Class', ascending=True)
print(df)

# List of the amount of promotions assigned to each class
nr_promo_class_i=df['Assigned promotions'].tolist()
print(nr_promo_class_i)

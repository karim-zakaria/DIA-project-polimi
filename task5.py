import numpy as np
from hungarian_algorithm import algorithm
from scipy.optimize import linear_sum_assignment
from munkres import Munkres

from matching.hungarian_algorithm import solve

n_classes = 4
n_promos = 4
fraction = 0.6

customers = [250, 250, 250, 250]
promotion_size = sum(customers) * fraction

margin2 = 50 * np.random.rand(n_promos)  # Margin of product 2 at different promos
conv_rate2 = np.random.rand(n_classes, n_promos)  # Conversion rate of the second item, from class i at promo k

m = np.zeros((n_classes, n_promos))

for x in range(n_classes):
    for y in range(n_promos):
        m[x][y] = margin2[y] * conv_rate2[x, y]

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

cost_matrix = np.matrix(cost_matrix)

hungarian = Munkres()
indexes = hungarian.compute(m)
indexes = [x[1] for x in indexes]

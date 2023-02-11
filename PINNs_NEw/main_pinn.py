import argparse
import numpy as np
import os
import torch
import random
from utils import *
import matplotlib.pyplot as plt
from get_data import *


# x = np.linspace(0, 2.8, 280, endpoint=True).reshape(-1,1) # not inclusive
# t = np.linspace(0, 5, 300, endpoint=True).reshape(-1,1)

# X, T = np.meshgrid(x, t) # all the X grid points T times, all the T grid points X times
# X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None])) #all the x,t points not inclusive

# #remove initial and boundary data from X_star
# t_no_intial = t[1:]

# #remove boundary at x = 0
# x_no_boundary = x[1:]

# X_NB, T_NI = np.meshgrid(x_no_boundary, t_no_intial)
# X_star_NB = np.hstack((X_NB.flatten()[:, None], T_NI.flatten()[:, None]))


# # Sample collocation points only from the interior (Where the PDE is enforce)
# # Number of collocation points
N_c = 10000
# X_f_train = sample_random(X_star_NB, N_c)

test_v, t = fetch_test(1)
print(test_v)
print(t)



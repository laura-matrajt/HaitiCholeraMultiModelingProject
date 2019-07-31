from __future__ import division
import numpy as np
import time
import imp
import pickle
# import pymc as pc
import sys, getopt
import csv
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from scipy import optimize
from choleraEqs import choleraEqs, choleraEqs2, choleraEqs3, choleraEqs2bis, choleraEqs5WithoutVaccinationSingleGroup
from functools import partial
# import seaborn as sns
# from pyDOE import *
import random
import os

def formTravelMat1(numDep, rlen, tau, vrate, Pop):
    """
    Form the travel matrix following a gravity model

    """
    TravelMat = np.zeros((numDep, numDep))
    totalTravel = 0
    for i in range(numDep):
        for j in range(numDep):
            if i != j:
                TravelMat[i, j] = vrate*Pop[i]*Pop[j]/(rlen[i, j]**tau)
    return TravelMat







# def lhs_params(low_bound, upper_bound, n_samples):
#     """
#     computes n_samples of parameter lists using latin hypercube sampling.
#     :param low_bound: a vector containing all the lower bounds of each of the sides of the hypercube
#     :param upper_bound: a vector containing all the upper bounds of each of the sides of the hypercube
#     :param n_samples: number of samples
#     :return:
#     """
#     #check that the length of each vector is correct
#     if len(low_bound) != len(upper_bound):
#         print("low_bound vector has different size than upper_bound vector")
#         # break
#
#     mylhs = lhs(len(low_bound), samples=n_samples)
#     # print mylhs
#     mysamples = np.zeros([n_samples, len(low_bound)])
#     #convert the samples to the adequate range:
#     for ivals in range(len(low_bound)):
#         mysamples[:, ivals] = (upper_bound[ivals] - low_bound[ivals])*mylhs[:, ivals] + low_bound[ivals]
#
#     # print mysamples
#     return mysamples





#in this file we will generate a matrix with parameters to be drawn for the sensitivity analysis.

from __future__ import division
import numpy as np
import time
import imp
import itertools
import pickle
# import pymc as pc
import sys, getopt
import csv
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from scipy import optimize
from choleraEqs import choleraEqs8WithVaccinationNetwork, choleraEqs10WithoutVaccinationNetwork
from functools import partial
# from runCholeraModel import formTravelMat
# from runCholeraModel2 import formTravelMatJon
import seaborn as sns
# from pyDOE import *
from functionsCholeraProject import *
import random
import os



today = time.strftime("%d%b%Y", time.localtime())

#Select the mean of all the parameter sets that were fitted to the data, after some cleaning of the matrix:

#open the matrix with all parameter sets, the last column is the MSE value for that parameter set
myfilename = 'leastSquaresDataUpTo2019/fitEpidemicPiece_resultsMatrix_18Mar2019.pickle'
f =  open(myfilename, 'r')
results =(pickle.load(f))
f.close()


#clean up the matrix
mymat = results[~np.all(results == 0, axis=1)]
print np.shape(mymat)
uniqueMat = np.unique(mymat, axis=0)
n = len(uniqueMat)
sortedResults = uniqueMat[uniqueMat[:,-1].argsort()]


#select the 95% bounds
highbound = int(n-n*0.025)
lowbound = int(n*0.025)
dim1 = highbound-lowbound


#take the mean, this is what we call the "best parameter set"
[betaMean, betaWMean, muMean] = np.mean(sortedResults[lowbound:highbound,0:-1], 0)

print [betaMean, betaWMean, muMean]

#the second set of parameters just repeats the first two (beta and beta_W, adds the best fit for mu and the frac susceptible
#which we set to 0.75
params2 = [9.92853814e-07, 4.03310744e-02, 2.58208148e+00, 0.75]

mybestParams = np.concatenate(([betaMean, betaWMean, muMean],params2))
print (mybestParams)
mybestParams


# myparamsfileName = 'mainResultsMarch2019/bestParamsVector' + today + '.pickle'
# myfile = open(myparamsfileName, 'wb')
# pickle.dump(mybestParams, myfile)
# myfile.close()
# print myparamsfileName
#
myval = 0.25
myval2 = 0.1



####### the following lines are commented for reproducibility purposes, as running them will create a different set of
#parameters that the ones we used in the sensitivity analysis. Uncomment these lines to obtain a matrix with 1000 parameter
#samples around the best parameter set.



# lowerBounds = np.zeros(5)
# upperBounds = np.zeros(5)
#
# lowerBounds[0:3] = [betaMean - (myval) * betaMean, betaWMean - (myval) * betaWMean, muMean/(10**(myval*10))]
# lowerBounds[3] = params2[2]/(10*myval)
# lowerBounds[4] = params2[3] - (myval2) * params2[3]
# print 'lowerBounds', lowerBounds
#
# upperBounds[0:3] = [betaMean + (myval)*betaMean, betaWMean + (myval)*betaWMean, muMean*(10**(myval*10))]
# upperBounds[3] = params2[2]*(10*myval)
# upperBounds[4] = params2[3] + (myval2) * params2[3]
# print 'upper', upperBounds
#
# mysamples = lhs_params(lowerBounds, upperBounds, 1000)
#
# matrixParams = np.column_stack([mysamples[:, 0:3], mysamples[:, 0:2], mysamples[:, 3:5]])
# print np.shape(matrixParams)
# print matrixParams[1,:]
#
# #
# #store the matrix in a pickle file
# today = time.strftime("%d%b%Y", time.localtime())
#
# mytime = time.localtime()
# myseed = np.abs(int((np.random.normal()) * 1000000))
# np.random.seed(myseed)
# print myseed
# #
# results = [myseed, matrixParams]
#
# mymatfileName = 'mainResultsMarch2019/matrixOfParamsForSensitivityAnalysis_width' + str(myval) + '_fracSusWidth' + str(myval2) +\
#                   today + '.pickle'
# myfile = open(mymatfileName, 'wb')
# pickle.dump(results, myfile)
# myfile.close()
# print mymatfileName
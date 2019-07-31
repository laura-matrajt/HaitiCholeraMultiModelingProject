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

#this parameter set was obtained as the mean of 1000 best fits that can be retrieved from this file:
myfilename = 'leastSquaresDataUpTo2019/fitEpidemicPiece_resultsMatrix_18Mar2019.pickle'
f =  open(myfilename, 'r')
results =(pickle.load(f))
f.close()

mymat = results[~np.all(results == 0, axis=1)]
print np.shape(mymat)
uniqueMat = np.unique(mymat, axis=0)
n = len(uniqueMat)
sortedResults = uniqueMat[uniqueMat[:,-1].argsort()]
highbound = int(n-n*0.025)
lowbound = int(n*0.025)
dim1 = highbound-lowbound



[betaMean, betaWMean, muMean] = np.mean(sortedResults[lowbound:highbound,0:-1], 0)

print [betaMean, betaWMean, muMean]


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

lowerBounds = np.zeros(5)
upperBounds = np.zeros(5)

lowerBounds[0:3] = [betaMean - (myval) * betaMean, betaWMean - (myval) * betaWMean, muMean/(10**(myval*10))]
lowerBounds[3] = params2[2]/(10*myval)
lowerBounds[4] = params2[3] - (myval2) * params2[3]
print 'lowerBounds', lowerBounds

upperBounds[0:3] = [betaMean + (myval)*betaMean, betaWMean + (myval)*betaWMean, muMean*(10**(myval*10))]
upperBounds[3] = params2[2]*(10*myval)
upperBounds[4] = params2[3] + (myval2) * params2[3]
print 'upper', upperBounds

# mysamples = lhs_params(lowerBounds, upperBounds, 1000)
#
# matrixParams = np.column_stack([mysamples[:, 0:3], mysamples[:, 0:2], mysamples[:, 3:5]])
# print np.shape(matrixParams)
# print matrixParams[1,:]
# # for jvals in range(3):
# # # #     print mybestParams[jvals]
# #     lowerBounds[jvals] = mybestParams[jvals] - mywidth*mybestParams[jvals]
# #     upperBounds[jvals] = mybestParams[jvals] + mywidth*mybestParams[jvals]
# #
# # for jvals in range(4):
# #     print mybestParams[jvals]
# #     lowerBounds[jvals] = mybestParams[jvals]**(1.1)
# #     upperBounds[jvals] = mybestParams[jvals]**(0.9)
# #
# #
# # lowerBounds[4] = mybestParams[4] - mywidth2*mybestParams[4]
# # upperBounds[4] = mybestParams[4] + mywidth2*mybestParams[4]




# # print lowerBounds
# # print upperBounds
# # matrixParams = lhs_params(lowerBounds, upperBounds, 1000)
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

# mymatfileName = 'mainResultsMarch2019/matrixOfParamsForSensitivityAnalysis_width' + str(myval) + '_fracSusWidth' + str(myval2) +\
#                   today + '.pickle'
# myfile = open(mymatfileName, 'wb')
# pickle.dump(results, myfile)
# myfile.close()
# print mymatfileName
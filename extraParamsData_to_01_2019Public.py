from __future__ import division
import numpy as np
import pandas as pd


#reads the extra parameters to run the least squares optimization in the cluster


# import the population for each department as of 2015
totalPopTable = pd.read_csv("../cholera_data/populationHaiti2015.txt",  )
# totalPopTable = np.genfromtxt("../cholera_data/populationHaiti2015.txt", skiprows=1, )
# totalPop = totalPopTable[:,1]
# print totalPopTable
# print totalPopTable.iloc[:, 1]
totalPop = np.array(totalPopTable.iloc[:, 1])
print totalPop
# print np.sum(totalPop)


# import the time series
cases = np.loadtxt("../cholera_data/incidenceHaiti_to_01_19.txt", skiprows=1, delimiter=',')
print len(cases)
# cases = np.array(cases[:, 1:-1])
# print cases[:, :]

#import travel and water matrices:
# import the rlen matrix
rlen = np.genfromtxt('../cholera_data/rlen_i_j.csv', delimiter=',')
# print rlen

# import the water matrix
waterMat = np.genfromtxt('../cholera_data/watT_i_j.csv', delimiter=',')
#as of May 23, 2019, use an updated verson of the water matrix:
waterMatUpdated = np.genfromtxt('../cholera_data/watT_i_j_updated.csv', delimiter=',')
# print waterMat

# import the vector of distances D:
D = np.genfromtxt('../cholera_data/D_i.txt', delimiter=',')




# tspan = range(0, len(cases), 1)

#### ########################     Fixed parameters:   ########################################################
delta = 1/3.0 #decay of cholera in water, assuming 21 days, or 3 weeks


k_weekly = 0.2 # fraction of symptomatic people,
gamma_weekly = 1 / (1.0)      #recovery rate (range 1-2 weeks)
gammaA_weekly = 1 / (1.0)     #recovery rate asymptomatic (range 1-2 weeks)
gammaE_weekly = 1 / (1.25 / 7)     # duration of incubation period, 12-72 hours, taken from Nelson2009

#reduction in infectiousness for asymptomatic:
red_beta_weekly = 10**(-3)
red_mu_weekly = 10**(-7)


# reduction in travel by symptomatic infected people
m_weekly = 1.0
m1_weekly = 1.0
m2_weekly = 1.0

# natural immunity waning
sigma_weekly = 1.0 / (52 * 5)  # Five years of natural immunity
#rho = 1.0

# numDep = 1
# Water saturation vector
V_weekly = (10 ** 5)#1 * np.array((10 ** 5) * np.ones(numDep))
aseason_weekly = 1.0
pseason_weekly = 52.0

#vaccination parameters:
#load the file with the VE estimates:
VEtable = pd.read_csv('../cholera_data/VEefficacyTable.csv', delimiter=',')
# print VEtable


#Make a matrix of VEs for each vaccine scenario (3 columns, one per scenario, 3 rows:
VEmatrix = np.zeros((2,3))
VEmatrix[:, 1] = 0.76
VEmatrix[0,0] = np.median(VEtable.iloc[:, 1]) #median of values of the first column of the table
VEmatrix[0,2] = np.median(VEtable.iloc[:, 3])
VEmatrix[1,0] = np.median(VEtable.iloc[:12, 1]) #median of values of the first column of the table for the first 12 months
VEmatrix[1,2] = np.median(VEtable.iloc[:12, 3])

print VEmatrix

# #will compute the mean VE for two doses for VE scenario 1:
# VE2_weekly = np.median(VEtable.iloc[:, 1])
# VE1_weekly = VE2_weekly
#
# theta1_weekly = 1 - VE1_weekly
# theta2_weekly = 1 - VE2_weekly
#
# VE1_under5_weekly = VE1_weekly*0.4688
# VE2_under5_weekly = VE2_weekly*0.4688
# theta1_under5_weekly = 1 - VE1_under5_weekly
# theta2_under5_weekly = 1 - VE2_under5_weekly



#for this round, we will assume that the VE is constant and the only thing that changes is the rate at which it wanes for
#one or two doses:
omega1_weekly = 1/(52.0)
omega2_weekly = 1/(52.0*5)

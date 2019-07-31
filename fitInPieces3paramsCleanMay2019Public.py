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

mycolors = sns.color_palette("hls", 10)

#in this script, I will fit the data from mybreakingpoint onward. the parameter mybreakingpoint can be chosen to be whatever we want
#use the old code to fit from 2010 to end of 2016



#use that fit to obtain the initial conditions for 2017 on:



def simulateUpTo2016FullPop3params(params, extraParams):
    """
    This function will just fit from the beginning of the epidemic to the first week of November 2016, right before the
    historical vaccination campaigns started.

    :param params:
    :param extraParams:
    :return:
    """
    beta, betaW, mu = params

    aseason, delta, gamma, gammaA, gammaE, k, m, nInfected, pseason, Pop, red_beta, red_mu, rlen, rep, \
    sigma, tau, tspan_length, V, vrate, waterMat, wRate = extraParams

    beta_A = red_beta*beta
    mu_A = red_mu*mu

    repMat = [rep]*10

    numDep = 10

    travelMat = formTravelMat1(numDep, rlen, tau, vrate, Pop)
    #set initial conditions:
    S0 = np.copy(Pop) - nInfected
    E0 = np.zeros(np.size(Pop))
    I0 = map(lambda x: x / reporting_rate, nInfected)#nInfected/rep # np.zeros(np.size(Pop))
    A0 = np.zeros(np.size(Pop))
    R0 = np.zeros(np.size(Pop))
    RA0 = np.zeros(np.size(Pop))
    W0 = np.zeros(np.size(Pop))  # 15*Pop*365
    C0 = I0



    initCond = np.array([S0, E0, I0, A0, R0, RA0, W0, C0]).reshape((8 * numDep))
    # print initCond
    tspan = range(tspan_length)

    paramsODE =  [aseason, beta, beta_A, betaW,  delta, gamma, gammaA,  gammaE, k, m, mu, mu_A, numDep, \
     pseason, sigma,  travelMat, V, wRate, waterMat]


    sol, info = odeint(choleraEqs10WithoutVaccinationNetwork, initCond, tspan,
           args=(paramsODE,), full_output=True)


    if info['message'] == 'Integration successful.':
        temp = sol[:, 70:]
        newcases = np.zeros((np.size(sol[:, 0]), 10))

        newcases[0, :] = sol[0, 70:80]

        for jvals in range(10):
            for t in range(1, len(tspan)):
                newcases[t, jvals] = (np.sum(temp[t, jvals]) - np.sum(temp[t - 1, jvals]))

            # newcases[:, jvals] = repMat[jvals] * newcases[:, jvals]


        # print np.shape(newcases)
        return [sol, newcases]
    else:
        # print 'hola'
        return [0, np.ones((tspan_length, 10))*10**8]



def fit2017on3params(params, extraParams):
    '''
    here I will fit to the epidemic data without taking into account vaccination.
    :param initCond:
    :param params:
    :param extraParams:
    :return:
    '''
    beta, betaW, mu = params

    aseason, delta, gamma, gammaA, gammaE, k, m, pseason, Pop, red_beta, red_mu, rlen, rep, \
    sigma, tau, tspan, V, vrate, waterMat, wRate, initCond, casesInit = extraParams

    beta_A = red_beta * beta
    mu_A = red_mu * mu

    repMat = [rep] * 10

    numDep = 10

    travelMat = formTravelMat1(numDep, rlen, tau, vrate, Pop)

    paramsODE =  [aseason, beta, beta_A, betaW,  delta, gamma, gammaA,  gammaE, k, m, mu, mu_A, numDep, \
     pseason, sigma,  travelMat, V, wRate, waterMat]


    sol, info = odeint(choleraEqs10WithoutVaccinationNetwork, initCond, tspan,
           args=(paramsODE,), full_output=True)


    if info['message'] == 'Integration successful.':
        temp = sol[:, 70:]
        newcases = np.zeros((np.size(sol[:, 0]), 10))

        newcases[0, :] = casesInit


        for jvals in range(10):
            for t in range(1, len(tspan)):

                newcases[t, jvals] = (np.sum(temp[t, jvals]) - np.sum(temp[t - 1, jvals]))

            # newcases[:, jvals] = repMat[jvals] * newcases[:, jvals]


        return newcases

    else:
        # print 'hola'
        return np.ones((len(tspan), 10)) * 10 ** 8

def fit2017on3paramsWithFracSus(params, extraParams):
    '''
    here I will fit to the epidemic data without taking into account vaccination.
    :param initCond:
    :param params:
    :param extraParams:
    :return:
    '''
    beta, betaW, mu, fracSus = params

    aseason, delta, gamma, gammaA, gammaE, k, m, nInfected2017on, pseason, Pop, red_beta, red_mu, rlen, rep, \
    sigma, tau, tspan, V, vrate, waterMat, wRate = extraParams

    numDep = 10

    # I0 = (1 / rep) * np.array(nInfected2017on)
    # A0 = ((1 - data.k_weekly) / data.k_weekly) * np.array(I0)
    #
    # S0 = fracSus*np.copy(data.totalPop) - (I0 + A0)
    # E0 = np.zeros(np.size(data.totalPop))
    #
    #
    #
    # R0 = (data.k_weekly)*(1- fracSus)*np.array(np.copy(data.totalPop))
    # RA0 = (1- data.k_weekly)*(1- fracSus)*np.array(np.copy(data.totalPop))
    # W0 = np.zeros(np.size(data.totalPop))  # 15*Pop*365
    # C0 = I0

    # set initial conditions based on the number of reported cases:
    I0 = (1.0 / rep) * np.array(nInfected2017on)
    A0 = ((1 - k) / k) * np.array(I0)
    R0 = k * (1 - fracSus) * np.array(np.copy(Pop) - (I0 + A0))
    RA0 = (1 - k) * (1 - fracSus) * np.array(np.copy(Pop) - (I0 + A0))
    S0 = fracSus * (np.copy(Pop) - (I0 + A0))
    E0 = np.zeros(np.size(Pop))
    W0 = np.zeros(np.size(Pop))
    C0 = I0


    initCond = np.array([S0, E0, I0, A0, R0, RA0, W0, C0]).reshape((8 * numDep))
    # print np.sum(initCond[:60])
    # print np.sum(data.totalPop)
    # print np.sum(initCond[:60]) - np.sum(data.totalPop)
    beta_A = red_beta * beta
    mu_A = red_mu * mu

    travelMat = formTravelMat1(numDep, rlen, tau, vrate, Pop)

    paramsODE =  [aseason, beta, beta_A, betaW,  delta, gamma, gammaA,  gammaE, k, m, mu, mu_A, numDep, \
     pseason, sigma,  travelMat, V, wRate, waterMat]


    sol, info = odeint(choleraEqs10WithoutVaccinationNetwork, initCond, tspan,
           args=(paramsODE,), full_output=True)


    if info['message'] == 'Integration successful.':
        temp = sol[:, 70:]
        newcases = np.zeros((np.size(sol[:, 0]), 10))

        newcases[0, :] = C0


        for jvals in range(10):
            for t in range(1, len(tspan)):

                newcases[t, jvals] = (np.sum(temp[t, jvals]) - np.sum(temp[t - 1, jvals]))

            # newcases[:, jvals] = repMat[jvals] * newcases[:, jvals]


        return newcases

    else:
        # print 'hola'
        return np.ones((len(tspan), 10)) * 10 ** 8



def fit2017onfitMu(params, extraParams):
    '''
    fitting only mu with the same fraction of susceptibles as in the epidemic piece
    '''
    mu = params[0]

    aseason, beta, betaW, casesInit , delta, gamma, gammaA, gammaE, initCond, k, m, pseason, Pop, red_beta, red_mu, rlen, rep, \
    sigma, tau, tspan, V, vrate, waterMat, wRate = extraParams

    beta_A = red_beta * beta
    mu_A = red_mu * mu


    numDep = 10

    travelMat = formTravelMat1(numDep, rlen, tau, vrate, Pop)

    paramsODE =  [aseason, beta, beta_A, betaW,  delta, gamma, gammaA,  gammaE, k, m, mu, mu_A, numDep, \
     pseason, sigma,  travelMat, V, wRate, waterMat]


    sol, info = odeint(choleraEqs10WithoutVaccinationNetwork, initCond, tspan,
           args=(paramsODE,), full_output=True)


    if info['message'] == 'Integration successful.':
        temp = sol[:, 70:]
        newcases = np.zeros((np.size(sol[:, 0]), 10))

        newcases[0, :] = casesInit


        for jvals in range(10):
            for t in range(1, len(tspan)):

                newcases[t, jvals] = (np.sum(temp[t, jvals]) - np.sum(temp[t - 1, jvals]))

            # newcases[:, jvals] = repMat[jvals] * newcases[:, jvals]


        return newcases

    else:

        return np.ones((len(tspan), 10)) * 10 ** 8



def fit2017onMuWithFracSusFixed(params, extraParams):
    '''
    fitting mu with the fraction of susceptibles fixed
    '''
    mu = params

    aseason, beta, betaW, delta, fracSus, gamma, gammaA, gammaE, k, m, nInfected2017on, pseason, Pop, red_beta, red_mu, rlen, rep, \
    sigma, tau, tspan, V, vrate, waterMat, wRate = extraParams

    numDep = 10

    # I0 = (1 / rep) * np.array(nInfected2017on)
    # A0 = ((1 - data.k_weekly) / data.k_weekly) * np.array(I0)
    #
    # S0 = fracSus*np.copy(data.totalPop) - (I0 + A0)
    # E0 = np.zeros(np.size(data.totalPop))
    #
    #
    #
    # R0 = (data.k_weekly)*(1- fracSus)*np.array(np.copy(data.totalPop))
    # RA0 = (1- data.k_weekly)*(1- fracSus)*np.array(np.copy(data.totalPop))
    # W0 = np.zeros(np.size(data.totalPop))  # 15*Pop*365
    # C0 = I0

    # set initial conditions based on the number of reported cases:
    I0 = (1.0 / rep) * np.array(nInfected2017on)
    A0 = ((1 - k) / k) * np.array(I0)
    R0 = k * (1 - fracSus) * np.array(np.copy(Pop) - (I0 + A0))
    RA0 = (1 - k) * (1 - fracSus) * np.array(np.copy(Pop) - (I0 + A0))
    S0 = fracSus * (np.copy(Pop) - (I0 + A0))
    E0 = np.zeros(np.size(Pop))
    W0 = np.zeros(np.size(Pop))
    C0 = I0


    initCond = np.array([S0, E0, I0, A0, R0, RA0, W0, C0]).reshape((8 * numDep))
    # print np.sum(initCond[:60])
    # print np.sum(data.totalPop)
    # print np.sum(initCond[:60]) - np.sum(data.totalPop)
    beta_A = red_beta * beta
    mu_A = red_mu * mu

    travelMat = formTravelMat1(numDep, rlen, tau, vrate, Pop)

    paramsODE =  [aseason, beta, beta_A, betaW,  delta, gamma, gammaA,  gammaE, k, m, mu, mu_A, numDep, \
     pseason, sigma,  travelMat, V, wRate, waterMat]


    sol, info = odeint(choleraEqs10WithoutVaccinationNetwork, initCond, tspan,
           args=(paramsODE,), full_output=True)


    if info['message'] == 'Integration successful.':
        temp = sol[:, 70:]
        newcases = np.zeros((np.size(sol[:, 0]), 10))

        newcases[0, :] = C0


        for jvals in range(10):
            for t in range(1, len(tspan)):

                newcases[t, jvals] = (np.sum(temp[t, jvals]) - np.sum(temp[t - 1, jvals]))

            # newcases[:, jvals] = repMat[jvals] * newcases[:, jvals]


        return newcases

    else:
        # print 'hola'
        return np.ones((len(tspan), 10)) * 10 ** 8





def errorWeeklyFunMu(params, mydata, extraParams):
    if (any(float(x)<0.0 for x in params)):
        mysim = 10 ** 8 * np.ones(np.shape(mydata))
    else:

        mysim = fit2017onfitMu(params, extraParams)

    return np.reshape((mydata - mysim), (np.size(mydata), ))#.tolist()



def errorWeeklyFunMuWithFracSusFixed(params, mydata, extraParams):
    if (any(float(x)<0.0 for x in params)):
        mysim = 10 ** 8 * np.ones(np.shape(mydata))
    else:

        mysim = fit2017onMuWithFracSusFixed(params, extraParams)

    return np.reshape((mydata - mysim), (np.size(mydata), ))#.tolist()




def errorWeeklyFunAllFracSus(params, mydata, extraParams):
    if (any(float(x)<0.0 for x in params)):
        mysim = 10 ** 8 * np.ones(np.shape(mydata))
    elif params[3] > 1.0:
        mysim = 10 ** 8 * np.ones(np.shape(mydata))
    elif params[3]< 0.1:
        mysim = 10 ** 8 * np.ones(np.shape(mydata))
    else:
        # mysim = fit2017on3params(params, extraParams)
        mysim = fit2017on3paramsWithFracSus(params, extraParams)
        # print np.shape(mysim)
    # print np.shape(mysim)
    return np.reshape((mydata - mysim), (np.size(mydata), ))#.tolist()

def errorWeeklyFunAll(params, mydata, extraParams):
    if (any(float(x) < 0.0 for x in params)):
        mysim = 10 ** 8 * np.ones(np.shape(mydata))
    else:
        # mysim = fit2017on3params(params, extraParams)
        mysim = fit2017on3params(params, extraParams)
        # print np.shape(mysim)
    # print np.shape(mysim)
    return np.reshape((mydata - mysim), (np.size(mydata), ))#.tolist()


def leastSquaresFunFracSus(p1, mydata, extraParams):
    val = np.sqrt(np.sum(errorWeeklyFunAllFracSus(p1, mydata, extraParams)**2, axis=0))
    if val < 15000:
        val = 10**5
    return val


def leastSquaresFun(p1, mydata, extraParams):
    val = np.sqrt(np.sum(errorWeeklyFunAll(p1, mydata, extraParams)**2, axis=0))
    if val < 10000:
        val = 10**5
    return val

def leastSquaresMu(p1, mydata, extraParams):
    val = np.sqrt(np.sum(errorWeeklyFunMu(p1, mydata, extraParams)**2, axis=0))
    if val < 10000:
        val = 10**5
    return val



def leastSquaresMuWithFracSusFixed(p1, mydata, extraParams):
    val = np.sqrt(np.sum(errorWeeklyFunMuWithFracSusFixed(p1, mydata, extraParams)**2, axis=0))
    if val < 10000:
        val = 10**5
    return val



if __name__ == "__main__":
    depNames = ['artibonite', 'centre', 'grand_anse', 'nippes', 'nord', 'nord_est', 'nord_ouest', 'ouest', 'sud',
                'sud_est']

    today = time.strftime("%d%b%Y", time.localtime())

    mytime = time.localtime()
    myseed = np.abs(int((np.random.normal())*1000000))
    np.random.seed(myseed)
    print myseed

    index = 1#os.environ['SLURM_ARRAY_TASK_ID']
    # folder = 'fullModelLessParameters'#os.environ['SLURM_JOB_NAME']

    #define a dictionary with the names of the departments and the integers:

    depNames = ['fitPieces3paramsFracSus']

    # depVal = depNames.index(folder) + 1
    # print depVal



    # # create vectors of low and upper bounds for the parameters #beta, betaW, mu, delta, rep
    low_bounds = [0.0, 0.0, 0.0, 10]#, 0.0, 0.0]
    # upper_bounds = [10.0 ** (-5), 10 ** (-1), 1]
    upper_bounds = [10.0 ** (-4), 10 ** (-1), 1, 60] #0.4, 1]
    # params0 = lhs_params(low_bounds, upper_bounds, 1)[0]

    params0 = np.random.uniform(1.0, 100, 1)

    filename = 'extraParamsData_to_01_2019.py'
    f = open(filename)
    global data
    data = imp.load_source('data', '', f)
    f.close()

    fullData = data.cases[:, 1:11]
    totalData = data.cases[:, 11]

    mybreakingPoint = 176#176#200
    #full data up to breaking point:
    fullData2016 = fullData[:mybreakingPoint, :mybreakingPoint]

    totalData2016 = totalData[:mybreakingPoint]

    tspan_length = len(fullData2016)
    nInfected = []

    for depVal in range(1, 11):
        dep_data = data.cases[:, depVal]
        nInfected.append(dep_data[np.nonzero(dep_data)[0][0]])
    # print nInfected

    reporting_rate = 0.2

    fullSetOfParams = [9.92853814e-07, 4.03310744e-02, 3.98208148e+03, 9.92853814e-07,
     4.03310744e-02, 2.58208148e+00, 0.750]#4.03310744e-02, 2.58208148e+00 ,0.6]

    beta1, betaW1, mu1 = fullSetOfParams[0:3]
    beta2, betaW2, mu2, fracSus = fullSetOfParams[3:]

    #mu2 = mu1

    # this is my old best fit: beta, betaW, mu = [6.95834824e-06, 4.59991608e-02, 5.18938848e-01]
    # beta, betaW, mu = [6.95834824e-06, 1.59991608e-01, 2]
    tau, vrate, wRate = [2.0, 1 * 10 ** (-12), 0.5]

    beta_A = data.red_beta_weekly * beta1
    mu_A = data.red_mu_weekly * mu1

    numDep = 10

    TravelMat = formTravelMat1(numDep, data.rlen, tau, vrate, data.totalPop)
    params = [beta1, betaW1, mu1]

    myfracSus = 0.6


    tempAseason = 0.4
    fullData2017on = fullData[mybreakingPoint:, :]
    totalData2017on = totalData[mybreakingPoint:]
    tspan2017on = range(tspan_length, tspan_length + len(fullData2017on))
    params2 = [beta2, betaW2, mu2, fracSus ]#[2.55834824e-06, 1.59991608e-02, 2, 0.6]




    extraParams = [tempAseason,  data.delta, data.gamma_weekly, data.gammaA_weekly, data.gammaE_weekly,
                   data.k_weekly, data.m_weekly, nInfected,
                   data.pseason_weekly, data.totalPop, data.red_beta_weekly, data.red_mu_weekly,
                   data.rlen, reporting_rate, data.sigma_weekly, tau,
                   tspan_length, data.V_weekly, vrate,
                   data.waterMatUpdated, wRate]



    extraParamsMu = [tempAseason, beta1, betaW1, data.delta, data.gamma_weekly, data.gammaA_weekly, data.gammaE_weekly,
                   data.k_weekly, data.m_weekly, nInfected,
                   data.pseason_weekly, data.totalPop, data.red_beta_weekly, data.red_mu_weekly,
                   data.rlen, reporting_rate, data.sigma_weekly, tau,
                   tspan_length, data.V_weekly, vrate,
                   data.waterMatUpdated, wRate]

    start = time.time()
    [sol, cases] = simulateUpTo2016FullPop3params(params, extraParams)


    #second attempt: take 1 -  fracSusc of the population to be recovered:


    nInfected2017on = fullData2017on[0, :]

    # nbYears = int(np.floor(tspan2017on[-1]/52))

    extraParamsEndemic = [tempAseason, data.delta, data.gamma_weekly, data.gammaA_weekly, data.gammaE_weekly,
                         data.k_weekly, data.m_weekly, nInfected2017on,
                         data.pseason_weekly, data.totalPop, data.red_beta_weekly, data.red_mu_weekly,
                         data.rlen, reporting_rate, data.sigma_weekly, tau,
                         tspan2017on, data.V_weekly, vrate,
                         data.waterMatUpdated, wRate]

    extraParamsEndemic_Mu = [tempAseason, beta1, betaW1, cases[-1], data.delta,
                             data.gamma_weekly, data.gammaA_weekly, data.gammaE_weekly,
                             sol[-1,:],
                              data.k_weekly, data.m_weekly,
                              data.pseason_weekly, data.totalPop, data.red_beta_weekly, data.red_mu_weekly,
                              data.rlen, reporting_rate, data.sigma_weekly, tau,
                              tspan2017on, data.V_weekly, vrate,
                              data.waterMatUpdated, wRate]



    extraParamsEndemic_MuFracSusFixed = [tempAseason, beta1, betaW1, data.delta, myfracSus,
                             data.gamma_weekly, data.gammaA_weekly, data.gammaE_weekly,
                              data.k_weekly, data.m_weekly,
                              nInfected2017on,
                              data.pseason_weekly, data.totalPop, data.red_beta_weekly, data.red_mu_weekly,
                              data.rlen, reporting_rate, data.sigma_weekly, tau,
                              tspan2017on, data.V_weekly, vrate,
                              data.waterMatUpdated, wRate]



    p2 = optimize.fmin(leastSquaresMuWithFracSusFixed, params0, args=(fullData2017on, extraParamsEndemic_MuFracSusFixed))
    # print p2
    print('[' + ','.join(map(str, p2)) + ']')

    results1 = [index, p2, params0, myseed, leastSquaresFunFracSus(p2, fullData2017on, extraParamsEndemic_MuFracSusFixed)]


    # # params3 = [2.55834824e-06,  1.59991608e-02, 2, 0.5]
    # #
    # # cases2017_1 = fit2017on3params(p1, extraParams2017on)
    # # cases2017_2 = fit2017on3paramsWithFracSus(params2, extraParams2017on_2)
    # cases2017_fit = fit2017onMuWithFracSusFixed(p2, extraParamsEndemic_MuFracSusFixed)
    # cases2017_3 = fit2017on3paramsWithFracSus(params2, extraParamsEndemic)
    # #
    # #
    # # fig2 = plt.figure(2)
    # # plt.plot(tspan2017on, reporting_rate * np.sum(cases2017_1, 1), color='r', linewidth=2,
    # #          label='using old sol')
    # # plt.plot(range(tspan_length), reporting_rate * np.sum(cases, 1), color='r', linewidth=2, label='old fit')
    # plt.plot(tspan2017on, reporting_rate * np.sum(cases2017_fit, 1), color='r', linewidth=2, label='new fit'
    #         )
    # plt.plot(range(tspan_length), reporting_rate * np.sum(cases, 1), color='g', linewidth=2, )
    # plt.plot(tspan2017on, reporting_rate * np.sum(cases2017_3, 1), color='g', linewidth=2,
    #          label='updated')
    # # plt.plot(tspan2017on, reporting_rate * np.sum(cases2017_3, 1), color='b', linewidth=2, label = 'New fit')
    # # plt.ylim([0, 10000])
    # # plt.xticks(np.arange(10, tspan2017on[-1], step=52), range(2011, 2011 + nbYears), rotation=90, fontsize=16)  #
    # # # plt.xlim([150, tspan2017on[-1]])
    # plt.plot(data.cases[:, 0], data.cases[:, 11], 'ko--', markersize=1, linewidth=2, label='data')
    # # #     # #     plt.plot(nbWeeks[:10], weeklyCases[:10], 'ko--', markersize=1, linewidth=0.75)
    # plt.legend()
    # #
    # #
    # #
    # #
    # plt.show()


    #store the results in a convenient pickle file:
    filename0 = 'leastSquaresDataUpto2019/'  + 'fitPiecesMuWithFracSusFixed' + '/' +  'fitPiecesMuWithFracSusFixed' + '_'\
                + 'breakingPoint_' + str(mybreakingPoint) + '_fracSus' + str(myfracSus) + '_' + str(index) + 'Iterations_' + \
                today + '.pickle'

    print filename0

    myfile = open(filename0, 'wb')
    pickle.dump(results1, myfile)
    myfile.close()
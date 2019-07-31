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
from choleraEqsPublic import  choleraEqs11WithVaccinationNetwork
from functionsCholeraProjectPublic import formTravelMat1
from functools import partial
# import seaborn as sns
import timeit
import functools
import random
import os

# mycolors = sns.color_palette("hls", 10)


#this file runs all the functions with the different coverage scenarios for a single parameter set and stores the results
#in a big list and dumps it on a pickle file.

def computeNationalWeeklyCases(sol,weeklyCases0):
    "this function receives the solution of the ode in weeks and will just read off the number of new cases per week"
    [dim1, dim2] = np.shape(sol)

    weeklyCases = np.zeros(dim1)
    # print 'a', np.shape(weeklyCases)
    temp = np.sum(sol[:, 310:360], 1)
    # print np.shape(temp)
    weeklyCases[0] = weeklyCases0
    for t in range(1, dim1):

        weeklyCases[t] = temp[t] - temp[t-1]

    return weeklyCases


def computeDepartmentWeeklyCases(sol, weeklyCases0):
    [dim1, dim2] = np.shape(sol)
    C = sol[:, 310:320]
    C1 = sol[:, 320:330]
    C2 = sol[:, 330:340]
    C1under5 = sol[:, 340:350]
    C2under5 = sol[:, 350:360]

    # cases = [C, C1, C2, C1under5, C2under5]
    weeklyCases = np.zeros((dim1, 10))

    casesDep = np.zeros((dim1, 10))
    weeklyCases = np.zeros((dim1, 10))
    for ivals in range(10):
        casesDep[:, ivals] = C[:, ivals] + C1[:, ivals] + C2[:, ivals] + C1under5[:, ivals] + C2under5[:, ivals]

    print np.shape(casesDep)
    weeklyCases[0, :] = weeklyCases0

    for t in range(1, dim1):
        weeklyCases[t, :] = casesDep[t, :] - casesDep[t - 1, :]

    return weeklyCases






def simulateBaselineInPieces(params1, params2, extraParams, mybreakingpoint):
    """
    Simulate the baseline epidemic (from Oct 2010 to Jan 2019) in two pieces, where the breaking point is determined by
    the parameter mybreakingpoint
    :param params1: Set of parameters to run the ODES from Oct 2010 to mybreaking point
    :param params2: Set of parameters to run the ODES from mybreakingpoint to Jan 2019
    :param extraParams: extraParameters necessary to run the ODEs
    :param mybreakingpoint: Date where we restart the simulations
    :return:
    """

    depNames = ['artibonite', 'centre', 'grand_anse', 'nippes', 'nord', 'nord_est', 'nord_ouest', 'ouest', 'sud',
                'sud_est']
    depVaccinated = ['grand_anse', 'sud', 'ouest', 'centre', 'artibonite']

    # pop_under_5 = 1288122
    percentage_under_5 = 0.118

    fracSus, k, Pop, nInfected, nInfectedAtBreakingPoint, reporting_rate, tspan_length = extraParams

    numDep = 10

    #set initial conditions: each of the following variables is a vector
    S0 = np.copy(Pop) - nInfected
    E0 = np.zeros(np.size(Pop))
    I0 = nInfected # np.zeros(np.size(Pop))
    A0 = np.zeros(np.size(Pop))
    R0 = np.zeros(np.size(Pop))
    RA0 = np.zeros(np.size(Pop))


    #for simplicity, we run the big model with all the vaccination classes even though there will be no vaccination here
    #since this is modeling the era pre-vaccination
    #vaccinated with one dose
    V10 = np.zeros(np.size(Pop))
    E10 = np.zeros(np.size(Pop))
    I10 = np.zeros(np.size(Pop))
    A10 = np.zeros(np.size(Pop))
    R10 = np.zeros(np.size(Pop))
    RA10 = np.zeros(np.size(Pop))

    #vaccinated with two doses
    V20 = np.zeros(np.size(Pop))
    E20 = np.zeros(np.size(Pop))
    I20 = np.zeros(np.size(Pop))
    A20 = np.zeros(np.size(Pop))
    R20 = np.zeros(np.size(Pop))
    RA20 = np.zeros(np.size(Pop))

    #children under 5:
    V1_under50 = np.zeros(np.size(Pop))
    E1_under50 = np.zeros(np.size(Pop))
    I1_under50 = np.zeros(np.size(Pop))
    A1_under50 = np.zeros(np.size(Pop))
    R1_under50 = np.zeros(np.size(Pop))
    RA1_under50 = np.zeros(np.size(Pop))
    V2_under50 = np.zeros(np.size(Pop))
    E2_under50 = np.zeros(np.size(Pop))
    I2_under50 = np.zeros(np.size(Pop))
    A2_under50 = np.zeros(np.size(Pop))
    R2_under50 = np.zeros(np.size(Pop))
    RA2_under50 = np.zeros(np.size(Pop))

    W0 = np.zeros(np.size(Pop))
    C0 = np.copy(I0)
    C10 = np.zeros(np.size(Pop))
    C20 = np.zeros(np.size(Pop))
    C1_under50 = np.zeros(np.size(Pop))
    C2_under50 = np.zeros(np.size(Pop))


    #initial conditions
    initCond = np.array([S0, E0, I0, A0, R0, RA0,
            V10, E10, I10, A10, R10, RA10,
            V20, E20, I20, A20, R20, RA20,
            V1_under50, E1_under50, I1_under50, A1_under50, R1_under50, RA1_under50,
            V2_under50, E2_under50, I2_under50, A2_under50, R2_under50, RA2_under50,
            W0, C0, C10, C20, C1_under50, C2_under50]).reshape((36 * numDep))

    #first time span
    tspan = range(0, mybreakingpoint)
    # run the ODEs
    sol, info = odeint(choleraEqs11WithVaccinationNetwork, initCond, tspan,
                        args=(params1, ), full_output=True)

    cases = computeNationalWeeklyCases(sol, (1/k)*((1.0 / reporting_rate) * np.sum(nInfected)))

    #run second portion of the baseline before vaccination with the new set of parameters:
    tspan2 = range(mybreakingpoint, tspan_length)


    #set the new initial conditions. To do this, we used the data for that particular breaking point. that is, we took
    #the number of reported infections for that date and back-calculated the number of symptomatic and asymptomatic infections
    #the parameter fracSus defines the fraction of susceptible people believed to be at that point in time. It was fit to
    #data.

    I0 = (1.0 / reporting_rate) * np.array(nInfectedAtBreakingPoint)
    A0 = ((1 - k) / k) * np.array(I0)
    R0 = k * (1 - fracSus) * np.array(np.copy(Pop) - (I0 + A0))
    RA0 = (1 - k) * (1 - fracSus) * np.array(np.copy(Pop) - (I0 + A0))
    S0 = fracSus * (np.copy(Pop) - (I0 + A0))
    E0 = np.zeros(np.size(Pop))


    #vaccinated with one dose
    V10 = np.zeros(np.size(Pop))
    E10 = np.zeros(np.size(Pop))
    I10 = np.zeros(np.size(Pop))
    A10 = np.zeros(np.size(Pop))
    R10 = np.zeros(np.size(Pop))
    RA10 = np.zeros(np.size(Pop))

    #vaccinated with two doses
    V20 = np.zeros(np.size(Pop))
    E20 = np.zeros(np.size(Pop))
    I20 = np.zeros(np.size(Pop))
    A20 = np.zeros(np.size(Pop))
    R20 = np.zeros(np.size(Pop))
    RA20 = np.zeros(np.size(Pop))

    #children under 5:
    V1_under50 = np.zeros(np.size(Pop))
    E1_under50 = np.zeros(np.size(Pop))
    I1_under50 = np.zeros(np.size(Pop))
    A1_under50 = np.zeros(np.size(Pop))
    R1_under50 = np.zeros(np.size(Pop))
    RA1_under50 = np.zeros(np.size(Pop))
    V2_under50 = np.zeros(np.size(Pop))
    E2_under50 = np.zeros(np.size(Pop))
    I2_under50 = np.zeros(np.size(Pop))
    A2_under50 = np.zeros(np.size(Pop))
    R2_under50 = np.zeros(np.size(Pop))
    RA2_under50 = np.zeros(np.size(Pop))

    W0 = np.zeros(np.size(Pop))
    C0 = np.copy(I0)
    C10 = np.zeros(np.size(Pop))
    C20 = np.zeros(np.size(Pop))
    C1_under50 = np.zeros(np.size(Pop))
    C2_under50 = np.zeros(np.size(Pop))


    #new set of initial conditions
    initCond2 = np.array([S0, E0, I0, A0, R0, RA0,
            V10, E10, I10, A10, R10, RA10,
            V20, E20, I20, A20, R20, RA20,
            V1_under50, E1_under50, I1_under50, A1_under50, R1_under50, RA1_under50,
            V2_under50, E2_under50, I2_under50, A2_under50, R2_under50, RA2_under50,
            W0, C0, C10, C20, C1_under50, C2_under50]).reshape((36 * numDep))



    #utilize the new set of parameters:
    sol2, info = odeint(choleraEqs11WithVaccinationNetwork, initCond2, tspan2,
                        args=(params2, ), full_output=True)

    # compute the number of cases
    cases2 = computeNationalWeeklyCases(sol2, (1/k)*((1.0/reporting_rate)*np.sum(nInfectedAtBreakingPoint)))

    #return the tspan and the number of cases
    tspanAll = np.concatenate((tspan, tspan2))
    solAll = np.concatenate((sol, sol2))
    casesAll = np.concatenate((cases, cases2))


    return [tspanAll, solAll,  casesAll]




def simulateVaccination(params1, params2, extraParams, vaccination_tspan, numberOfYearsAfterVaccination, vacRecovered, mybreakingPoint):
    """run the equations for vaccinating individuals in numberOfYearsVaccination and run equations for
    numberOfYearsAfterVaccination afterwards.
    We will vaccinate 70% of the population with two doses, 10% with a single dose
    vacRecovered is a switch = 0 or 1, if 1, we will vaccinate the recovered people and move them to the recovered vaccinated class
    thereby assuming some sort of boost from the vaccine in recovered people.
    """

    depNames = ['artibonite', 'centre', 'grand_anse', 'nippes', 'nord', 'nord_est', 'nord_ouest', 'ouest', 'sud',
                'sud_est']

    #we vaccinate the depts with higher incidence first, and follow down along the list.
    # per table, the order for vaccinating is:
    departments_ordered = ['centre', 'artibonite', 'ouest', 'nord_ouest', 'nord', 'sud', 'nippes', 'nord_est', \
                           'sud_est', 'grand_anse']

    # oer table, the population in each of these departements (ordered)
    pop_ordered = np.array([746236, 1727524, 4029705, 728807, 1067177, 774976, 342525, 393967, 632601, 468301])
    haitiPop = np.sum(pop_ordered)

    # population under 5 (taken from HopkinsIID github), 2015 projection:
    pop_under_5 = 1288122
    percentage_under_5 = pop_under_5 / haitiPop  # 11.8%

    vaccination_tspan_weeks = int(vaccination_tspan * 52)

    # want to vaccinate 80 % in vaccination_tspan_weeks, so we need to vaccinate numVacWeek persons per week:
    numVacWeek = round(0.8 * haitiPop / (vaccination_tspan_weeks))
    # print (numVacWeek)

    # number of weeks to be spent vaccinating each that department:
    #this gives a vector with the number of weeks to be spent in each department so that at the end we vaccinated 80% of
    #the population in each department and we spend overall the numVacWeek weeks we want to spend
    numWeekPerDepartment = np.round(0.8 * pop_ordered / numVacWeek)
    # print 'numWeekPerDepartment', numWeekPerDepartment
    real_vaccination_tspan_weeks = int(np.sum(numWeekPerDepartment))
    # print real_vaccination_tspan_weeks

    # fraction of people to vaccinate in each departement per week:
    fractionPerWeek = [numVacWeek / x for x in pop_ordered]
    # print fractionPerWeek


    # totalVaccines = 0
    totalWeeksVacCampaign = 0

    # totalWeeksVacCampaignPerDept = np.zeros(10)
    # totalVacGiven = 0

    #run baseline model:
    [tspan0, sol,  cases] = simulateBaselineInPieces(params1, params2, extraParams, mybreakingPoint)

    #copy the last line of the baseline run to use that as initial conditions
    finalSol = np.copy(sol[-1,:]).reshape(36, numDep)
    initCondDoNothing = np.copy(sol[-1,:])


    # # print np.shape(cases)
    vacSol = np.zeros((real_vaccination_tspan_weeks + 1, 36 * numDep))
    doNothing = np.zeros((real_vaccination_tspan_weeks + 1, 36 * numDep))

    vacSol[0, :] = sol[-1, :]
    doNothing[0,:]= sol[-1, :]
    # ######## VACCINATION     ########################
    # # march through each department to vaccinate for the number of weeks given in numWeekPerDepartment:
    finalSol = sol[-1,:].reshape(36, numDep)

    for ivals in range(10):

        numOfPulses = int(numWeekPerDepartment[ivals])
        # print numOfPulses
        myname = departments_ordered[ivals]
        # print myname
        myfraction = np.min([fractionPerWeek[ivals], 1])
        # print 'myfraction', myfraction
        myindex = depNames.index(myname)
        # print myindex



        # form vaccination multipliers:
        vacOneDoseOver5 = myfraction * (1.0 / 8) * (1 - percentage_under_5)
        vacTwoDosesOver5 = myfraction * (7. / 8) * (1 - percentage_under_5)
        vacOneDoseUnder5 = myfraction * (1.0 / 8) * (percentage_under_5)
        vacTwoDosesUnder5 = myfraction * (7. / 8) * (percentage_under_5)


        # march through the weeks vaccinating a single department in each week, as given by myindex.
        for weeks in range(numOfPulses):

            newInitCond = np.copy(finalSol)

            #move people from the susceptible class for that department to the vaccinated classes:
            newInitCond[0, myindex] = finalSol[0, myindex] - myfraction * finalSol[0, myindex]  #remove myfraction fraction of susceptibles

            newInitCond[6, myindex] = finalSol[6, myindex] + vacOneDoseOver5 * myfraction * finalSol[0, myindex]
            newInitCond[12, myindex] = finalSol[12, myindex] + vacTwoDosesOver5 * myfraction * finalSol[0, myindex]
            newInitCond[18, myindex] = finalSol[18, myindex] + vacOneDoseUnder5 * myfraction * finalSol[0, myindex]
            newInitCond[24, myindex] = finalSol[24, myindex] + vacTwoDosesUnder5 * myfraction * finalSol[0, myindex]


            if vacRecovered == 1: #if this is 1, then we also vaccinate recovered groups.
                newInitCond[4, myindex] = finalSol[4, myindex] - vacTwoDosesOver5 * myfraction * finalSol[4, myindex]

                newInitCond[10, myindex] = finalSol[10, myindex] + vacOneDoseOver5 * myfraction * finalSol[4, myindex]
                newInitCond[16, myindex] = finalSol[16, myindex] + vacTwoDosesOver5 * myfraction * finalSol[4, myindex]
                newInitCond[22, myindex] = finalSol[22, myindex] + vacOneDoseUnder5 * myfraction * finalSol[4, myindex]
                newInitCond[28, myindex] = finalSol[28, myindex] + vacTwoDosesUnder5 * myfraction * finalSol[4, myindex]
                newInitCond[5, myindex] = finalSol[5, myindex] - vacTwoDosesOver5 * myfraction * finalSol[5, myindex]
                newInitCond[11, myindex] = finalSol[11, myindex] + vacOneDoseOver5 * myfraction * finalSol[5, myindex]
                newInitCond[17, myindex] = finalSol[17, myindex] + vacTwoDosesOver5 * myfraction * finalSol[5, myindex]
                newInitCond[23, myindex] = finalSol[23, myindex] + vacOneDoseUnder5 * myfraction * finalSol[5, myindex]
                newInitCond[29, myindex] = finalSol[29, myindex] + vacTwoDosesUnder5 * myfraction * finalSol[5, myindex]
            # print newInitCond[17, myindex]


            ######run the ODE
            newInitCond = newInitCond.reshape(36 * numDep)
            tspanWeek = np.linspace(tspan_length -1 + totalWeeksVacCampaign, tspan_length -1 + totalWeeksVacCampaign + 1, 5)

            temp, info = odeint(choleraEqs11WithVaccinationNetwork, newInitCond, tspanWeek,
                        args=(params2,), full_output=True)

            ### run the ODE without modifying the initial conditions, hence in the baseline situation where no vaccination
            ### is applied.
            temp2, info = odeint(choleraEqs11WithVaccinationNetwork, initCondDoNothing, tspanWeek,
                         args=(params2,), full_output=True)

            #increase counter for number of weeks we have vaccinated
            totalWeeksVacCampaign += 1

            #store the solution in the vaccine Solution vector and the donothing vector
            vacSol[totalWeeksVacCampaign, :] = temp[-1, :]
            doNothing[totalWeeksVacCampaign, :] = temp2[-1, :]


            #reshape the last line of the solution so it can be used as initial condition in the next iteration
            finalSol = temp[-1, :].reshape(36, numDep)
            initCondDoNothing = temp2[-1, :]

    # print vacSol[-2:,:]

    #run the equations for numberOfYearsAfterVaccination:

    tspan_after_vaccination = range(tspan_length-1 + len(vacSol), tspan_length-1 + len(vacSol) + numberOfYearsAfterVaccination*52)
    initCondAfterVac = vacSol[-1,:]
    initCondAfterDoNothing = doNothing[-1,:]
    solAfterVac, info = odeint(choleraEqs11WithVaccinationNetwork, initCondAfterVac, tspan_after_vaccination,
                        args=(params2,), full_output=True)

    solAfterDoNothing, info = odeint(choleraEqs11WithVaccinationNetwork, initCondAfterDoNothing, tspan_after_vaccination,
                         args=(params2,), full_output=True)



    # compute number of cases during the vaccination campaign and afterwards:

    cases_vac = computeNationalWeeklyCases(vacSol, cases[-1])
    cases_doNothing = computeNationalWeeklyCases(doNothing, cases[-1])
    cases_AfterVac = computeNationalWeeklyCases(solAfterVac, cases_vac[-1])
    cases_AfterDoNothing = computeNationalWeeklyCases(solAfterDoNothing, cases_doNothing[-1])


    casesVac = np.concatenate((cases_vac, cases_AfterVac))
    casesBaseline = np.concatenate((cases_doNothing, cases_AfterDoNothing))


    totalCasesBaseline = np.sum(casesBaseline)
    totalCasesVac = np.sum(casesVac)


    #if there were no cases with vaccination or in the baseline, set the number of total Cases to 10**(-5) (this is to
    # avoid weird situations where the fraction reduction becomes negative if these two numbers are too close to zero and
    #totalCasesVac > totalCasesBaseline:
    if totalCasesBaseline < 10**(-5):
        totalCasesBaseline = 10 ** (-5)

    if totalCasesVac < 10**(-5):
        totalCasesVac = 10 ** (-5)

    fractionReduction = (totalCasesBaseline - totalCasesVac)/(totalCasesBaseline)
    tspan_Vaccination_on = range(tspan_length-1, tspan_length -1 + len(vacSol) + numberOfYearsAfterVaccination*52)

    return [tspan0,  tspan_Vaccination_on, cases, casesVac, casesBaseline, fractionReduction]


def simulateVaccinationScenario2(params1, params2, extraParams, vaccination_tspan, numberOfYearsAfterVaccination, vacRecovered, mybreakingPoint):
    """run the equations for vaccinating individuals in numberOfYearsVaccination and run equations for
    numberOfYearsAfterVaccination afterwards.
    We will vaccinate 70% of the population with two doses, 10% with a single dose for Artibonite and Centre
    vacRecovered is a switch = 0 or 1, if 1, we will vaccinate the recovered people and move them to the recovered vaccinated class
    thereby assuming some sort of boost from the vaccine in recovered people.
    """

    numDep = 10
    depNames = ['artibonite', 'centre', 'grand_anse', 'nippes', 'nord', 'nord_est', 'nord_ouest', 'ouest', 'sud',
                'sud_est']

    #we vaccinate the depts with higher incidence first, and follow down along the list.
    # per table, the order for vaccinating is:
    departments_ordered = ['centre', 'artibonite', 'ouest', 'nord_ouest', 'nord', 'sud', 'nippes', 'nord_est', \
                           'sud_est', 'grand_anse']

    # oer table, the population in each of these departements (ordered)
    pop_ordered = np.array([746236, 1727524, 4029705, 728807, 1067177, 774976, 342525, 393967, 632601, 468301])
    haitiPop = np.sum(pop_ordered)
    pop_centre_artibonite = pop_ordered[0] + pop_ordered[1]

    # population under 5 (taken from HopkinsIID github), 2015 projection:
    pop_under_5 = 1288122
    percentage_under_5 = pop_under_5 / haitiPop  # 11.8%

    vaccination_tspan_weeks = int(vaccination_tspan * 52)

    deps_vaccinated = ['centre', 'artibonite']
    pop_deps_vaccinated = pop_ordered[0:2]


    # want to vaccinate 80 % in vaccination_tspan_weeks, so we need to vaccinate numVacWeek persons per week:
    numVacWeek = round(0.8 * pop_centre_artibonite / (vaccination_tspan_weeks))
    # print (numVacWeek)

    # number of weeks to be spent vaccinating each that department:
    #this gives a vector with the number of weeks to be spent in each department so that at the end we vaccinated 80% of
    #the population in each department and we spend overall the numVacWeek weeks we want to spend
    numWeekPerDepartment = np.round(0.8 * pop_deps_vaccinated / numVacWeek)
    # print 'numWeekPerDepartment', numWeekPerDepartment
    real_vaccination_tspan_weeks = int(np.sum(numWeekPerDepartment))
    # print real_vaccination_tspan_weeks

    # fraction of people to vaccinate in each departement per week:
    fractionPerWeek = [numVacWeek / x for x in pop_deps_vaccinated]
    # print fractionPerWeek



    totalVaccines = 0
    totalWeeksVacCampaign = 0

    totalWeeksVacCampaignPerDept = np.zeros(10)
    totalVacGiven = 0

    #run baseline model:
    [tspan0, sol,  cases] = simulateBaselineInPieces(params1, params2, extraParams, mybreakingPoint)
    # print np.shape(sol)
    finalSol = np.copy(sol[-1,:]).reshape(36, numDep)
    initCondDoNothing = np.copy(sol[-1,:])


    vacSol = np.zeros((real_vaccination_tspan_weeks + 1, 36 * numDep))
    doNothing = np.zeros((real_vaccination_tspan_weeks + 1, 36 * numDep))


    #starting conditions for the vaccine and the doNothing cases: copy the baseline last line
    vacSol[0, :] = sol[-1, :]
    doNothing[0,:]= sol[-1, :]
    # ######## VACCINATION     ########################
    # # march through each department to vaccinate for the number of weeks given in numWeekPerDepartment:
    finalSol = sol[-1,:].reshape(36, numDep)

    for ivals in range(2):

        numOfPulses = int(numWeekPerDepartment[ivals])
        # print numOfPulses
        myname = departments_ordered[ivals]
        # print myname
        myfraction = np.min([fractionPerWeek[ivals], 1])
        # print 'myfraction', myfraction
        myindex = depNames.index(myname)
        # print myindex



        # form vaccination multipliers:
        vacOneDoseOver5 = myfraction * (1.0 / 8) * (1 - percentage_under_5)
        vacTwoDosesOver5 = myfraction * (7. / 8) * (1 - percentage_under_5)
        vacOneDoseUnder5 = myfraction * (1.0 / 8) * (percentage_under_5)
        vacTwoDosesUnder5 = myfraction * (7. / 8) * (percentage_under_5)

        # print (vacOneDoseOver5 + vacTwoDosesOver5 + vacTwoDosesUnder5 + vacOneDoseUnder5) - myfraction

        for weeks in range(numOfPulses):
            # print weeks
        # print numOfPulses
            newInitCond = np.copy(finalSol)
            #move people from the susceptible class for that department to the vaccinated classes:
            newInitCond[0, myindex] = finalSol[0, myindex] - myfraction * finalSol[0, myindex]  #remove myfraction fraction of susceptibles
            # print newInitCond[0, myindex]
            newInitCond[6, myindex] = finalSol[6, myindex] + vacOneDoseOver5 * myfraction * finalSol[0, myindex]
            newInitCond[12, myindex] = finalSol[12, myindex] + vacTwoDosesOver5 * myfraction * finalSol[0, myindex]
            newInitCond[18, myindex] = finalSol[18, myindex] + vacOneDoseUnder5 * myfraction * finalSol[0, myindex]
            newInitCond[24, myindex] = finalSol[24, myindex] + vacTwoDosesUnder5 * myfraction * finalSol[0, myindex]
            # print newInitCond[12, myindex]
            if vacRecovered == 1:
                newInitCond[4, myindex] = finalSol[4, myindex] - vacTwoDosesOver5 * myfraction * finalSol[4, myindex]
                # print newInitCond[4, myindex]
                newInitCond[10, myindex] = finalSol[10, myindex] + vacOneDoseOver5 * myfraction * finalSol[4, myindex]
                newInitCond[16, myindex] = finalSol[16, myindex] + vacTwoDosesOver5 * myfraction * finalSol[4, myindex]
                newInitCond[22, myindex] = finalSol[22, myindex] + vacOneDoseUnder5 * myfraction * finalSol[4, myindex]
                newInitCond[28, myindex] = finalSol[28, myindex] + vacTwoDosesUnder5 * myfraction * finalSol[4, myindex]
                # print newInitCond[16, myindex]
                newInitCond[5, myindex] = finalSol[5, myindex] - vacTwoDosesOver5 * myfraction * finalSol[5, myindex]
                # print newInitCond[5, myindex]
                newInitCond[11, myindex] = finalSol[11, myindex] + vacOneDoseOver5 * myfraction * finalSol[5, myindex]
                newInitCond[17, myindex] = finalSol[17, myindex] + vacTwoDosesOver5 * myfraction * finalSol[5, myindex]
                newInitCond[23, myindex] = finalSol[23, myindex] + vacOneDoseUnder5 * myfraction * finalSol[5, myindex]
                newInitCond[29, myindex] = finalSol[29, myindex] + vacTwoDosesUnder5 * myfraction * finalSol[5, myindex]
            # print newInitCond[17, myindex]
            ######run the ODE
            newInitCond = newInitCond.reshape(36 * numDep)
            tspanWeek = np.linspace(tspan_length -1 + totalWeeksVacCampaign, tspan_length -1 + totalWeeksVacCampaign + 1, 5)

            temp, info = odeint(choleraEqs11WithVaccinationNetwork, newInitCond, tspanWeek,
                        args=(params2,), full_output=True)

            temp2, info = odeint(choleraEqs11WithVaccinationNetwork, initCondDoNothing, tspanWeek,
                         args=(params2,), full_output=True)

            totalWeeksVacCampaign += 1
            #print tspan_length -1 + totalWeeksVacCampaign
            vacSol[totalWeeksVacCampaign, :] = temp[-1, :]
            doNothing[totalWeeksVacCampaign, :] = temp2[-1, :]



            finalSol = temp[-1, :].reshape(36, numDep)
            initCondDoNothing = temp2[-1, :]

    # print vacSol[-2:,:]
    #run the equations for numberOfYearsAfterVaccination:

    tspan_after_vaccination = range(tspan_length-1 + len(vacSol), tspan_length-1 + len(vacSol) + numberOfYearsAfterVaccination*52)
    initCondAfterVac = vacSol[-1,:]
    initCondAfterDoNothing = doNothing[-1,:]
    solAfterVac, info = odeint(choleraEqs11WithVaccinationNetwork, initCondAfterVac, tspan_after_vaccination,
                        args=(params2,), full_output=True)

    solAfterDoNothing, info = odeint(choleraEqs11WithVaccinationNetwork, initCondAfterDoNothing, tspan_after_vaccination,
                         args=(params2,), full_output=True)

    # print totalWeeksVacCampaign

    # # compute number of cases during the vaccination campaign:

    cases_vac = computeNationalWeeklyCases(vacSol, cases[-1])
    cases_doNothing = computeNationalWeeklyCases(doNothing, cases[-1])
    cases_AfterVac = computeNationalWeeklyCases(solAfterVac, cases_vac[-1])
    cases_AfterDoNothing = computeNationalWeeklyCases(solAfterDoNothing, cases_doNothing[-1])


    casesVac = np.concatenate((cases_vac, cases_AfterVac))
    casesBaseline = np.concatenate((cases_doNothing, cases_AfterDoNothing))


    totalCasesBaseline = np.sum(casesBaseline)
    totalCasesVac = np.sum(casesVac)

    if totalCasesBaseline < 10 ** (-5):
        totalCasesBaseline = 10 ** (-5)

    if totalCasesVac < 10 ** (-5):
        totalCasesVac = 10 ** (-5)

    fractionReduction = (totalCasesBaseline - totalCasesVac) / (totalCasesBaseline)
    tspan_Vaccination_on = range(tspan_length - 1, tspan_length - 1 + len(vacSol) + numberOfYearsAfterVaccination * 52)

    return [tspan0,  tspan_Vaccination_on, cases, casesVac, casesBaseline, fractionReduction]




def simulateVaccinationCoverage95(params1, params2, extraParams, vaccination_tspan, numberOfYearsAfterVaccination, vacRecovered, mybreakingPoint):
    """run the equations for vaccinating individuals in numberOfYearsVaccination and run equations for
    numberOfYearsAfterVaccination afterwards.
    We will vaccinate 95% of the population with two doses, 1.667% with a single dose
    vacRecovered is a switch = 0 or 1, if 1, we will vaccinate the recovered people and move them to the recovered vaccinated class
    thereby assuming some sort of boost from the vaccine in recovered people.
    """

    depNames = ['artibonite', 'centre', 'grand_anse', 'nippes', 'nord', 'nord_est', 'nord_ouest', 'ouest', 'sud',
                'sud_est']

    #we vaccinate the depts with higher incidence first, and follow down along the list.
    # per table, the order for vaccinating is:
    departments_ordered = ['centre', 'artibonite', 'ouest', 'nord_ouest', 'nord', 'sud', 'nippes', 'nord_est', \
                           'sud_est', 'grand_anse']

    # oer table, the population in each of these departements (ordered)
    pop_ordered = np.array([746236, 1727524, 4029705, 728807, 1067177, 774976, 342525, 393967, 632601, 468301])
    haitiPop = np.sum(pop_ordered)

    # population under 5 (taken from HopkinsIID github), 2015 projection:
    pop_under_5 = 1288122
    percentage_under_5 = pop_under_5 / haitiPop  # 11.8%

    vaccination_tspan_weeks = int(vaccination_tspan * 52)

    # want to vaccinate 80 % in vaccination_tspan_weeks, so we need to vaccinate numVacWeek persons per week:
    numVacWeek = round(0.9667 * haitiPop / (vaccination_tspan_weeks))
    # print (numVacWeek)

    # number of weeks to be spent vaccinating each that department:
    #this gives a vector with the number of weeks to be spent in each department so that at the end we vaccinated 80% of
    #the population in each department and we spend overall the numVacWeek weeks we want to spend
    numWeekPerDepartment = np.round(0.9667 * pop_ordered / numVacWeek)
    # print 'numWeekPerDepartment', numWeekPerDepartment
    real_vaccination_tspan_weeks = int(np.sum(numWeekPerDepartment))
    # print real_vaccination_tspan_weeks

    # fraction of people to vaccinate in each departement per week:
    fractionPerWeek = [numVacWeek / x for x in pop_ordered]
    # print fractionPerWeek

    # this is a check that I did the math correctly: print np.sum(fractionPerWeek*pop_ordered*numWeekPerDepartment)/haitiPop
    # total number of weeks spent in the vaccination campaign is 161 instead of 156
    # print np.sum(numWeekPerDepartment)

    totalVaccines = 0
    totalWeeksVacCampaign = 0

    totalWeeksVacCampaignPerDept = np.zeros(10)
    totalVacGiven = 0

    #run baseline model:
    [tspan0, sol,  cases] = simulateBaselineInPieces(params1, params2, extraParams, mybreakingPoint)
    # print np.shape(sol)
    finalSol = np.copy(sol[-1,:]).reshape(36, numDep)
    initCondDoNothing = np.copy(sol[-1,:])

    # cases = computeDepartmentWeeklyCases(sol, (sol[0, 310:320]), reporting_rate)
    # cases = computeNationalWeeklyCases(sol, np.sum(sol[0, 310:320]))
    # # print np.shape(cases)
    vacSol = np.zeros((real_vaccination_tspan_weeks + 1, 36 * numDep))
    doNothing = np.zeros((real_vaccination_tspan_weeks + 1, 36 * numDep))

    vacSol[0, :] = sol[-1, :]
    doNothing[0,:]= sol[-1, :]
    # ######## VACCINATION     ########################
    # # march through each department to vaccinate for the number of weeks given in numWeekPerDepartment:
    finalSol = sol[-1,:].reshape(36, numDep)

    for ivals in range(10):

        numOfPulses = int(numWeekPerDepartment[ivals])
        # print numOfPulses
        myname = departments_ordered[ivals]
        # print myname
        myfraction = np.min([fractionPerWeek[ivals], 1])
        # print 'myfraction', myfraction
        myindex = depNames.index(myname)
        # print myindex

        # form vaccination multipliers:
        vacOneDoseOver5 = myfraction * (1.67 / 96.67) * (1 - percentage_under_5)
        vacTwoDosesOver5 = myfraction * (95. / 96.67) * (1 - percentage_under_5)
        vacOneDoseUnder5 = myfraction * (1.67 / 96.67) * (percentage_under_5)
        vacTwoDosesUnder5 = myfraction * (95. / 96.67) * (percentage_under_5)

        # print (vacOneDoseOver5 + vacTwoDosesOver5 + vacTwoDosesUnder5 + vacOneDoseUnder5) - myfraction

        for weeks in range(numOfPulses):
            # print weeks
        # print numOfPulses
            newInitCond = np.copy(finalSol)
            #move people from the susceptible class for that department to the vaccinated classes:
            newInitCond[0, myindex] = finalSol[0, myindex] - myfraction * finalSol[0, myindex]  #remove myfraction fraction of susceptibles
            # print newInitCond[0, myindex]
            newInitCond[6, myindex] = finalSol[6, myindex] + vacOneDoseOver5 * myfraction * finalSol[0, myindex]
            newInitCond[12, myindex] = finalSol[12, myindex] + vacTwoDosesOver5 * myfraction * finalSol[0, myindex]
            newInitCond[18, myindex] = finalSol[18, myindex] + vacOneDoseUnder5 * myfraction * finalSol[0, myindex]
            newInitCond[24, myindex] = finalSol[24, myindex] + vacTwoDosesUnder5 * myfraction * finalSol[0, myindex]
            # print newInitCond[12, myindex]
            if vacRecovered == 1:
                newInitCond[4, myindex] = finalSol[4, myindex] - vacTwoDosesOver5 * myfraction * finalSol[4, myindex]
                # print newInitCond[4, myindex]
                newInitCond[10, myindex] = finalSol[10, myindex] + vacOneDoseOver5 * myfraction * finalSol[4, myindex]
                newInitCond[16, myindex] = finalSol[16, myindex] + vacTwoDosesOver5 * myfraction * finalSol[4, myindex]
                newInitCond[22, myindex] = finalSol[22, myindex] + vacOneDoseUnder5 * myfraction * finalSol[4, myindex]
                newInitCond[28, myindex] = finalSol[28, myindex] + vacTwoDosesUnder5 * myfraction * finalSol[4, myindex]
                # print newInitCond[16, myindex]
                newInitCond[5, myindex] = finalSol[5, myindex] - vacTwoDosesOver5 * myfraction * finalSol[5, myindex]
                # print newInitCond[5, myindex]
                newInitCond[11, myindex] = finalSol[11, myindex] + vacOneDoseOver5 * myfraction * finalSol[5, myindex]
                newInitCond[17, myindex] = finalSol[17, myindex] + vacTwoDosesOver5 * myfraction * finalSol[5, myindex]
                newInitCond[23, myindex] = finalSol[23, myindex] + vacOneDoseUnder5 * myfraction * finalSol[5, myindex]
                newInitCond[29, myindex] = finalSol[29, myindex] + vacTwoDosesUnder5 * myfraction * finalSol[5, myindex]
            # print newInitCond[17, myindex]
            ######run the ODE
            newInitCond = newInitCond.reshape(36 * numDep)
            tspanWeek = np.linspace(tspan_length -1 + totalWeeksVacCampaign, tspan_length -1 + totalWeeksVacCampaign + 1, 5)

            temp, info = odeint(choleraEqs11WithVaccinationNetwork, newInitCond, tspanWeek,
                        args=(params2,), full_output=True)

            temp2, info = odeint(choleraEqs11WithVaccinationNetwork, initCondDoNothing, tspanWeek,
                         args=(params2,), full_output=True)

            totalWeeksVacCampaign += 1
            #print tspan_length -1 + totalWeeksVacCampaign
            vacSol[totalWeeksVacCampaign, :] = temp[-1, :]
            doNothing[totalWeeksVacCampaign, :] = temp2[-1, :]



            finalSol = temp[-1, :].reshape(36, numDep)
            initCondDoNothing = temp2[-1, :]

    # print vacSol[-2:,:]
    #run the equations for numberOfYearsAfterVaccination:

    tspan_after_vaccination = range(tspan_length-1 + len(vacSol), tspan_length-1 + len(vacSol) + numberOfYearsAfterVaccination*52)
    initCondAfterVac = vacSol[-1,:]
    initCondAfterDoNothing = doNothing[-1,:]
    solAfterVac, info = odeint(choleraEqs11WithVaccinationNetwork, initCondAfterVac, tspan_after_vaccination,
                        args=(params2,), full_output=True)

    solAfterDoNothing, info = odeint(choleraEqs11WithVaccinationNetwork, initCondAfterDoNothing, tspan_after_vaccination,
                         args=(params2,), full_output=True)



    # compute number of cases during the vaccination campaign:

    cases_vac = computeNationalWeeklyCases(vacSol, cases[-1])
    cases_doNothing = computeNationalWeeklyCases(doNothing, cases[-1])
    cases_AfterVac = computeNationalWeeklyCases(solAfterVac, cases_vac[-1])
    cases_AfterDoNothing = computeNationalWeeklyCases(solAfterDoNothing, cases_doNothing[-1])


    casesVac = np.concatenate((cases_vac, cases_AfterVac))
    casesBaseline = np.concatenate((cases_doNothing, cases_AfterDoNothing))


    totalCasesBaseline = np.sum(casesBaseline)
    totalCasesVac = np.sum(casesVac)

    if totalCasesBaseline < 10 ** (-5):
        totalCasesBaseline = 10 ** (-5)

    if totalCasesVac < 10 ** (-5):
        totalCasesVac = 10 ** (-5)

    fractionReduction = (totalCasesBaseline - totalCasesVac) / (totalCasesBaseline)
    tspan_Vaccination_on = range(tspan_length - 1, tspan_length - 1 + len(vacSol) + numberOfYearsAfterVaccination * 52)
    # return cases
    return [tspan0,  tspan_Vaccination_on, cases, casesVac, casesBaseline, fractionReduction]


def simulateVaccinationScenario2Coverage95(params1, params2, extraParams, vaccination_tspan, numberOfYearsAfterVaccination, vacRecovered, mybreakingPoint):
    """run the equations for vaccinating individuals in numberOfYearsVaccination and run equations for
    numberOfYearsAfterVaccination afterwards.
    We will vaccinate 95% of the population with two doses, 1.67% with a single dose for Artibonite and Centre
    vacRecovered is a switch = 0 or 1, if 1, we will vaccinate the recovered people and move them to the recovered vaccinated class
    thereby assuming some sort of boost from the vaccine in recovered people.
    """


    depNames = ['artibonite', 'centre', 'grand_anse', 'nippes', 'nord', 'nord_est', 'nord_ouest', 'ouest', 'sud',
                'sud_est']

    #we vaccinate the depts with higher incidence first, and follow down along the list.
    # per table, the order for vaccinating is:
    departments_ordered = ['centre', 'artibonite', 'ouest', 'nord_ouest', 'nord', 'sud', 'nippes', 'nord_est', \
                           'sud_est', 'grand_anse']

    # oer table, the population in each of these departements (ordered)
    pop_ordered = np.array([746236, 1727524, 4029705, 728807, 1067177, 774976, 342525, 393967, 632601, 468301])
    haitiPop = np.sum(pop_ordered)
    pop_centre_artibonite = pop_ordered[0] + pop_ordered[1]

    # population under 5 (taken from HopkinsIID github), 2015 projection:
    pop_under_5 = 1288122
    percentage_under_5 = pop_under_5 / haitiPop  # 11.8%

    vaccination_tspan_weeks = int(vaccination_tspan * 52)

    deps_vaccinated = ['centre', 'artibonite']
    pop_deps_vaccinated = pop_ordered[0:2]


    # want to vaccinate 80 % in vaccination_tspan_weeks, so we need to vaccinate numVacWeek persons per week:
    numVacWeek = round(0.9667 * pop_centre_artibonite / (vaccination_tspan_weeks))
    # print (numVacWeek)

    # number of weeks to be spent vaccinating each that department:
    #this gives a vector with the number of weeks to be spent in each department so that at the end we vaccinated 80% of
    #the population in each department and we spend overall the numVacWeek weeks we want to spend
    numWeekPerDepartment = np.round(0.8 * pop_deps_vaccinated / numVacWeek)
    # print 'numWeekPerDepartment', numWeekPerDepartment
    real_vaccination_tspan_weeks = int(np.sum(numWeekPerDepartment))
    # print real_vaccination_tspan_weeks

    # fraction of people to vaccinate in each departement per week:
    fractionPerWeek = [numVacWeek / x for x in pop_deps_vaccinated]
    # print fractionPerWeek



    totalVaccines = 0
    totalWeeksVacCampaign = 0

    totalWeeksVacCampaignPerDept = np.zeros(10)
    totalVacGiven = 0

    #run baseline model:
    [tspan0, sol,  cases] = simulateBaselineInPieces(params1, params2, extraParams, mybreakingPoint)
    # print np.shape(sol)
    finalSol = np.copy(sol[-1,:]).reshape(36, numDep)
    initCondDoNothing = np.copy(sol[-1,:])


    vacSol = np.zeros((real_vaccination_tspan_weeks + 1, 36 * numDep))
    doNothing = np.zeros((real_vaccination_tspan_weeks + 1, 36 * numDep))

    vacSol[0, :] = sol[-1, :]
    doNothing[0,:]= sol[-1, :]
    # ######## VACCINATION     ########################
    # # march through each department to vaccinate for the number of weeks given in numWeekPerDepartment:
    finalSol = sol[-1,:].reshape(36, numDep)

    for ivals in range(2):

        numOfPulses = int(numWeekPerDepartment[ivals])
        # print numOfPulses
        myname = departments_ordered[ivals]
        # print myname
        myfraction = np.min([fractionPerWeek[ivals], 1])
        # print 'myfraction', myfraction
        myindex = depNames.index(myname)
        # print myindex



        # form vaccination multipliers:
        vacOneDoseOver5 = myfraction * (1.67 / 96.67) * (1 - percentage_under_5)
        vacTwoDosesOver5 = myfraction * (95. / 96.67) * (1 - percentage_under_5)
        vacOneDoseUnder5 = myfraction * (1.67 / 96.67) * (percentage_under_5)
        vacTwoDosesUnder5 = myfraction * (95. / 96.67) * (percentage_under_5)
        # print (vacOneDoseOver5 + vacTwoDosesOver5 + vacTwoDosesUnder5 + vacOneDoseUnder5) - myfraction

        for weeks in range(numOfPulses):
            # print weeks
        # print numOfPulses
            newInitCond = np.copy(finalSol)
            #move people from the susceptible class for that department to the vaccinated classes:
            newInitCond[0, myindex] = finalSol[0, myindex] - myfraction * finalSol[0, myindex]  #remove myfraction fraction of susceptibles
            # print newInitCond[0, myindex]
            newInitCond[6, myindex] = finalSol[6, myindex] + vacOneDoseOver5 * myfraction * finalSol[0, myindex]
            newInitCond[12, myindex] = finalSol[12, myindex] + vacTwoDosesOver5 * myfraction * finalSol[0, myindex]
            newInitCond[18, myindex] = finalSol[18, myindex] + vacOneDoseUnder5 * myfraction * finalSol[0, myindex]
            newInitCond[24, myindex] = finalSol[24, myindex] + vacTwoDosesUnder5 * myfraction * finalSol[0, myindex]
            # print newInitCond[12, myindex]
            if vacRecovered == 1:
                newInitCond[4, myindex] = finalSol[4, myindex] - vacTwoDosesOver5 * myfraction * finalSol[4, myindex]

                newInitCond[10, myindex] = finalSol[10, myindex] + vacOneDoseOver5 * myfraction * finalSol[4, myindex]
                newInitCond[16, myindex] = finalSol[16, myindex] + vacTwoDosesOver5 * myfraction * finalSol[4, myindex]
                newInitCond[22, myindex] = finalSol[22, myindex] + vacOneDoseUnder5 * myfraction * finalSol[4, myindex]
                newInitCond[28, myindex] = finalSol[28, myindex] + vacTwoDosesUnder5 * myfraction * finalSol[4, myindex]

                newInitCond[5, myindex] = finalSol[5, myindex] - vacTwoDosesOver5 * myfraction * finalSol[5, myindex]

                newInitCond[11, myindex] = finalSol[11, myindex] + vacOneDoseOver5 * myfraction * finalSol[5, myindex]
                newInitCond[17, myindex] = finalSol[17, myindex] + vacTwoDosesOver5 * myfraction * finalSol[5, myindex]
                newInitCond[23, myindex] = finalSol[23, myindex] + vacOneDoseUnder5 * myfraction * finalSol[5, myindex]
                newInitCond[29, myindex] = finalSol[29, myindex] + vacTwoDosesUnder5 * myfraction * finalSol[5, myindex]

            ######run the ODE
            newInitCond = newInitCond.reshape(36 * numDep)
            tspanWeek = np.linspace(tspan_length -1 + totalWeeksVacCampaign, tspan_length -1 + totalWeeksVacCampaign + 1, 5)

            temp, info = odeint(choleraEqs11WithVaccinationNetwork, newInitCond, tspanWeek,
                        args=(params2,), full_output=True)

            temp2, info = odeint(choleraEqs11WithVaccinationNetwork, initCondDoNothing, tspanWeek,
                         args=(params2,), full_output=True)

            totalWeeksVacCampaign += 1

            vacSol[totalWeeksVacCampaign, :] = temp[-1, :]
            doNothing[totalWeeksVacCampaign, :] = temp2[-1, :]



            finalSol = temp[-1, :].reshape(36, numDep)
            initCondDoNothing = temp2[-1, :]


    #run the equations for numberOfYearsAfterVaccination:

    tspan_after_vaccination = range(tspan_length-1 + len(vacSol), tspan_length-1 + len(vacSol) + numberOfYearsAfterVaccination*52)
    initCondAfterVac = vacSol[-1,:]
    initCondAfterDoNothing = doNothing[-1,:]
    solAfterVac, info = odeint(choleraEqs11WithVaccinationNetwork, initCondAfterVac, tspan_after_vaccination,
                        args=(params2,), full_output=True)

    solAfterDoNothing, info = odeint(choleraEqs11WithVaccinationNetwork, initCondAfterDoNothing, tspan_after_vaccination,
                         args=(params2,), full_output=True)

    # print totalWeeksVacCampaign

    # # compute number of cases during the vaccination campaign:

    cases_vac = computeNationalWeeklyCases(vacSol, cases[-1])
    cases_doNothing = computeNationalWeeklyCases(doNothing, cases[-1])
    cases_AfterVac = computeNationalWeeklyCases(solAfterVac, cases_vac[-1])
    cases_AfterDoNothing = computeNationalWeeklyCases(solAfterDoNothing, cases_doNothing[-1])


    casesVac = np.concatenate((cases_vac, cases_AfterVac))
    casesBaseline = np.concatenate((cases_doNothing, cases_AfterDoNothing))


    totalCasesBaseline = np.sum(casesBaseline)
    totalCasesVac = np.sum(casesVac)

    if totalCasesBaseline < 10 ** (-5):
        totalCasesBaseline = 10 ** (-5)

    if totalCasesVac < 10 ** (-5):
        totalCasesVac = 10 ** (-5)

    fractionReduction = (totalCasesBaseline - totalCasesVac) / (totalCasesBaseline)
    tspan_Vaccination_on = range(tspan_length - 1, tspan_length - 1 + len(vacSol) + numberOfYearsAfterVaccination * 52)

    return [tspan0,  tspan_Vaccination_on, cases, casesVac, casesBaseline, fractionReduction]
    # return [casesVac, fractionReduction]




def simulateVaccinationScenario4(params1, params2, extraParams, vaccination_tspan, numberOfYearsAfterVaccination, vacRecovered, mybreakingPoint):
    """run the equations for vaccinating individuals in numberOfYearsVaccination and run equations for
    numberOfYearsAfterVaccination afterwards.
    We will vaccinate 70% of the population with two doses, 10% with a single dose for Artibonite and Centre and Ouest
    vacRecovered is a switch = 0 or 1, if 1, we will vaccinate the recovered people and move them to the recovered vaccinated class
    thereby assuming some sort of boost from the vaccine in recovered people.
    """


    depNames = ['artibonite', 'centre', 'grand_anse', 'nippes', 'nord', 'nord_est', 'nord_ouest', 'ouest', 'sud',
                'sud_est']

    #we vaccinate the depts with higher incidence first, and follow down along the list.
    # per table, the order for vaccinating is:
    departments_ordered = ['centre', 'artibonite', 'ouest', 'nord_ouest', 'nord', 'sud', 'nippes', 'nord_est', \
                           'sud_est', 'grand_anse']

    # oer table, the population in each of these departements (ordered)
    pop_ordered = np.array([746236, 1727524, 4029705, 728807, 1067177, 774976, 342525, 393967, 632601, 468301])
    haitiPop = np.sum(pop_ordered)
    total_pops_vaccinated = np.sum(pop_ordered[0:3])

    # population under 5 (taken from HopkinsIID github), 2015 projection:
    pop_under_5 = 1288122
    percentage_under_5 = pop_under_5 / haitiPop  # 11.8%

    vaccination_tspan_weeks = int(vaccination_tspan * 52)

    deps_vaccinated = ['centre', 'artibonite', 'ouest']
    pop_deps_vaccinated = pop_ordered[0:3]


    # want to vaccinate 80 % in vaccination_tspan_weeks, so we need to vaccinate numVacWeek persons per week:
    numVacWeek = round(0.8 * total_pops_vaccinated / (vaccination_tspan_weeks))
    # print (numVacWeek)

    # number of weeks to be spent vaccinating each that department:
    #this gives a vector with the number of weeks to be spent in each department so that at the end we vaccinated 80% of
    #the population in each department and we spend overall the numVacWeek weeks we want to spend
    numWeekPerDepartment = np.round(0.8 * pop_deps_vaccinated / numVacWeek)
    # print 'numWeekPerDepartment', numWeekPerDepartment
    real_vaccination_tspan_weeks = int(np.sum(numWeekPerDepartment))
    # print real_vaccination_tspan_weeks

    # fraction of people to vaccinate in each departement per week:
    fractionPerWeek = [numVacWeek / x for x in pop_deps_vaccinated]
    # print fractionPerWeek


    totalVaccines = 0
    totalWeeksVacCampaign = 0

    totalWeeksVacCampaignPerDept = np.zeros(10)
    totalVacGiven = 0

    #run baseline model:
    [tspan0, sol,  cases] = simulateBaselineInPieces(params1, params2, extraParams, mybreakingPoint)
    # print np.shape(sol)
    finalSol = np.copy(sol[-1,:]).reshape(36, numDep)
    initCondDoNothing = np.copy(sol[-1,:])


    vacSol = np.zeros((real_vaccination_tspan_weeks + 1, 36 * numDep))
    doNothing = np.zeros((real_vaccination_tspan_weeks + 1, 36 * numDep))

    vacSol[0, :] = sol[-1, :]
    doNothing[0,:]= sol[-1, :]
    # ######## VACCINATION     ########################
    # # march through each department to vaccinate for the number of weeks given in numWeekPerDepartment:
    finalSol = sol[-1,:].reshape(36, numDep)

    for ivals in range(3):

        numOfPulses = int(numWeekPerDepartment[ivals])
        # print numOfPulses
        myname = departments_ordered[ivals]
        # print myname
        myfraction = np.min([fractionPerWeek[ivals], 1])
        # print 'myfraction', myfraction
        myindex = depNames.index(myname)
        # print myindex



        # form vaccination multipliers:
        vacOneDoseOver5 = myfraction * (1.0 / 8) * (1 - percentage_under_5)
        vacTwoDosesOver5 = myfraction * (7. / 8) * (1 - percentage_under_5)
        vacOneDoseUnder5 = myfraction * (1.0 / 8) * (percentage_under_5)
        vacTwoDosesUnder5 = myfraction * (7. / 8) * (percentage_under_5)

        # print (vacOneDoseOver5 + vacTwoDosesOver5 + vacTwoDosesUnder5 + vacOneDoseUnder5) - myfraction

        for weeks in range(numOfPulses):
            # print weeks
        # print numOfPulses
            newInitCond = np.copy(finalSol)
            #move people from the susceptible class for that department to the vaccinated classes:
            newInitCond[0, myindex] = finalSol[0, myindex] - myfraction * finalSol[0, myindex]  #remove myfraction fraction of susceptibles
            # print newInitCond[0, myindex]
            newInitCond[6, myindex] = finalSol[6, myindex] + vacOneDoseOver5 * myfraction * finalSol[0, myindex]
            newInitCond[12, myindex] = finalSol[12, myindex] + vacTwoDosesOver5 * myfraction * finalSol[0, myindex]
            newInitCond[18, myindex] = finalSol[18, myindex] + vacOneDoseUnder5 * myfraction * finalSol[0, myindex]
            newInitCond[24, myindex] = finalSol[24, myindex] + vacTwoDosesUnder5 * myfraction * finalSol[0, myindex]

            if vacRecovered == 1:
                newInitCond[4, myindex] = finalSol[4, myindex] - vacTwoDosesOver5 * myfraction * finalSol[4, myindex]

                newInitCond[10, myindex] = finalSol[10, myindex] + vacOneDoseOver5 * myfraction * finalSol[4, myindex]
                newInitCond[16, myindex] = finalSol[16, myindex] + vacTwoDosesOver5 * myfraction * finalSol[4, myindex]
                newInitCond[22, myindex] = finalSol[22, myindex] + vacOneDoseUnder5 * myfraction * finalSol[4, myindex]
                newInitCond[28, myindex] = finalSol[28, myindex] + vacTwoDosesUnder5 * myfraction * finalSol[4, myindex]

                newInitCond[5, myindex] = finalSol[5, myindex] - vacTwoDosesOver5 * myfraction * finalSol[5, myindex]

                newInitCond[11, myindex] = finalSol[11, myindex] + vacOneDoseOver5 * myfraction * finalSol[5, myindex]
                newInitCond[17, myindex] = finalSol[17, myindex] + vacTwoDosesOver5 * myfraction * finalSol[5, myindex]
                newInitCond[23, myindex] = finalSol[23, myindex] + vacOneDoseUnder5 * myfraction * finalSol[5, myindex]
                newInitCond[29, myindex] = finalSol[29, myindex] + vacTwoDosesUnder5 * myfraction * finalSol[5, myindex]

            ######run the ODE
            newInitCond = newInitCond.reshape(36 * numDep)
            tspanWeek = np.linspace(tspan_length -1 + totalWeeksVacCampaign, tspan_length -1 + totalWeeksVacCampaign + 1, 5)

            temp, info = odeint(choleraEqs11WithVaccinationNetwork, newInitCond, tspanWeek,
                        args=(params2,), full_output=True)

            temp2, info = odeint(choleraEqs11WithVaccinationNetwork, initCondDoNothing, tspanWeek,
                         args=(params2,), full_output=True)

            totalWeeksVacCampaign += 1
            #print tspan_length -1 + totalWeeksVacCampaign
            vacSol[totalWeeksVacCampaign, :] = temp[-1, :]
            doNothing[totalWeeksVacCampaign, :] = temp2[-1, :]



            finalSol = temp[-1, :].reshape(36, numDep)
            initCondDoNothing = temp2[-1, :]

    # print vacSol[-2:,:]
    #run the equations for numberOfYearsAfterVaccination:

    tspan_after_vaccination = range(tspan_length-1 + len(vacSol), tspan_length-1 + len(vacSol) + numberOfYearsAfterVaccination*52)
    initCondAfterVac = vacSol[-1,:]
    initCondAfterDoNothing = doNothing[-1,:]
    solAfterVac, info = odeint(choleraEqs11WithVaccinationNetwork, initCondAfterVac, tspan_after_vaccination,
                        args=(params2,), full_output=True)

    solAfterDoNothing, info = odeint(choleraEqs11WithVaccinationNetwork, initCondAfterDoNothing, tspan_after_vaccination,
                         args=(params2,), full_output=True)

    # print totalWeeksVacCampaign

    # # compute number of cases during the vaccination campaign:

    cases_vac = computeNationalWeeklyCases(vacSol, cases[-1])
    cases_doNothing = computeNationalWeeklyCases(doNothing, cases[-1])
    cases_AfterVac = computeNationalWeeklyCases(solAfterVac, cases_vac[-1])
    cases_AfterDoNothing = computeNationalWeeklyCases(solAfterDoNothing, cases_doNothing[-1])


    casesVac = np.concatenate((cases_vac, cases_AfterVac))
    casesBaseline = np.concatenate((cases_doNothing, cases_AfterDoNothing))

    totalCasesBaseline = np.sum(casesBaseline)
    totalCasesVac = np.sum(casesVac)

    if totalCasesBaseline < 10 ** (-5):
        totalCasesBaseline = 10 ** (-5)

    if totalCasesVac < 10 ** (-5):
        totalCasesVac = 10 ** (-5)

    fractionReduction = (totalCasesBaseline - totalCasesVac) / (totalCasesBaseline)
    tspan_Vaccination_on = range(tspan_length - 1, tspan_length - 1 + len(vacSol) + numberOfYearsAfterVaccination * 52)

    return [tspan0,  tspan_Vaccination_on, cases, casesVac, casesBaseline, fractionReduction]





if __name__ == "__main__":
    # start = time.time()


    today = time.strftime("%d%b%Y", time.localtime())
    #To run in the cluster
    index = 1#os.environ['SLURM_ARRAY_TASK_ID']
    # folder = os.environ['SLURM_JOB_NAME']


    #Uncomment the following lines if we are running a parameter set from the sensitivity analysis matrix:
    # filename1 = 'mainResults2019AllInfections/matrixOfParamsForSensitivityAnalysis_width0.25_fracSusWidth0.107May2019.pickle'
    # f = open(filename1, 'r')
    # mymat = (pickle.load(f))
    # # mymat = res[1]
    # f.close()
    # #
    # print np.shape(mymat)
    # #
    # fullSetOfParams = mymat[int(index),:]

    #run the upper and lower bounds for sensitivity analysis with indices 1000 and 1001
    # betaLB, betaWLB, muLB, mu2LB, fracSusLB = [7.44640361e-07, 3.02483058e-02, 1.25924473e+01, 1.03283259e+00, 6.75000000e-01]
    # betaUB, betaWUB, muUB, mu2UB, fracSusUB = [1.24106727e-06, 5.04138430e-02, 1.25924473e+06, 6.45520370e+00, 8.25000000e-01]



    #run the best params:
    filename1 = 'mainResultsMarch2019/bestParamsVector25Mar2019.pickle'
    f = open(filename1, 'r')
    fullSetOfParams = (pickle.load(f))
    f.close()
    # fullSetOfParams = [betaLB, betaWLB, muLB, betaLB, betaWLB, mu2LB, fracSusLB] #lower bound, index 1000
    # fullSetOfParams = [betaUB, betaWUB, muUB, betaUB, betaWUB,mu2UB, fracSusUB]  #upper bound, index 1001

    print fullSetOfParams
    beta1, betaW1, mu1 = fullSetOfParams[0:3]
    beta2, betaW2, mu2, fracSus = fullSetOfParams[3:]

    print [beta1, betaW1, mu1]
    print [beta2, betaW2, mu2, fracSus]

    #load the data:
    depNames = ['artibonite', 'centre', 'grand_anse', 'nippes', 'nord', 'nord_est', 'nord_ouest', 'ouest', 'sud',
                'sud_est']

    mylabels = ['V1', 'E1', 'I1', 'A1', 'R1', 'RA1', 'V2', 'E2', 'I2', 'A2', 'R2', 'RA2']
    #####load the data from the parameters file:
    # getVarFromFile('extraParams.py')
    filename = 'extraParamsData_to_01_2019.py'
    f = open(filename)
    global data
    data = imp.load_source('data', '', f)
    f.close()



    fullData = data.cases[:, 1:11]
    totalData = data.cases[:, 11]
    # print np.shape(totalData)
    tspan_length = len(fullData)


    mybreakingPoint = 176
    fullDataBeforeBP = fullData[:mybreakingPoint, :mybreakingPoint]

    totalDataBeforeBP = totalData[:mybreakingPoint]

    tspan_length1 = len(fullDataBeforeBP)
    nInfectedBeforeBP = []

    for depVal in range(1, 11):
        dep_data = data.cases[:, depVal]
        nInfectedBeforeBP.append(dep_data[np.nonzero(dep_data)[0][0]])
    nInfectedBeforeBP = np.array(nInfectedBeforeBP)


    fullDataAfterBP = fullData[mybreakingPoint:, :]
    totalDataAfterBP = totalData[mybreakingPoint:]
    tspan_length2 = tspan_length1 + len(fullDataAfterBP)
    tspanAfterBP = range(tspan_length1, tspan_length2)



    reporting_rate = 0.2


    tau, vrate, wRate = [2.0, 1 * 10 ** (-12), 5]

    beta_A1 = data.red_beta_weekly*beta1
    mu_A1 = data.red_mu_weekly*mu1

    numDep = 10
    aseason = 0.4
    TravelMat = formTravelMat1(numDep, data.rlen, tau, vrate, data.totalPop)

    # VE Scenario 1:
    [VE2, VE1] = data.VEmatrix[:, 0]
    # print VE2, VE1
    [theta2, theta1] = [1 - VE2, 1 - VE1]

    vac_children = 0.4688 #reduction in vaccine effectiveness in children

    params1 = [aseason, beta1, beta_A1, betaW1, data.delta, data.gamma_weekly, data.gammaA_weekly, data.gammaE_weekly,
              data.k_weekly, data.m_weekly, mu1, mu_A1, numDep,
              data.omega1_weekly, data.omega2_weekly, data.pseason_weekly, data.sigma_weekly,
              theta1, vac_children * theta1, theta2, theta2 * vac_children,
              TravelMat, data.V_weekly, wRate, data.waterMatUpdated]



    beta_A2 = data.red_beta_weekly*beta2
    mu_A2 = data.red_mu_weekly*mu2



    nInfectedAfterBP = np.array(fullDataAfterBP[0, :])
    params2 = [aseason, beta2, beta_A2, betaW2, data.delta, data.gamma_weekly, data.gammaA_weekly, data.gammaE_weekly,
              data.k_weekly, data.m_weekly, mu2, mu_A2, numDep,
              data.omega1_weekly, data.omega2_weekly, data.pseason_weekly, data.sigma_weekly,
              theta1, vac_children* theta1, theta2, vac_children*theta2,
              TravelMat, data.V_weekly, wRate, data.waterMatUpdated]


    extraParamsBis = [fracSus, data.k_weekly, data.totalPop, nInfectedBeforeBP, nInfectedAfterBP, reporting_rate,
                   tspan_length]


    mainScenarioID1 = simulateVaccination(params1, params2, extraParamsBis, 2, 8, 0, mybreakingPoint)
    mainScenarioID2 = simulateVaccinationScenario2(params1, params2, extraParamsBis, 2, 8, 0, mybreakingPoint)
    mainScenarioID3 = simulateVaccination(params1, params2, extraParamsBis, 5, 5, 0, mybreakingPoint)
    mainScenarioID4 = simulateVaccinationScenario4(params1, params2, extraParamsBis, 2, 8, 0, mybreakingPoint)
    mainScenarioID25 =simulateVaccinationCoverage95(params1, params2, extraParamsBis, 2, 8, 0, mybreakingPoint)


    #store all the results in a list
    main = [mainScenarioID1, mainScenarioID2, mainScenarioID3, mainScenarioID4, mainScenarioID25]



    # # dump the results of the main run in a pickle file for later use:
    # mymatfileName = 'mainResults2019AllInfections/sensitivityAnalysis/mainResults_1to4and25_width0.25_fracSusWidth0.1' + today + '_' +  str(index) + '.pickle'
    # myfile = open(mymatfileName, 'wb')
    # pickle.dump(main, myfile)
    # myfile.close()
    # print mymatfileName


    # # #dump the results of the main run in a pickle file for later use:
    # mymatfileName = 'mainResults2019AllInfections/sensitivityAnalysis/mainResults_1to4and25' + today + '_' +  'bestParamsMean' + '.pickle'
    # myfile = open(mymatfileName, 'wb')
    # pickle.dump(main, myfile)
    # myfile.close()


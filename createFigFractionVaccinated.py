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
from choleraEqs import  choleraEqs11WithVaccinationNetwork
# from functionsCholeraProject import binningWeeklyCases, formTravelMat1, binningWeeklyCasesMat
from functools import partial
import seaborn as sns
# from pyDOE import *
# import pandas as pd
import timeit
import functools
import random
import os
mycolors = sns.color_palette("hls", 5)

#create a graph with the number of vaccines allocated for each scenario. The code for these functions was copied from
#the file: simulateHaitiFullTimeInPiecesAllVaccinationScenariosSingleParameterVectorAllInfections.py



def numberOfDosesNational70_10(vaccination_tspan):
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

    tspan = range(real_vaccination_tspan_weeks)

    # fraction of people to vaccinate in each departement per week:
    fractionPerWeek = [numVacWeek / x for x in pop_ordered]
    # print fractionPerWeek




    totalWeeksVacCampaign = 0



    vacOneDose = np.zeros(10*52)
    vacTwoDoses = np.zeros(10*52)

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



        for weeks in range(numOfPulses):
            # print weeks
            vacOneDose[totalWeeksVacCampaign] = (vacOneDoseOver5 + vacOneDoseUnder5)*pop_ordered[ivals]
            vacTwoDoses[[totalWeeksVacCampaign]] = (vacTwoDosesOver5 + vacTwoDosesUnder5)*pop_ordered[ivals]

            totalWeeksVacCampaign += 1


    return [tspan, np.cumsum(vacOneDose), np.cumsum(vacTwoDoses)]




def numberOfDosesNational95(vaccination_tspan):
    """
    vaccinate 95% with two doses and 1.667% with a single dose
    :param vaccination_tspan: number of years to perform vaccination campaign
    :return:
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
    #this gives a vector with the number of weeks to be spent in each department so that at the end we vaccinated 95% of
    #the population in each department with two doses and 1.67% with one and we spend overall the numVacWeek weeks we want to spend
    numWeekPerDepartment = np.round(0.9667 * pop_ordered / numVacWeek)


    # fraction of people to vaccinate in each departement per week:
    fractionPerWeek = [numVacWeek / x for x in pop_ordered]
    # print fractionPerWeek

    tspan = range(10*52)

    totalWeeksVacCampaign = 0

    vacOneDose = np.zeros(10*52)
    vacTwoDoses = np.zeros(10*52)

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
            vacOneDose[totalWeeksVacCampaign] = (vacOneDoseOver5 + vacOneDoseUnder5)*pop_ordered[ivals]
            vacTwoDoses[[totalWeeksVacCampaign]] = (vacTwoDosesOver5 + vacTwoDosesUnder5)*pop_ordered[ivals]

            totalWeeksVacCampaign += 1


    return [tspan, np.cumsum(vacOneDose), np.cumsum(vacTwoDoses)]




def numberOfDosesTwoDepartments70_10(vaccination_tspan):
    """
    Vaccinate two departments Centre and Antibonite with 70% two doses and 10% one dose.
    :param vaccination_tspan: amount of time (years) to spend vaccinating.
    :return:
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
    print (0.7*pop_centre_artibonite)/haitiPop
    print (0.1 * pop_centre_artibonite) / haitiPop
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



    tspan = range(10*52)

    # fraction of people to vaccinate in each departement per week:
    fractionPerWeek = [numVacWeek / x for x in pop_deps_vaccinated]
    # print fractionPerWeek




    totalWeeksVacCampaign = 0



    vacOneDose = np.zeros(10*52)
    vacTwoDoses = np.zeros(10*52)

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
            vacOneDose[totalWeeksVacCampaign] = (vacOneDoseOver5 + vacOneDoseUnder5)*pop_ordered[ivals]
            vacTwoDoses[[totalWeeksVacCampaign]] = (vacTwoDosesOver5 + vacTwoDosesUnder5)*pop_ordered[ivals]

            totalWeeksVacCampaign += 1


    return [tspan, np.cumsum(vacOneDose), np.cumsum(vacTwoDoses)]



def numberOfDosesThreeDepartments70_10(vaccination_tspan):
    """
      Vaccinate three departments Centre, Ouest, and Antibonite with 70% two doses and 10% one dose.
      :param vaccination_tspan: amount of time (years) to spend vaccinating.
      :return:
      """
    depNames = ['artibonite', 'centre', 'grand_anse', 'nippes', 'nord', 'nord_est', 'nord_ouest', 'ouest', 'sud',
                'sud_est']

    # we vaccinate the depts with higher incidence first, and follow down along the list.
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
    print (0.7*np.sum(pop_deps_vaccinated))/haitiPop
    print (0.1 * np.sum(pop_deps_vaccinated)) / haitiPop
    # want to vaccinate 80 % in vaccination_tspan_weeks, so we need to vaccinate numVacWeek persons per week:
    numVacWeek = round(0.8 * total_pops_vaccinated / (vaccination_tspan_weeks))
    # print (numVacWeek)

    # number of weeks to be spent vaccinating each that department:
    # this gives a vector with the number of weeks to be spent in each department so that at the end we vaccinated 80% of
    # the population in each department and we spend overall the numVacWeek weeks we want to spend
    numWeekPerDepartment = np.round(0.8 * pop_deps_vaccinated / numVacWeek)

    tspan = range(10*52)

    # fraction of people to vaccinate in each departement per week:
    fractionPerWeek = [numVacWeek / x for x in pop_deps_vaccinated]
    # print fractionPerWeek




    totalWeeksVacCampaign = 0



    vacOneDose = np.zeros(52*10)
    vacTwoDoses = np.zeros(52*10)

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
            vacOneDose[totalWeeksVacCampaign] = (vacOneDoseOver5 + vacOneDoseUnder5)*pop_ordered[ivals]
            vacTwoDoses[[totalWeeksVacCampaign]] = (vacTwoDosesOver5 + vacTwoDosesUnder5)*pop_ordered[ivals]

            totalWeeksVacCampaign += 1


    return [tspan, np.cumsum(vacOneDose), np.cumsum(vacTwoDoses)]



if __name__ == "__main__":
    [tspan0, one0, two0]= numberOfDosesNational95(2)
    [tspan, one, two]= numberOfDosesThreeDepartments70_10(2)
    [tspan1, one1, two1] = numberOfDosesNational70_10(5)
    [tspan2, one2, two2] = numberOfDosesNational70_10(2)
    [tspan3, one3, two3] = numberOfDosesTwoDepartments70_10(2)
    pop_ordered = np.array([746236, 1727524, 4029705, 728807, 1067177, 774976, 342525, 393967, 632601, 468301])
    haitiPop = np.sum(pop_ordered)


    ones = [one0, one, one1, one2, one3]
    twos = [two0, two, two1, two2, two3]
    mylabels = ['95% coverage, Fast national', '3 departments', 'Fast national','Slow national', '2 departments' ]
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(1, 1, 1)
    for ivals in xrange(5):
        plt.plot(tspan0, ones[ivals]/haitiPop, '--',  color= mycolors[ivals],linewidth=2)
        plt.plot(tspan0, twos[ivals]/haitiPop,  label = mylabels[ivals], color= mycolors[ivals],linewidth=2)

    plt.legend(loc=1, fontsize=12)
    plt.ylabel('Fraction of the population vaccinated', fontweight='bold', fontsize=12)
    plt.ylim([0,1.05])
    myticks = range(2019, 2030)
    ax.set_xticks(range(0,11*52, 52))
    ax.set_xticklabels(myticks, fontweight='bold', rotation=90, fontsize=12)
    plt.xlim([-1, 10*52])

    # plt.savefig('HaitiMultimodelingOCV/figuresFredHutch/vaccineDistribution.pdf')
    plt.show()
from __future__ import division
import numpy as np
import time
# import imp
import pickle
import pandas as pd
# import pymc as pc
import sys, getopt
import csv
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from scipy import optimize
# from choleraEqs import choleraEqs, choleraEqs2, choleraEqs3, choleraEqs2bis, choleraEqs5WithoutVaccinationSingleGroup
from functools import partial
# import seaborn as sns
from pyDOE import *
import os
# from readResultsFitSinglePop import getMatchingFiles


def extractParametersFromPickleFiles(depName, mydate):
    results = np.zeros((1000, 5))
    #open the folder and extract for each file the line with the results and store it in a big matrix
    for ivals in range(1000):

        myfullpath = 'leastSquares/' + depName + '/' + depName + '_' + str(ivals)+ 'Iterations_' + mydate + '.pickle'
        if os.path.isfile(myfullpath):

            # print myfullpath
            f = open(myfullpath, 'r')
            temp = pickle.load(f)
            # # matchingFiles = getMatchingFiles(dir=myfoldername, regex=myfullpath, full_names=True)[0]
            results[ivals,:] = temp[1]
            f.close()
        else:
            print ivals
    # print results

    # store the results in a pickle file:
    myresultsFileName = 'leastSquaresResults/' + depName + '_resultsMatrix' + mydate + '.pickle'
    myfile = open(myresultsFileName, 'wb')
    pickle.dump(results, myfile)
    return results



def extractParametersFromPickleFiles2(depName, mydate):
    results = np.zeros((1000, 6))
    #open the folder and extract for each file the line with the results and store it in a big matrix
    for ivals in range(1000):

        # myfullpath = 'leastSquares/' + depName + '/' + depName + '_NewDataSet' + str(ivals)+ 'Iterations_' + mydate + '.pickle'
        myfullpath = 'leastSquares/' + depName + '/' + depName + '_NewDataSetOneYearSeasonality' + str(ivals) + 'Iterations_' + mydate + '.pickle'
        if os.path.isfile(myfullpath):

            # print myfullpath
            f = open(myfullpath, 'r')
            temp = pickle.load(f)
            # print temp
            # # matchingFiles = getMatchingFiles(dir=myfoldername, regex=myfullpath, full_names=True)[0]
            results[ivals,0:5] = temp[1]
            results[ivals, 5] = temp[-1]
            f.close()
        else:
            print ivals
    # print results

    # store the results in a pickle file:
    # myresultsFileName = 'leastSquaresResults/' + depName + '_resultsMatrix' + mydate + '.pickle'
    myresultsFileName = 'leastSquaresResults/' + depName + '_resultsMatrixOneYearSeasonality' + mydate + '.pickle'
    print myresultsFileName
    myfile = open(myresultsFileName, 'wb')
    pickle.dump(results, myfile)
    return results


def extractParametersFromPickleFilesICfit(depName, mydate):
    results = np.zeros((1000, 7))
    #open the folder and extract for each file the line with the results and store it in a big matrix
    for ivals in range(1000):

        myfullpath = 'leastSquares/' + depName + '/' + depName + 'ICfit_' + str(ivals)+ 'Iterations_' + mydate + '.pickle'
        if os.path.isfile(myfullpath):

            # print myfullpath
            f = open(myfullpath, 'r')
            temp = pickle.load(f)
            # # matchingFiles = getMatchingFiles(dir=myfoldername, regex=myfullpath, full_names=True)[0]
            results[ivals,0:6] = temp[1]
            results[ivals, 5] = temp[-1]
            f.close()
        else:
            print ivals
    # print results

    # store the results in a pickle file:
    myresultsFileName = 'leastSquaresResults/' + depName + 'ICfit_resultsMatrix' + mydate + '.pickle'
    myfile = open(myresultsFileName, 'wb')
    pickle.dump(results, myfile)
    return results


def extractResultsListPickleFiles(depName, mydate):
    results = []
    #open the folder and extract for each file the line with the results and store it in a big matrix
    for ivals in range(1000):

        # myfullpath = 'leastSquares/' + depName + '/' + depName + '_' + str(ivals) + 'Iterations_' + mydate + '.pickle'
        myfullpath = 'leastSquares/' + depName + '/' + depName + '_NewDataSet' + str(
            ivals) + 'Iterations_' + mydate + '.pickle'
        if os.path.isfile(myfullpath):

            # print myfullpath
            f = open(myfullpath, 'r')
            temp = pickle.load(f)
            # # matchingFiles = getMatchingFiles(dir=myfoldername, regex=myfullpath, full_names=True)[0]
            results.append(temp) #= temp[1]
            #results[ivals, 5] = temp[-1]
            f.close()
        else:
            print ivals
    # print results

    # store the results in a pickle file:
    myresultsFileName = 'leastSquaresResults/' + depName + 'resultsList_' + mydate + '.pickle'
    myfile = open(myresultsFileName, 'wb')
    pickle.dump(results, myfile)
    return results

def extractParametersFromPickleFiles3(depName, mydate):
    results = np.zeros((1000, 18))
    counter = 0
    #open the folder and extract for each file the line with the results and store it in a big matrix
    for ivals in range(1000):

        myfullpath = 'leastSquares/' + depName + '/' + depName + '_' + str(ivals)+ 'Iterations_' + mydate + '.pickle'
        print myfullpath
        if os.path.isfile(myfullpath):

            # print myfullpath
            f = open(myfullpath, 'r')
            temp = pickle.load(f)
            # # matchingFiles = getMatchingFiles(dir=myfoldername, regex=myfullpath, full_names=True)[0]
            results[ivals,0:17] = temp[1]
            results[ivals, 17] = temp[-1]
            # print results[ivals, :]
            f.close()
        else:

            print 'error for iteration',  ivals
            counter = counter +1
    print results
    print counter
    # store the results in a pickle file:
    myresultsFileName = 'leastSquaresResults/' + depName + '_resultsMatrix' + mydate + '.pickle'
    myfile = open(myresultsFileName, 'wb')
    pickle.dump(results, myfile)
    return results

def extractParametersFromPickleFilesLessParameters(depName, mydate):
    results = np.zeros((1000, 14))
    #open the folder and extract for each file the line with the results and store it in a big matrix
    for ivals in range(1000):

        # myfullpath = 'leastSquares/' + depName + '/' + depName + '_NewDataSet' + str(ivals)+ 'Iterations_' + mydate + '.pickle'
        myfullpath = 'leastSquares/' + depName + '/' + depName + '_' + str(ivals) + 'Iterations_' + mydate + '.pickle'
        print myfullpath
        if os.path.isfile(myfullpath):

            # print myfullpath
            f = open(myfullpath, 'r')
            temp = pickle.load(f)
            # print temp
            # # matchingFiles = getMatchingFiles(dir=myfoldername, regex=myfullpath, full_names=True)[0]
            results[ivals,0:13] = temp[1]
            results[ivals, 13] = temp[-1]
            f.close()
        else:
            print ivals
    # print results

    # store the results in a pickle file:
    # myresultsFileName = 'leastSquaresResults/' + depName + '_resultsMatrix' + mydate + '.pickle'
    myresultsFileName = 'leastSquaresResults/' + depName + '_resultsMatrixOneYearSeasonality' + mydate + '.pickle'
    print myresultsFileName
    myfile = open(myresultsFileName, 'wb')
    pickle.dump(results, myfile)
    return results



def extractParametersFromPickleFilesNew(depName, mydate):
    results = np.zeros((1000, 6))
    #open the folder and extract for each file the line with the results and store it in a big matrix
    for ivals in range(1000):

        # myfullpath = 'leastSquares/' + depName + '/' + depName + '_NewDataSet' + str(ivals)+ 'Iterations_' + mydate + '.pickle'
        # myfullpath = 'leastSquaresNew/' + depName + '/' + depName + '_weeklyParams_NewDataSetOneYearSeasonality' + \
        #              str(ivals) + 'Iterations_' + mydate + '.pickle'
        myfullpath = 'leastSquaresNew/' + depName + '/' + depName + '_weeklyParams_NewDataSet6monthsSeasonality' + \
                     str(ivals) + 'Iterations_' + mydate + '.pickle'
        if os.path.isfile(myfullpath):

            # print myfullpath
            f = open(myfullpath, 'r')
            temp = pickle.load(f)
            # print temp
            # # matchingFiles = getMatchingFiles(dir=myfoldername, regex=myfullpath, full_names=True)[0]
            results[ivals,0:5] = temp[1]
            results[ivals, 5] = temp[-1]
            f.close()
        else:
            print ivals
    # print results

    # store the results in a pickle file:
    # myresultsFileName = 'leastSquaresResults/' + depName + '_resultsMatrix' + mydate + '.pickle'
    # myresultsFileName = 'leastSquaresResultsNew/' + depName + '_weeklyParams_resultsMatrixOneYearSeasonality' + mydate + '.pickle'
    myresultsFileName = 'leastSquaresResultsNew/' + depName + '_weeklyParams_resultsMatrix6monthsSeasonality' + mydate + '.pickle'
    print myresultsFileName
    myfile = open(myresultsFileName, 'wb')
    pickle.dump(results, myfile)
    return results


def extractParametersFromPickleFilesNew3Params(depName, mydate):
    results = np.zeros((1000, 4))
    #open the folder and extract for each file the line with the results and store it in a big matrix
    for ivals in range(1000):

        # myfullpath = 'leastSquares/' + depName + '/' + depName + '_NewDataSet' + str(ivals)+ 'Iterations_' + mydate + '.pickle'
        # myfullpath = 'leastSquaresNew/' + depName + '/' + depName + '_weeklyParams_NewDataSetOneYearSeasonality' + \
        #              str(ivals) + 'Iterations_' + mydate + '.pickle'
        # myfullpath = 'leastSquaresNew/' + depName + '/' + depName + '_' + \
        #              str(ivals) + 'Iterations_' + mydate + '.pickle'
        myfullpath = 'leastSquaresNationalFit/' + depName + '/' + depName + 'Betas2_' + \
                     str(ivals) + 'Iterations_' + mydate + '.pickle'

        print myfullpath
        if os.path.isfile(myfullpath):

            # print myfullpath
            f = open(myfullpath, 'r')
            temp = pickle.load(f)
            print temp
            # # matchingFiles = getMatchingFiles(dir=myfoldername, regex=myfullpath, full_names=True)[0]
            results[ivals,0:3] = temp[1]
            results[ivals, 3] = temp[-1]
            f.close()
        else:
            print ivals
    # print results

    # store the results in a pickle file:
    # myresultsFileName = 'leastSquaresResults/' + depName + '_resultsMatrix' + mydate + '.pickle'
    # myresultsFileName = 'leastSquaresResultsNew/' + depName + '_weeklyParams_resultsMatrixOneYearSeasonality' + mydate + '.pickle'
    myresultsFileName = 'leastSquaresResultsNew/' + depName + '_resultsMatrix3paramsBetas2' + mydate + '.pickle'
    print myresultsFileName
    myfile = open(myresultsFileName, 'wb')
    pickle.dump(results, myfile)
    return results


def extractParametersFromPickleFilesFixedRepRate(depName, mydate):
    results = np.zeros((1000, 6))
    #open the folder and extract for each file the line with the results and store it in a big matrix
    for ivals in range(1000):

        # myfullpath = 'leastSquares/' + depName + '/' + depName + '_NewDataSet' + str(ivals)+ 'Iterations_' + mydate + '.pickle'
        myfullpath = 'leastSquaresRepRateFixed/' + depName + '/' + depName  +  '_weeklyParams_NewDataSet6monthsSeasonality' + \
                     str(ivals) + 'Iterations_' + mydate + '.pickle'
        print myfullpath
        if os.path.isfile(myfullpath):

            # print myfullpath
            f = open(myfullpath, 'r')
            temp = pickle.load(f)
            # print temp
            # # matchingFiles = getMatchingFiles(dir=myfoldername, regex=myfullpath, full_names=True)[0]
            results[ivals,0:5] = temp[1]
            results[ivals, 5] = temp[-1]
            f.close()
        else:
            print ivals
    # print results

    # store the results in a pickle file:
    # myresultsFileName = 'leastSquaresResults/' + depName + '_resultsMatrix' + mydate + '.pickle'
    myresultsFileName = 'leastSquaresResultsNew/' + depName +'_RepRateFixed_' +  '_resultsMatrix6monthsSeasonality' + mydate + '.pickle'
    print myresultsFileName
    myfile = open(myresultsFileName, 'wb')
    pickle.dump(results, myfile)
    return results

def extractParametersFromPickleFilesFixedRepRateNewModel2019(depName, mydate, reportingRate):
    results = np.zeros((1000, 6))
    #open the folder and extract for each file the line with the results and store it in a big matrix
    for ivals in range(1000):

        # myfullpath = 'leastSquares/' + depName + '/' + depName + '_NewDataSet' + str(ivals)+ 'Iterations_' + mydate + '.pickle'
        # myfullpath = 'leastSquares2019RepRateFixed/' + depName + '/' + depName  +  '_3params_reportingRateOf_' + \
        #              reportingRate + '_' + str(ivals) + 'Iterations_' + mydate + '.pickle'

        myfullpath = 'leastSquares2016on/' + depName + '/' + depName  +  '_3params_reportingRateOf_' + \
                     reportingRate + '_' + str(ivals) + 'Iterations_' + mydate + '.pickle'

        # myfullpath = 'leastSquares2019RepRateFixed/' + depName + '/' + depName + '_3params_' + str(ivals) + 'Iterations_' + mydate + '.pickle'

        # print myfullpath
        if os.path.isfile(myfullpath):

            # print myfullpath
            f = open(myfullpath, 'r')
            temp = pickle.load(f)
            # print temp
            # # matchingFiles = getMatchingFiles(dir=myfoldername, regex=myfullpath, full_names=True)[0]
            results[ivals,0:5] = temp[1]
            results[ivals, 5] = temp[-1]
            f.close()
        else:
            print ivals
    # print results

    # store the results in a pickle file:
    # myresultsFileName = 'leastSquaresResults/' + depName + '_resultsMatrix' + mydate + '.pickle'
    myresultsFileName = 'leastSquaresResults2016on/' + depName + '_resultsMatrix_3params_reportingRateOf_' + \
                     reportingRate + mydate + '.pickle'
    print myresultsFileName
    myfile = open(myresultsFileName, 'wb')
    pickle.dump(results, myfile)
    return results

def extractParametersFromPickleFilesFixedRepRateAllHaitiNewModel2019(depName, mydate, reportingRate):
    results = np.zeros((1000, 6))
    #open the folder and extract for each file the line with the results and store it in a big matrix
    for ivals in range(0,1000):

        # myfullpath = 'leastSquares2019/' + depName + '/' + depName + '_NewDataSet' + str(ivals)+ 'Iterations_' + mydate + '.pickle'
        # myfullpath = 'leastSquaresNationalFit/' + depName + '/' + depName  +  '_3paramsConstrainedM2_reportingRateOf_' + \
        #              reportingRate + '_' + str(ivals) + 'Iterations_' + mydate + '.pickle'

        myfullpath = 'leastSquaresNationalFit/' + depName + '/' + depName  +  '_3params_reportingRateOf_' + \
                     reportingRate + '_' + str(ivals) + 'Iterations_' + mydate + '.pickle'




        print myfullpath
        if os.path.isfile(myfullpath):

            # print myfullpath
            f = open(myfullpath, 'r')
            temp = pickle.load(f)
            print temp
            # # matchingFiles = getMatchingFiles(dir=myfoldername, regex=myfullpath, full_names=True)[0]
            results[ivals,0:5] = temp[1]
            results[ivals, 5] = temp[-1]
            f.close()
        else:
            print ivals
    # print results

    # store the results in a pickle file:
    # myresultsFileName = 'leastSquaresResults/' + depName + '_resultsMatrix' + mydate + '.pickle'
    myresultsFileName = 'leastSquaresResults2019/' + depName + '_resultsMatrix_3params_reportingRateOf_' + \
                     reportingRate + mydate + '.pickle'
    # print myresultsFileName
    myfile = open(myresultsFileName, 'wb')
    pickle.dump(results, myfile)
    return results


def extractParametersFromPickleFilesFixedRepRate6Params2019(depName, mydate):
    results = np.zeros((1000, 7))
    #open the folder and extract for each file the line with the results and store it in a big matrix
    for ivals in range(0,1000):

        # myfullpath = 'leastSquares2019/' + depName + '/' + depName + '_NewDataSet' + str(ivals)+ 'Iterations_' + mydate + '.pickle'
        # myfullpath = 'leastSquaresNationalFit/' + depName + '/' + depName  +  '_3paramsConstrainedM2_reportingRateOf_' + \
        #              reportingRate + '_' + str(ivals) + 'Iterations_' + mydate + '.pickle'

        myfullpath = 'leastSquaresNationalFit/' + depName + '/' + depName  + '_'+ \
                      str(ivals) + 'Iterations_' + mydate + '.pickle'

        print myfullpath
        if os.path.isfile(myfullpath):

            # print myfullpath
            f = open(myfullpath, 'r')
            temp = pickle.load(f)
            print temp
            # # matchingFiles = getMatchingFiles(dir=myfoldername, regex=myfullpath, full_names=True)[0]
            results[ivals,0:6] = temp[1]
            results[ivals, 6] = temp[-1]
            f.close()
        else:
            print ivals
    # print results

    # store the results in a pickle file:
    # myresultsFileName = 'leastSquaresResults/' + depName + '_resultsMatrix' + mydate + '.pickle'
    myresultsFileName = 'leastSquaresResults2019/' + depName + '_resultsMatrix_6params_' + \
                     mydate + '.pickle'
    # print myresultsFileName
    myfile = open(myresultsFileName, 'wb')
    pickle.dump(results, myfile)
    return results


def extractParametersFromPickleFilesGeneral(depName, mydate, numParams, myfoldername, extraName, dim1=1000):
    results = np.zeros((dim1, numParams+1))
    #open the folder and extract for each file the line with the results and store it in a big matrix
    for ivals in range(0,dim1):

        # myfullpath = 'leastSquares2019/' + depName + '/' + depName + '_NewDataSet' + str(ivals)+ 'Iterations_' + mydate + '.pickle'
        # myfullpath = 'leastSquaresNationalFit/' + depName + '/' + depName  +  '_3paramsConstrainedM2_reportingRateOf_' + \
        #              reportingRate + '_' + str(ivals) + 'Iterations_' + mydate + '.pickle'

        myfullpath = myfoldername + '/' +  depName + '/' + depName  + '_'+ \
                      str(ivals) + 'Iterations_' + mydate + '.pickle'

        print myfullpath
        if os.path.isfile(myfullpath):

            # print myfullpath
            f = open(myfullpath, 'r')
            temp = pickle.load(f)
            # print temp
            # # matchingFiles = getMatchingFiles(dir=myfoldername, regex=myfullpath, full_names=True)[0]
            results[ivals,0:numParams] = temp[1]
            results[ivals, numParams] = temp[-1]
            f.close()
        else:
            print ivals
    # print results

    # store the results in a pickle file:
    # myresultsFileName = 'leastSquaresResults/' + depName + '_resultsMatrix' + mydate + '.pickle'
    myresultsFileName = myfoldername + '/' + depName + '_resultsMatrix_' + extraName + \
                     mydate + '.pickle'
    print myresultsFileName
    myfile = open(myresultsFileName, 'wb')
    pickle.dump(results, myfile)
    return results

def extractParametersFromPickleFilesGeneral2(depName, mydate, numParams, myfoldername, extraName1, extraName2, dim1=1000):
    results = np.zeros((dim1, numParams+1))
    #open the folder and extract for each file the line with the results and store it in a big matrix
    for ivals in range(0,dim1):

        # myfullpath = 'leastSquares2019/' + depName + '/' + depName + '_NewDataSet' + str(ivals)+ 'Iterations_' + mydate + '.pickle'
        # myfullpath = 'leastSquaresNationalFit/' + depName + '/' + depName  +  '_3paramsConstrainedM2_reportingRateOf_' + \
        #              reportingRate + '_' + str(ivals) + 'Iterations_' + mydate + '.pickle'

        myfullpath = myfoldername + '/' +  depName + '/' + depName  + '_'+ extraName1 + \
                      str(ivals) + 'Iterations_' + mydate + '.pickle'

        print myfullpath
        if os.path.isfile(myfullpath):

            # print myfullpath
            f = open(myfullpath, 'r')
            temp = pickle.load(f)
            # print temp
            # # matchingFiles = getMatchingFiles(dir=myfoldername, regex=myfullpath, full_names=True)[0]
            results[ivals,0:numParams] = temp[1]
            results[ivals, numParams] = temp[-1]
            f.close()
        else:
            print ivals
    # print results

    # store the results in a pickle file:
    # myresultsFileName = 'leastSquaresResults/' + depName + '_resultsMatrix' + mydate + '.pickle'
    myresultsFileName = myfoldername + '/' + depName + '_resultsMatrix_' + extraName2 + \
                     mydate + '.pickle'
    print myresultsFileName
    myfile = open(myresultsFileName, 'wb')
    pickle.dump(results, myfile)
    return results


def extractMainResults(mydate,reportingRate):
    today = time.strftime("%d%b%Y", time.localtime())
    results = []#np.zeros((1000, 7))
    redFrac = np.zeros((2733, 4))
    #open the folder and extract for each file the line with the results and store it in a big matrix
    for ivals in range(0,2732):

        myfullpath0 = 'mainResults/' +  'results_main1_and_main3reportingRateOf_' + reportingRate + '_'+  mydate[0] + \
                      str(ivals)+ '.pickle'
        myfullpath1 = 'mainResults/' + 'results_main1_and_main3reportingRateOf_' + reportingRate + '_' + mydate[1] + \
                      str(ivals) + '.pickle'
        myfullpath2 = 'mainResults/' +  'results_main2reportingRateOf_' + reportingRate + '_'+  mydate[1] + \
                      str(ivals)+ '.pickle'


        # print myfullpath1
        if os.path.isfile(myfullpath0):
            if os.path.isfile(myfullpath2):
            # print myfullpath
                f = open(myfullpath0, 'r')
                temp = pickle.load(f)
                f.close()

                redFrac[ivals, 0] = ivals
                redFrac[ivals, 1] = 1 - temp[1][1]
                redFrac[ivals, 3] = 1 - temp[2][1]

                f = open(myfullpath2, 'r')
                temp2 = pickle.load(f)
                f.close()

                redFrac[ivals, 2] = 1 - temp2[1][1]

                results.append([temp[0], temp[1][0], temp2[1][0], temp[2][0]])
            else:
                print 'main2 absent', ivals

        elif os.path.isfile(myfullpath1):
            if os.path.isfile(myfullpath2):
                # print myfullpath
                f = open(myfullpath1, 'r')
                temp = pickle.load(f)
                f.close()

                redFrac[ivals, 0] = ivals
                redFrac[ivals, 1] = 1 - temp[1][1]
                redFrac[ivals, 3] = 1 - temp[2][1]

                f = open(myfullpath2, 'r')
                temp2 = pickle.load(f)
                f.close()

                redFrac[ivals, 2] = 1 - temp2[1][1]

                results.append([temp[0], temp[1][0], temp2[1][0], temp[2][0]])
            else:
                print 'main2 absent', ivals

        else:

            print 'main1 absent', ivals
    # print len(results)
    print np.median(redFrac[:, 1:], 0)

    print np.shape(temp[1][0])


    #store the results in a pickle file:
    myresultsFileName = 'mainResultsRuns/modelWithNoNetwork_'  + reportingRate + '_'+  today + '.pickle'
    myfile = open(myresultsFileName, 'wb')
    pickle.dump(results, myfile)
    print myresultsFileName

    #store the results in a pickle file:
    mymatfileName = 'mainResultsRuns/reduction_matrix_ modelWithNoNetwork_'  + reportingRate + '_'+  today + '.pickle'
    myfile = open(mymatfileName, 'wb')
    pickle.dump(redFrac, myfile)



def extractMainResults2(mydate,reportingRate):
    today = time.strftime("%d%b%Y", time.localtime())
    results = []#np.zeros((1000, 7))
    # redFrac = np.zeros((4493, 4))
    redFrac = np.zeros((950, 4))
    #open the folder and extract for each file the line with the results and store it in a big matrix
    for ivals in range(950):


        myfullpath0 = 'mainResults/' +  'results_allMain_ConstrainedP0_Coverage95reportingRateOf_' + reportingRate + '_'+  mydate + \
                      str(ivals)+ '.pickle'

        print myfullpath0

        if os.path.isfile(myfullpath0):
            f = open(myfullpath0, 'r')
            temp = pickle.load(f)
            f.close()

            redFrac[ivals, 0] = ivals
            redFrac[ivals, 1] = 1 - temp[1][1]
            redFrac[ivals, 2] = 1 - temp[2][1]
            redFrac[ivals, 3] = 1 - temp[3][1]


            results.append([temp[0], temp[1][0], temp[2][0], temp[3][0]])
        else:
                print 'main absent', ivals


    print np.median(redFrac[:, 1:], 0)




    #store the results in a pickle file:
    myresultsFileName = 'mainResultsRuns/modelWithNoNetwork_1_ConstrainedP0_Coverage95'  + reportingRate + '_'+  today + '.pickle'
    myfile = open(myresultsFileName, 'wb')
    pickle.dump(results, myfile)
    print myresultsFileName

    #store the results in a pickle file:
    mymatfileName = 'mainResultsRuns/reduction_matrix_ modelWithNoNetwork_1_ConstrainedP0_Coverage95'  + reportingRate + '_'+  today + '.pickle'
    myfile = open(mymatfileName, 'wb')
    pickle.dump(redFrac, myfile)


def extractMainResultsMarch2019(mydate,reportingRate, dispersionSA, filename):
    today = time.strftime("%d%b%Y", time.localtime())

    scenarioNums = [1, 2, 3, 4, 25]
    results = []#np.zeros((1000, 7))
    # redFrac = np.zeros((4493, 4))
    redFrac = np.zeros((1002, 4))

    for scevals in range(5):
        print scevals
        cases = 7.4*np.ones(((426), 1002))
        casesVac = 7.4*np.ones(((521), 1002))
        casesDoNothing = 7.4*np.ones(((521), 1002))
        redFrac = 7.4**np.ones(1002)
        #open the folder and extract for each file the line with the results and store it in a big matrix
        for ivals in range(1002):
            # myfullpath0 = 'mainResultsMarch2019/sensitivityAnalysis/' + filename +  mydate + '_' +\
            #               str(ivals) + '.pickle'
            myfullpath0 = 'mainResultsMarch2019/sensitivityAnalysis/' + filename +  '_'  + str(dispersionSA) +  mydate + '_' +\
                          str(ivals) + '.pickle'

            print myfullpath0


            if os.path.isfile(myfullpath0):
                # print ivals
                f = open(myfullpath0, 'r')
                temp = pickle.load(f)
                f.close()

                tempres = temp[scevals]
                dim1 = len(tempres[3])
                dim2 = len(tempres[4])
                cases[:, ivals] = tempres[2]
                casesVac[:dim1, ivals] = tempres[3]
                casesDoNothing[:dim2, ivals] = tempres[4]
                redFrac[ivals] = tempres[5]

                matResults = [cases, casesVac, casesDoNothing, redFrac]
                # print matResults[0]
            else:
                print 'main absent', ivals

        #store the results in pickle files:
        myresultsname = 'mainResultsMarch2019/sensitivityAnalysisMatrices/matrixResultForScenario_' + \
                        str(scenarioNums[scevals]) + '_' + str(dispersionSA) + 'width_' + reportingRate + \
                        'repRate'+  today + '.pickle'
        myfile = open(myresultsname, 'wb')
        pickle.dump(matResults, myfile)
        print myresultsname


def extractMainResults2019AllInfections(mydate,reportingRate, dispersionSA, filename):
    today = time.strftime("%d%b%Y", time.localtime())

    scenarioNums = [1, 2, 3, 4, 25]
    results = []#np.zeros((1000, 7))
    # redFrac = np.zeros((4493, 4))
    redFrac = np.zeros((1002, 4))

    for scevals in range(5):
        print scevals
        cases = 7.4*np.ones(((426), 1002))
        if scevals == 0 or scevals == 2 or scevals ==4:
            casesVac = 7.4*np.ones(((519), 1002))
            casesDoNothing = 7.4*np.ones(((519), 1002))
        else:
            casesVac = 7.4*np.ones(((521), 1002))
            casesDoNothing = 7.4*np.ones(((521), 1002))
        redFrac = 7.4**np.ones(1002)
        #open the folder and extract for each file the line with the results and store it in a big matrix
        for ivals in range(1002):
            # myfullpath0 = 'mainResultsMarch2019/sensitivityAnalysis/' + filename +  mydate + '_' +\
            #               str(ivals) + '.pickle'
            myfullpath0 = 'mainResults2019AllInfections/sensitivityAnalysis/' + filename +  '_'  + str(dispersionSA) +  mydate + '_' +\
                          str(ivals) + '.pickle'

            # print myfullpath0


            if os.path.isfile(myfullpath0):
                # print ivals
                f = open(myfullpath0, 'r')
                temp = pickle.load(f)
                f.close()

                tempres = temp[scevals]
                dim1 = len(tempres[3])
                dim2 = len(tempres[4])
                # print [dim1, dim2]
                cases[:, ivals] = tempres[2]
                casesVac[:, ivals] = tempres[3]
                casesDoNothing[:, ivals] = tempres[4]
                redFrac[ivals] = tempres[5]

                # print len(tempres[3])
                # plt.plot(range(len(tempres[2])), tempres[2])

                matResults = [cases, casesVac, casesDoNothing, redFrac]
                # print matResults[0]
            else:
                print 'main absent', ivals

        #store the results in pickle files:
        myresultsname = 'mainResults2019AllInfections/sensitivityAnalysisMatrices/matrixResultForScenario_' + \
                        str(scenarioNums[scevals]) + '_' + str(dispersionSA) + 'width_' + reportingRate + \
                        'repRate'+  today + '.pickle'
        myfile = open(myresultsname, 'wb')
        pickle.dump(matResults, myfile)
        # print myresultsname
        # plt.show()


def extractMainResults2019AllInfectionsScenario0(mydate,reportingRate, dispersionSA, filename, mydate2):
    today = time.strftime("%d%b%Y", time.localtime())

    scenarioNums = [0]
    results = []#np.zeros((1000, 7))
    # redFrac = np.zeros((4493, 4))
    redFrac = np.zeros((1002, 4))

    cases = 7.4 * np.ones(((426), 1002))
    casesDoNothing = 7.4*np.ones(((521), 1002))

    #open the folder and extract for each file the line with the results and store it in a big matrix
    for ivals in range(1002):
        # myfullpath0 = 'mainResultsMarch2019/sensitivityAnalysis/' + filename +  mydate + '_' +\
        #               str(ivals) + '.pickle'
        myfullpath0 = 'mainResults2019AllInfections/sensitivityAnalysis/' + filename +  '_'  + str(dispersionSA) +  mydate + '_' +\
                      str(ivals) + '.pickle'

        # print myfullpath0


        if os.path.isfile(myfullpath0):
            # print ivals
            f = open(myfullpath0, 'r')
            temp = pickle.load(f)
            f.close()

            # print np.shape(temp[1])

            tempres = temp[1]
            dim2 = len(tempres[4])
            cases[:, ivals] = tempres[2]
            casesDoNothing[:dim2, ivals] = tempres[4]
            # redFrac[ivals] = tempres[5]

            # plt.plot(range(len(casesDoNothing)), casesDoNothing)


            matResults = [cases, casesDoNothing]
            # print matResults[0]
        else:
            print 'main absent', ivals

    #store the results in pickle files:
    myresultsname = 'mainResults2019AllInfections/sensitivityAnalysisMatrices/matrixResultForScenario_' + \
                    str(scenarioNums[0]) + '_' + str(dispersionSA) + 'width_' + reportingRate + \
                    'repRate'+ mydate2 + '.pickle'
    myfile = open(myresultsname, 'wb')
    pickle.dump(matResults, myfile)
    print myresultsname



def extractMainResults2019AllInfections(mydate,reportingRate, dispersionSA, filename):
    today = time.strftime("%d%b%Y", time.localtime())

    scenarioNums = [1, 2, 3, 4, 25]
    results = []#np.zeros((1000, 7))
    # redFrac = np.zeros((4493, 4))
    redFrac = np.zeros((1002, 4))

    for scevals in range(5):
        print scevals
        cases = 7.4*np.ones(((426), 1002))
        if scevals == 0 or scevals == 2 or scevals ==4:
            casesVac = 7.4*np.ones(((519), 1002))
            casesDoNothing = 7.4*np.ones(((519), 1002))
        else:
            casesVac = 7.4*np.ones(((521), 1002))
            casesDoNothing = 7.4*np.ones(((521), 1002))
        redFrac = 7.4**np.ones(1002)
        #open the folder and extract for each file the line with the results and store it in a big matrix
        for ivals in range(1002):
            # myfullpath0 = 'mainResultsMarch2019/sensitivityAnalysis/' + filename +  mydate + '_' +\
            #               str(ivals) + '.pickle'
            myfullpath0 = 'mainResults2019AllInfections/sensitivityAnalysis/' + filename +  '_'  + str(dispersionSA) +  mydate + '_' +\
                          str(ivals) + '.pickle'

            # print myfullpath0


            if os.path.isfile(myfullpath0):
                # print ivals
                f = open(myfullpath0, 'r')
                temp = pickle.load(f)
                f.close()

                tempres = temp[scevals]
                dim1 = len(tempres[3])
                dim2 = len(tempres[4])
                # print [dim1, dim2]
                cases[:, ivals] = tempres[2]
                casesVac[:, ivals] = tempres[3]
                casesDoNothing[:, ivals] = tempres[4]
                redFrac[ivals] = tempres[5]

                # print len(tempres[3])
                # plt.plot(range(len(tempres[2])), tempres[2])

                matResults = [cases, casesVac, casesDoNothing, redFrac]
                # print matResults[0]
            else:
                print 'main absent', ivals

        #store the results in pickle files:
        myresultsname = 'mainResults2019AllInfections/sensitivityAnalysisMatrices/matrixResultForScenario_' + \
                        str(scenarioNums[scevals]) + '_' + str(dispersionSA) + 'width_' + reportingRate + \
                        'repRate'+  today + '.pickle'
        myfile = open(myresultsname, 'wb')
        pickle.dump(matResults, myfile)
        # print myresultsname
        # plt.show()

def extractMainResults2019AllInfectionsPriorVaccination(mydate,reportingRate, dispersionSA, filename):
    today = time.strftime("%d%b%Y", time.localtime())

    scenarioNums = [1, 2, 3, 4, 25]
    results = []#np.zeros((1000, 7))
    # redFrac = np.zeros((4493, 4))
    redFrac = np.zeros((1002, 4))

    for scevals in range(5):
        print scevals
        cases = 7.4*np.ones(((426), 1002))
        if scevals == 0 or scevals == 2 or scevals ==4:
            casesVac = 7.4*np.ones(((519), 1002))
            casesDoNothing = 7.4*np.ones(((519), 1002))
        else:
            casesVac = 7.4*np.ones(((521), 1002))
            casesDoNothing = 7.4*np.ones(((521), 1002))
        redFrac = 7.4**np.ones(1002)
        #open the folder and extract for each file the line with the results and store it in a big matrix
        for ivals in range(1002):
            # myfullpath0 = 'mainResultsMarch2019/sensitivityAnalysis/' + filename +  mydate + '_' +\
            #               str(ivals) + '.pickle'
            myfullpath0 = 'mainResults2019AllInfections/sensitivityAnalysis/' + filename +  '_'  + str(dispersionSA) +  mydate + '_' +\
                          str(ivals) + '.pickle'

            # print myfullpath0


            if os.path.isfile(myfullpath0):
                # print ivals
                f = open(myfullpath0, 'r')
                temp = pickle.load(f)
                f.close()

                tempres = temp[scevals]
                dim1 = len(tempres[3])
                dim2 = len(tempres[4])
                # print [dim1, dim2]
                cases[:, ivals] = tempres[2]
                casesVac[:, ivals] = tempres[3]
                casesDoNothing[:, ivals] = tempres[4]
                redFrac[ivals] = tempres[5]

                # print len(tempres[3])
                # plt.plot(range(len(tempres[2])), tempres[2])

                matResults = [cases, casesVac, casesDoNothing, redFrac]
                # print matResults[0]
            else:
                print 'main absent', ivals

        #store the results in pickle files:
        myresultsname = 'mainResults2019AllInfections/sensitivityAnalysisMatrices/matrixResultForScenario_' + \
                        str(scenarioNums[scevals]) + '_' + str(dispersionSA) + 'width_' + reportingRate + \
                        'repRate'+  today + '.pickle'
        myfile = open(myresultsname, 'wb')
        pickle.dump(matResults, myfile)
        # print myresultsname
        # plt.show()



if __name__ == "__main__":
    depNames = ['artibonite', 'centre', 'grand_anse', 'nippes', 'nord', 'nord_est', 'nord_ouest', 'ouest', 'sud',
                 'sud_est']
    depNames2 = ['allHaiti']
    depNames3 = ['fullModel']
    depNames5 = ['fullModelLessParameters']
    depNames6 = ['fullModel3Parameters']
    depNames4 = ['sud_est']
    depNames7 = ['fullModel6Parameters']
    depNames8 = ['main']
    # for names in depNames6:
    #     print names
    #     # extractResultsListPickleFiles(names, '19Oct2018')
    #     # extractParametersFromPickleFilesLessParameters(names, '29Oct2018')
    #     # extractParametersFromPickleFilesICfit(names, '23Aug2018')
    #     # extractResultsListPickleFiles(names, '29Aug2018')
    #     # extractParametersFromPickleFiles3(names, '08Sep2018')
    #     # extractParametersFromPickleFiles3(names, '09Sep2018')
    #     # extractParametersFromPickleFilesNew(names, '30Nov2018')
    #     extractParametersFromPickleFilesNew3Params(names, '28Jan2019')
    #     # extractParametersFromPickleFilesFixedRepRateNewModel2019(names, '25Jan2019', str(20.0))
    #     # extractParametersFromPickleFilesFixedRepRateAllHaitiNewModel2019(names, '23Jan2019', str(20.0))
    #     # extractParametersFromPickleFilesFixedRepRate6Params2019(names, '19Jan2019')

    # mydates = ['23Jan2019', '24Jan2019']
    # # extractMainResults(mydates, str(20.0))
    # extractMainResults2('28Jan2019', str(20.0))
    # extractParametersFromPickleFilesGeneral('fitPieces3paramsFracSus', '14Mar2019', 4, 'leastSquaresDataUpTo2019', '')
    # extractParametersFromPickleFilesGeneral('fitEpidemicPiece', '18Mar2019', 3, 'leastSquaresDataUpTo2019', '')
    # extractParametersFromPickleFilesGeneral2('fitPieces3paramsFracSus', '19Mar2019', 4, 'leastSquaresDataUpTo2019', 'breakingPoint_136_', 'breakingPoint_136_')

    # extractMainResultsMarch2019('01Apr2019', str(20.0), 'width0.25_fracSusWidth0.1', 'mainResults_1to4and25')
    extractMainResults2019AllInfectionsScenario0('29May2019', str(20.0), 'width0.25_fracSusWidth0.1', 'mainResults_1to4and25', '30May2019')
    # extractMainResults2019AllInfections('29May2019', str(20.0), 'width0.25_fracSusWidth0.1',
    #                                              'mainResults_1to4and25')
    # plt.show()

    # extractParametersFromPickleFilesGeneral2('fitEndemicMuWithFracSusFixed', '30May2019', 1, 'leastSquaresDataUpto2019',
    #                             'breakingPoint_176_fracSus0.6_', 'breakingPoint_176_fracSus0.6_',
    #                                          dim1=1000)
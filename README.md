# HaitiCholeraMultiModelingProject
Code used to generate the model/output/etc used in the multi-modeling exercise for Haiti mass-vaccination.

## Files:
- choleraEqsPublic.py contains the differential equations used in this model.
- functionsCholeraProjectPublic.py contains some functions used in the model.
- fitInPieces3paramsCleanMay2019Public.py is the code that was used to fit the epidemic portion of the data (up to April 2011).
- fitInPiecesMuWithFracSusFixedAllInfectionsPublic.py contains the function used to fit the endemic portion of the data . 
- runFitInPiecesPublic.sh Runs in a cluster 1000 times these functions to obtain 1000 different "best fits"
- extractAllRunsFromEachDepartmentPublic.py is used to extract the results of these fits and puts them into a single matrix to further 
simulations.
- generateSetsOfParametersForSensitivityAnalysisPublic.py generates the best parameter fit and a 1000 parameters around this best fit using latin hypercube sampling. 
-simulateHaitiFullTimeInPiecesAllVaccinationScenariosSingleParameterVectorAllInfectionsPublic.py simulates each of these parameter sets for sensitivity analysis for the baseline case and the vaccine scenarios considered in this exercise. It saves the outputs as pickle files. 
- runFitInPiecesPublic.sh runs the simulations in the cluster.
- computeProbabilityOfElimination2019AllInfectionsPublic.ipynb computes all the outcome measures (prob of elimination, resurgence, etc) used in the paper. 
- creates a figure with the proportion of the population of Haiti that is vaccinated each week under each of the different sceanarios. 


[![DOI](https://zenodo.org/badge/199940767.svg)](https://zenodo.org/badge/latestdoi/199940767)

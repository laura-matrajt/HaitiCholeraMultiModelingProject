#!/bin/bash


for i in 'mainResults2019AllInfections'
do 
sbatch --array=0-1001 -n 1 -c 1 --job-name=$i --partition=campus --wrap="python simulateHaitiFullTimeInPiecesAllVaccinationScenariosSingleParameterVectorAllInfections.py" --output=output/$i/%A_%a.out
   
done
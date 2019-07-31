#!/bin/bash


for i in 'fitInPieces3params'
do
sbatch --array=0-999 -n 1 -c 1 --job-name=$i --partition=campus --wrap="python fitInPieces3paramsClean.py" --output=output/$i/%A_%a.out

done
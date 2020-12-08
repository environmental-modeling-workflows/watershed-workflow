#!/bin/sh

#Rosetta-V3 H2w model; 
#please make sure that you are using Rosetta-V3 H2w model: the first input for 'PTF_MODEL' function is 2 at line 86 in Rpredict.py
#the data format is soil sand, silt, and clay percentages (in weight %)
python Rpredict.py   -i   ./output/test_input_H2w.txt  -o  ./output/test_output_H2w.txt --predict  --sqlite=./sqlite/Rosetta.sqlite

#Rosetta-V3 H3w model
#please make sure that you are using Rosetta-V3 H3w model: the first input for 'PTF_MODEL' function is 3 at line 86 in Rpredict.py
#the data format is soil sand, silt, clay percentages (in weight %), and bulk density (in g/cm3)
python Rpredict.py   -i   ./output/test_input_H3w.txt  -o  ./output/test_output_H3w.txt --predict  --sqlite=./sqlite/Rosetta.sqlite
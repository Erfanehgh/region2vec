#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --partition=standard

python main.py './bedfiles/*'  './representations/antibody_universe.txt' 4 10 10 2 8
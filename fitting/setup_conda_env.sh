#!/bin/bash

# exectute this once to create the conda environment for scikit-optimize
# see also the section on conda enviroments in the MaRC3 user guide

module load miniconda
source $CONDA_ROOT/bin/activate
export LC_ALL=C
conda create --name scikit
conda activate scikit
conda install scikit-learn
conda install scikit-optimize
conda install matplotlib
conda install pandas
pip install inferactively-pymdp
conda deactivate
conda deactivate
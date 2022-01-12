#!/bin/bash

# create environment
conda create -n sfct python=3.6

# install packages
conda activate sfct
conda install -c openbabel openbabel -y
pip install -U scikit-learn==0.23.2
conda install pandas -y
pip install mdtraj
conda install -c conda-forge biopandas -y


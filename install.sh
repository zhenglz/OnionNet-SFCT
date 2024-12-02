#!/bin/bash

# create environment
conda create -n sfct python=3.6 -y

# install packages
conda activate sfct
conda install -c openbabel openbabel -y
pip install -U scikit-learn==0.23.2
conda install pandas -y
pip install mdtraj
conda install -c conda-forge biopandas -y

# downloading models through web-link is not working. please use google drive link to download files.
cd models
#for i in {1..4}
#do
#  wget http://jtmeng.sharelatex.top:9001/sfct_models/onionnet-sfct_s${i}.model   
#done

#wget http://jtmeng.sharelatex.top:9001/sfct_models/onionnet-sfct_std.model

#cd ../

echo "installing sfct done ..." 

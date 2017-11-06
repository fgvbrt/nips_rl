#!/usr/bin/env bash
conda create -n nips_rl_fast3 python=3.5.1 mkl-service -y
source activate nips_rl_fast3
conda install -c conda-forge lapack git -y
pip install -r requirements.txt
pip install -e opensim/python3.5/site-packages
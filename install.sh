#!/bin/bash

# install.sh

# --
# Conda env

conda create -n obtext_env python=3.7 pip -y
conda activate obtext_env

# --
# Requirements

conda install -y pytorch==1.1 -c pytorch
pip install -r requirements.txt

# --
# Install obtext

pip install -e .
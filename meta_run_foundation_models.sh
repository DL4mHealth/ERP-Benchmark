#!/bin/bash

# Run scripts sequentially


# Medical Time Series Classification methods
# BIOT
bash scripts/BIOT/supervised/BIOT/S-1.sh   # Foundation Model


# EEG classification methods
bash scripts/LaBraM/supervised/LaBraM/S-1.sh   # Foundation Model
# CBraMod
bash scripts/CBraMod/supervised/CBraMod/S-1.sh   # Foundation Model
#!/bin/bash

# Run scripts sequentially

# General Time Series Classification methods
# TCN
bash scripts/TCN/supervised/TCN/S-1.sh
# TimesNet
bash scripts/TimesNet/supervised/TimesNet/S-1.sh
# ModernTCN
bash scripts/ModernTCN/supervised/ModernTCN/S-1.sh
# PatchTST
bash scripts/PatchTST/supervised/PatchTST/S-1.sh
# iTransformer
bash scripts/iTransformer/supervised/iTransformer/S-1.sh


# Medical Time Series Classification methods
# Medformer
bash scripts/Medformer/supervised/Medformer/S-1.sh
# MedGNN
bash scripts/MedGNN/supervised/MedGNN/S-1.sh


# EEG classification methods
# EEGNet
bash scripts/EEGNet/supervised/EEGNet/S-1.sh
# EEGInception
bash scripts/EEGInception/supervised/EEGInception/S-1.sh
# EEGConformer
bash scripts/EEGConformer/supervised/EEGConformer/S-1.sh
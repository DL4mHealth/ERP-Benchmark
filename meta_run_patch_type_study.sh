#!/bin/bash

# Run scripts sequentially


# Patch type study
# Uni-Variate
bash scripts/TestFormer/supervised/TestFormer/S-1-Uni-Variate.sh

# Multi-Variate
bash scripts/TestFormer/supervised/TestFormer/S-1-Multi-Variate.sh

# Whole-Variate
bash scripts/TestFormer/supervised/TestFormer/S-1-Whole-Variate.sh
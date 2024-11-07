# Task Shift: From Classification to Regression in Overparameterized Linear Models

### Installation
```
conda update -n base -c defaults conda
conda create -n taskshift python==3.10
conda activate taskshift
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

### User Guide

To see all commands and options, run
```
python shift.py -h
```

The most important option is `--cov_type`, which enables either isotropic, spiked, or polynomial decay covariance.

Usage example for a simulation with spiked covariance and 2-sparse signal:
```
python shift.py  --cov_type spiked --sparse_inds 1 2 --sparse_vals 0.2 -0.1 --spiked_r 0.5 --spiked_q 0.6
```

If you are not using a GPU, set `--cuda False`. It is True by default. If the code runs too slow, you may want to try `--solver gd`, although sometimes gradient descent is slower than solving for the MNI directly.

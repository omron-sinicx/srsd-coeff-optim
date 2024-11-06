# Two-Stage Coefficient Estimation in Symbolic Regression for Scientific Discovery
This repository contains the official implementation of ["Two-Stage Coefficient Estimation in Symbolic Regression for Scientific Discovery"](https://openreview.net/forum?id=gIscWmLoYf) accepted at the Machine Learning and the Physical Sciences Workshop @ NeurIPS 2024.

## Setup
### To reproduce experiments in the workshop paper
Make a virtual environment.
```
$ pyenv global 3.11
$ python -m venv env
$ source env/bin/activate
$ pip install --upgrade pip
```
Install srsd-coeff-optim library.
```
$ pip install .
```
Install libraries for experiments.
```
$ pip install -r expt/requirements.txt
```
### To install our library for your own project
```
$ pip install git+https://github.com/omron-sinicx/srsd-coeff-optim
```

## Reproduction experiments
Make data directory under expt and prepare [srsd-benchmark datasets](https://github.com/omron-sinicx/srsd-benchmark).
```
$ pwd
***/srsd-coeff-optim/expt/data
$ git lfs install
$ git clone https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_easy
$ git clone https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_medium
$ git clone https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_hard
```
### Estimate coefficients given the ground truth skeleton
E.g.)
```
$ python expt/main.py --dataset srsd --n_data 20 --discrete_method brute-force --continuous_method lm-jump --max_contiter 100 --outiter 2 --beamsize 10 --init_method uniform --n_core 8 --seed 0
```
### Evaluate the estimation results
E.g.)
```
$ python expt/analyze.py --eval_method stats_by_difficulty --dataset srsd --n_data 20 --discrete_method brute-force --continuous_method lm-jump --outiter 2 --beamsize 10 --init_method uniform --seed 0
```
### Generate figures in the paper or poster
E.g.)
```
$ python expt/figure.py
```

## Call our optimizer in your project
All you need to do is instantiate an Optimizer class object and call the optimize method.

E.g.)
```
import math

import sympy

from srsd_coeff_optim import Optimizer

optimizer = Optimizer(
    discrete_method="brute-force",
    continuous_method="lm-jump",
    max_contiter=100,
    outiter=2,
    beamsize=10,
    seed=0,
    discrete_candidates=[-1, 0.5, -0.5, 1.5, -1.5, 2, -2, 3, -3, 4, -4, 5, -5],
)

f = sympy.sympify("x0 * c0 + log(c1 + x1 ** c2)")  # x0 * 1 + log(2 + x1^3)
data = [
    [2.0, 4.0, 2.0 + math.log(2.0 + 4.0**3.0)],
    [3.0, 6.0, 3.0 + math.log(2.0 + 6.0**3.0)],
    [4.0, 1.0, 4.0 + math.log(2.0 + 1.0**3.0)],
]  # [x0, x1, y]
coeff_init = None
init_method = "uniform"
allinfo = True

(
    formula,
    status,
    ffinal,
    coeff_opt,
    disc_time_hist,
    cont_time_hist,
    expo_coeffs,
    other_coeffs,
    disc_coeffs,
    cont_coeffs,
    coeff_opt_hist,
    cont_error_hist,
) = optimizer.optimize(f, data, coeff_init, init_method, allinfo)
assert abs(coeff_opt[0] - 1.0) < 1e-6  # c0
assert abs(coeff_opt[1] - 2.0) < 1e-6  # c1
assert coeff_opt[2] == 3.0  # c2

```

## Citation
TODO

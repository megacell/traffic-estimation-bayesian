Bayesian inference for traffic estimation
======

Dependencies
------

    brew install gfortran
    pip install -r requirements.txt
    pip install ipdb

Install PyMC3 (since this is in active development, checkout a particular commit we've tested against)

    git clone https://github.com/pymc-devs/pymc.git
    pushd pymc
    git checkout 3bf4ad3285f658d02a6b4297160b45354666fe46
    python setup.py install

Usage
------

    python grid_simulation.py

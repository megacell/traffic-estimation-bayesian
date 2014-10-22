from pymc import DiscreteUniform, Exponential, Poisson, \
        Uniform, Bernoulli, Normal, Beta, Gamma, Dirichlet
import pymc as pm
import numpy as np
from scipy.io import loadmat
from itertools import chain

# dot product for weighted probability distributions
def dot(a,b):
    return sum([x*y for (x,y) in zip(a,b)])

# load data and change variable names
# fname = '2_3_3_1_20140421T151732_1_small_graph_OD_dense'
fname = '3_3_3_1_20140421T173515_5_small_graph_OD'
# fname = '4_6_3_1_20140421T155253_1_small_graph_OD'
data = loadmat('data/%s.mat' % fname)
sparse = True
A = data['phi']
b_obs = data['f']
x_true = data['real_a']
block_sizes = data['block_sizes']


if sparse == True:
    alpha = 0.3
else:
    alpha = 1

# construct graphical model
# -----------------------------------------------------------------------------

import time
with pm.Model() as model:
    START = time.time()
    # construct sparse prior on all routes
    # Dirichlet distribution doesn't give values that sum to 1 (continuous distribution), so 
    # instead we normalize draws from a Gamma distribution
    x_blocks = [Gamma('x%d' % i,np.array([alpha]*(x[0])), 1, shape=x[0]) for (i,x) in \
            enumerate(block_sizes)]
    x_blocks_expanded = [[x[xi]/x.sum() for xi in range(i-1)] for (i,x) in \
            zip(block_sizes,x_blocks)]
    [x.append(1 - sum(x)) for x in x_blocks_expanded]
    x_pri = list(chain(*x_blocks_expanded))

    # construct skinny normal distributions with observations
    mus = [dot(a,x_pri) for a in A]
    b = [Normal('b%s' % i,mu=mu, tau=10000,observed=b_obsi[0]) for \
            (i,(mu,b_obsi)) in enumerate(zip(mus,b_obs))]
    print 'Time to build model: %ds' % (time.time()-START)

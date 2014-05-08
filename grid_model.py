from pymc import DiscreteUniform, Exponential, deterministic, Poisson, \
        Uniform, stochastic, Bernoulli, Normal, MCMC, Beta, Gamma, Dirichlet
import numpy as np
from scipy.io import loadmat
from itertools import chain

# dot product for weighted probability distributions
def dot(a,b):
    return sum([x*y for (x,y) in zip(a,b)])

# load data and change variable names
# data = loadmat('data/2_3_3_1_20140421T151732_1_small_graph_OD_dense.mat')
# data = loadmat('data/3_3_3_1_20140421T173515_5_small_graph_OD.mat')
data = loadmat('data/4_6_3_1_20140421T155253_1_small_graph_OD.mat')
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

# construct sparse prior on all routes
x_blocks = [Dirichlet('x%d' % i,np.array([alpha]*(x[0]))) for (i,x) in \
        enumerate(block_sizes)]
x_blocks_expanded = [[x[xi] for xi in range(i-1)] for (i,x) in \
        zip(block_sizes,x_blocks)]
[x.append(1 - sum(x)) for x in x_blocks_expanded]
x_pri = list(chain(*x_blocks_expanded))

# construct skinny normal distributions with observations
mus = [dot(a,x_pri) for a in A]
b = [Normal('b%s' % i,mu=mu, tau=10000,value=b_obsi[0],observed=True) for \
        (i,(mu,b_obsi)) in enumerate(zip(mus,b_obs))]

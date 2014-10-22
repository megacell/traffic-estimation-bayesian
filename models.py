from pymc import DiscreteUniform, Exponential, deterministic, Poisson, \
        Uniform, stochastic, Bernoulli, Normal, MCMC, Beta, Gamma, Dirichlet
import numpy as np
from scipy.io import loadmat
from itertools import chain

# dot product for weighted probability distributions
def dot(a,b):
    return sum([x*y for (x,y) in zip(a,b)])

def grid_model(fname,tau=10000,sparse=True):
    data = loadmat('data/%s.mat' % fname)
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
    x_blocks = [Dirichlet('x%d' % i,np.array([alpha]*(x[0])),trace=True) for (i,x) in \
            enumerate(block_sizes)]
    x_blocks_expanded = [[x[xi] for xi in range(i-1)] for (i,x) in \
            zip(block_sizes,x_blocks)]
    [x.append(1 - sum(x)) for x in x_blocks_expanded]
    x_pri = list(chain(*x_blocks_expanded))
    
    # construct skinny normal distributions with observations
    mus = [dot(a,x_pri) for a in A]
    b = [Normal('b%s' % i,mu=mu, tau=tau,value=b_obsi[0],observed=True) for \
            (i,(mu,b_obsi)) in enumerate(zip(mus,b_obs))]

    return locals()


def toy_model(tau=10000,prior='Beta0.5'):
    b_obs = 200
    f_AB = 400
    f_CB = 1000
    f_CA = 600

    A = np.array([0, f_AB, f_CA, 0, f_CB, 0])
    
    if prior == 'Normal':
        ABp  = Normal('ABp',mu=0.5,tau=100,trace=True)
        CBp  = Normal('CBp',mu=0.5,tau=100,trace=True)
        CAp  = Normal('CAp',mu=0.5,tau=100,trace=True)
    elif prior == 'Uniform':
        ABp  = Uniform('ABp',lower=0.0,upper=1.0,trace=True)
        CBp  = Uniform('CBp',lower=0.0,upper=1.0,trace=True)
        CAp  = Uniform('CAp',lower=0.0,upper=1.0,trace=True)
    elif prior == 'Beta0.25':
        ABp  = Beta('ABp',alpha=0.25,beta=0.25,trace=True)
        CBp  = Beta('CBp',alpha=0.25,beta=0.25,trace=True)
        CAp  = Beta('CAp',alpha=0.25,beta=0.25,trace=True)
    elif prior == 'Beta0.5':
        ABp  = Beta('ABp',alpha=0.5,beta=0.5,trace=True)
        CBp  = Beta('CBp',alpha=0.5,beta=0.5,trace=True)
        CAp  = Beta('CAp',alpha=0.5,beta=0.5,trace=True)
    elif prior == 'Beta2':
        ABp  = Beta('ABp',alpha=2,beta=2,trace=True)
        CBp  = Beta('CBp',alpha=2,beta=2,trace=True)
        CAp  = Beta('CAp',alpha=2,beta=2,trace=True)
    elif prior == 'Gamma':
        ABp  = Gamma('ABp',alpha=1,beta=0.5,trace=True)
        CBp  = Gamma('CBp',alpha=1,beta=0.5,trace=True)
        CAp  = Gamma('CAp',alpha=1,beta=0.5,trace=True)
    
    AB1  = ABp
    AB3  = 1-ABp
    CB4  = CBp
    CB5  = 1-CBp
    CA42 = CAp
    CA52 = 1-CAp
    
    b = Normal('b',mu=f_AB*AB3+ f_CB*CB4 + f_CA*CA42, tau=tau,value=b_obs,
            observed=True,trace=True)
    
    # print [x.value for x in [ABp,CBp,CAp]]
    # print b.logp

    return locals()

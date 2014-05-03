from pymc import DiscreteUniform, Exponential, deterministic, Poisson, Uniform, stochastic, Bernoulli, Normal, MCMC, Beta, Gamma
import numpy as np

prior = 'Gamma'

if prior == 'Normal':
    ABp  = Normal('ABp',mu=0.5,tau=100)
    CBp  = Normal('CBp',mu=0.5,tau=100)
    CAp  = Normal('CAp',mu=0.5,tau=100)
elif prior == 'Uniform':
    ABp  = Uniform('ABp',lower=0.0,upper=1.0)
    CBp  = Uniform('CBp',lower=0.0,upper=1.0)
    CAp  = Uniform('CAp',lower=0.0,upper=1.0)
elif prior == 'Beta':
    ABp  = Beta('ABp',alpha=0.5,beta=0.5)
    CBp  = Beta('CBp',alpha=0.5,beta=0.5)
    CAp  = Beta('CAp',alpha=0.5,beta=0.5)
elif prior == 'Gamma':
    ABp  = Gamma('ABp',alpha=1,beta=0.5)
    CBp  = Gamma('CBp',alpha=1,beta=0.5)
    CAp  = Gamma('CAp',alpha=1,beta=0.5)

AB1  = ABp
AB3  = 1-ABp
CB4  = CBp
CB5  = 1-CBp
CA42 = CAp
CA52 = 1-CAp

b = Normal('b',mu=400*AB3+ 1000*CB4 + 600*CA42, tau=10000,value=200,observed=True)

print [x.value for x in [ABp,CBp,CAp]]
print b.logp


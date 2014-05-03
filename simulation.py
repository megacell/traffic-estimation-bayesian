from pymc import MCMC
from pymc.Matplot import plot
import numpy as np

import small_model

A = MCMC(small_model)
A.sample(iter=50000)
plot(A,suffix='-gamma')

print '%s prior' % small_model.prior
print [(x,A.stats()[x]['mean']) for x in A.stats()]
error = (1 - A.stats()['ABp']['mean'])*400 + A.stats()['CAp']['mean']*600 + A.stats()['CBp']['mean']*1000 - 200
print 'Error: %s' % error

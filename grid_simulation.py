from pymc import MCMC
from pymc.Matplot import plot
import numpy as np
from itertools import chain

# load model
import grid_model as model

# run simulation
A = MCMC(model)
A.sample(iter=50000)
plot(A,suffix='-grid')

# construct solution vector
x_ans = sorted([(i,list(A.stats()[i]['mean'])) for i in A.stats()], key=lambda x: x[0])
[x[1].append(1-sum(x[1])) for x in x_ans]
x_ans = list(chain(*[x[1] for x in x_ans]))

print [(x,A.stats()[x]['mean']) for x in A.stats()]

error_b = np.linalg.norm(model.A.dot(np.array(x_ans)) - model.b_obs[:,0])
print "norm(Ax-b): %s" % error_b
print np.vstack((model.A.dot(np.array(x_ans)),model.b_obs[:,0]))

error_x = np.linalg.norm(model.x_true[:,0]-np.array(x_ans))
print "norm(x-x*): %s" % error_x
print np.vstack((np.array(x_ans),model.x_true[:,0]))

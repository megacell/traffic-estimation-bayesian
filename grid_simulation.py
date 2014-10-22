import pymc as pm
from pymc import HamiltonianMC as MC
# from pymc.Matplot import plot
import numpy as np
from itertools import chain
from matplotlib import pyplot as plt
import cPickle as pickle

# load model
import grid_model as model

# construct solution vector
def getX(A):
    x_ans = sorted([(i,list(A.stats()[i]['mean'])) for i in A.stats()], \
            key=lambda x: x[0])
    [x[1].append(1-sum(x[1])) for x in x_ans]
    x_ans = list(chain(*[x[1] for x in x_ans]))
    return x_ans

def sample(A,iters=100,logp=[],errors_b=[],errors_x=[]):
    for i in range(iters):
        A.sample(iter=1)
        x_ans = getX(A)
        error_b = np.linalg.norm(model.A.dot(np.array(x_ans)) - model.b_obs[:,0])
        error_x = np.linalg.norm(model.x_true[:,0]-np.array(x_ans))
        logp.append(A.logp)
        errors_b.append(error_b)
        errors_x.append(error_x)
        if i % 50 == 0:
            print i

    x_ans = getX(A)
    print [(x,A.stats()[x]['mean']) for x in A.stats()]

    error_b = np.linalg.norm(model.A.dot(np.array(x_ans)) - model.b_obs[:,0])
    print "norm(Ax-b): %s" % error_b
    print np.vstack((model.A.dot(np.array(x_ans)),model.b_obs[:,0]))

    error_x = np.linalg.norm(model.x_true[:,0]-np.array(x_ans))
    print "norm(x-x*): %s" % error_x
    print np.vstack((np.array(x_ans),model.x_true[:,0]))
 
    return A, logp, errors_b, errors_x

def plot(logp, errors_b, errors_x):
    plt.figure(1)
    plt.subplot(221)
    plt.plot(range(len(logp)),logp)
    plt.title('Log likelihood')
    plt.ylabel('Log likelihood')
    plt.xlabel('Sample')
    plt.subplot(222)
    plt.plot(range(len(errors_b)),errors_b)
    plt.title('Objective')
    plt.ylabel('norm(Ax-b)')
    plt.xlabel('Sample')
    plt.subplot(223)
    plt.plot(range(len(errors_x)),errors_x)
    plt.title('Recovery')
    plt.ylabel('norm(x-x*)')
    plt.xlabel('Sample')
    plt.show()

def save(fmetaname,logp,errors_b,errors_x):
    with open(fmetaname,'wb') as f:
        pickle.dump((logp,errors_b,errors_x), f)

if __name__ == "__main__":
    # fname = '%s.pickle' % model.fname
    # fmetaname = '%s_meta.pickle' % model.fname

    # try:
    #     # load previous simulation
    #     db = pm.database.pickle.load(fname)
    #     A = MC(model, db=db)
    #     with open(fmetaname,'r') as f:
    #         logp, errors_b, errors_x = pickle.load(f)
    # except IOError:
    #     # run new simulation
    #     A = MC(model, db='pickle', dbname=fname)
    #     logp = []
    #     errors_b = []
    #     errors_x = []
    #     A.sample(iter=100)
    with model.model:
        start = pm.find_MAP()
        trace = pm.sample(100000,pm.HamiltonianMC(),start)
        # trace = pm.sample(100000,pm.Metropolis(),start)
        # trace = pm.sample(300,pm.Slice(),start)
        fig = pm.traceplot(trace)
        plt.show()

    import ipdb
    ipdb.set_trace()

    # A.sample(iter=50000)
    # plot(A,suffix='-grid')

    # A, logp, errors_b, errors_x = sample(A,iters=300,logp=logp,\
    #         errors_b=errors_b,errors_x=errors_x)
    # save(fmetaname,logp,errors_b,errors_x)

    # plot(logp, errors_b, errors_x)

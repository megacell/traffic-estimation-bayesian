import pymc as pm
from pymc import HamiltonianMC as MC
# from pymc.Matplot import plot
import numpy as np
from itertools import chain
from matplotlib import pyplot as plt
import cPickle as pickle

# load model
from grid_model import create_model

# construct solution vector
def getX(A):
    x_ans = sorted([(i,list(A.stats()[i]['mean'])) for i in A.stats()], \
            key=lambda x: x[0])
    [x[1].append(1-sum(x[1])) for x in x_ans]
    x_ans = list(chain(*[x[1] for x in x_ans]))
    return x_ans

def error(trace,A,x_true,b_obs,scaling,output=None):
    if output is None:
        output = {}
    x_blocks = None
    for varname in sorted(trace.varnames):
        # flatten the trace and normalize
        if trace.get_values(varname).shape[1] == 0:
            continue
        x_block = np.array([x/sum(x) for x in trace.get_values(varname)])
        if x_blocks is not None:
            x_blocks = np.hstack((x_blocks, x_block))
        else:
            x_blocks = x_block
    # compute link flow and route flow error
    n = x_blocks.shape[0]

    b = A.dot(x_blocks.T)
    error_b = 0.5 * np.linalg.norm(b - np.tile(b_obs,(n,1)).T,axis=0) ** 2

    x_true_block = np.tile(x_true,(n,1)).T
    x_diff = x_true_block-x_blocks.T

    scaling_block = np.tile(scaling,(n,1)).T
    x_diff_scaled = scaling_block * x_diff
    x_true_scaled = scaling_block * x_true_block
    dist_from_true = np.max(x_diff_scaled,axis=0)

    output['incorrect x entries'] = np.bincount(np.where(x_diff > 1e-3)[1])
    per_flow = np.sum(x_diff_scaled, axis=0) / np.sum(x_true_scaled, axis=0)
    output['percent flow allocated incorrectly'] = per_flow
    output['max|f * (x-x_true)|'] = dist_from_true

    error_x = 0.5 * np.linalg.norm(x_diff,axis=0) ** 2
    output['error_x'] = error_x
    output['error_b'] = error_b

    import ipdb
    ipdb.set_trace()
    return error_b, error_x, output

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

def plot(error_b, error_x):
    plt.figure()
    plt.subplot(121)
    plt.plot(range(len(error_b)),error_b)
    plt.title('Objective')
    plt.ylabel('norm(Ax-b)')
    plt.xlabel('Sample')
    plt.subplot(122)
    plt.plot(range(len(error_x)),error_x)
    plt.title('Recovery')
    plt.ylabel('norm(x-x*)')
    plt.xlabel('Sample')
    plt.show()

def save(fmetaname,logp,errors_b,errors_x):
    with open(fmetaname,'wb') as f:
        pickle.dump((logp,errors_b,errors_x), f)

def MCMC(model):
    import time
    with model:
        n = 6000
        START = time.time()
        try:
            start = pm.find_MAP()
        except AssertionError:
            return model, {'error':'AssertionError in pm.find_MAP()'}
        print 'Time to initialize: %ds' % (time.time()-START)

        START = time.time()
        trace = pm.sample(n,pm.Metropolis(),start)
        print 'Time to sample (MH): %ds' % (time.time()-START)

        # START = time.time()
        # trace = pm.sample(n,pm.Slice(),start)
        # print 'Time to sample (Slice): %ds' % (time.time()-START)

        # START = time.time()
        # trace = pm.sample(n,pm.HamiltonianMC(),start)
        # print 'Time to sample (HMC): %ds' % (time.time()-START)

        error_b, error_x, output = error(trace,model.data.A,model.data.x_true,
                                 model.data.b_obs,model.data.scaling)

        # fig = pm.traceplot(trace)
        # plot(error_b,error_x)
        # plt.show()
    return model, output

if __name__ == "__main__":
    fname = 'data/2_3_3_1_20140421T151732_1_small_graph_OD_dense.mat'
    sparse = False
    model = create_model(fname,sparse)
    trace = MCMC(model)

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

    # A.sample(iter=50000)
    # plot(A,suffix='-grid')

    # A, logp, errors_b, errors_x = sample(A,iters=300,logp=logp,\
    #         errors_b=errors_b,errors_x=errors_x)
    # save(fmetaname,logp,errors_b,errors_x)

    # plot(logp, errors_b, errors_x)

import pymc
from pymc import MCMC
# from pymc.Matplot import plot
import numpy as np
from itertools import chain
from matplotlib import pyplot as plt
import cPickle as pickle

def logp_trace(model):
    """
    return a trace of logp for model
    src: https://groups.google.com/forum/#!searchin/pymc/imri$20sofer/pymc/
            u9v3XPOMWTY/vWVXBHuRVGkJ
    """

    #init
    db = model.db
    n_samples = db.trace('deviance').length()
    logp = np.empty(n_samples, np.double)

    #loop over all samples
    for i_sample in xrange(n_samples):
        #set the value of all stochastic to their 'i_sample' value
        for stochastic in model.stochastics:
            try:
                value = db.trace(stochastic.__name__)[i_sample]
                stochastic.value = value

            except KeyError:
                print "No trace available for %s. " % stochastic.__name__

        #get logp
        logp[i_sample] = model.logp

    return logp


# construct solution vector
def getX(A):
    x_ans = sorted([(i,list(A.stats()[i]['mean'])) for i in A.stats()], \
            key=lambda x: x[0])
    [x[1].append(1-sum(x[1])) for x in x_ans]
    x_ans = list(chain(*[x[1] for x in x_ans]))
    return x_ans

def sample(model,A,fmetaname,iters=100,logp=[],errors_b=[],errors_x=[]):
    for i in range(iters):
        A.sample(iter=1)
        x_ans = getX(A)
        error_b = np.linalg.norm(model['A'].dot(np.array(x_ans)) - \
                model['b_obs'][:,0])
        error_x = np.linalg.norm(model['x_true'][:,0]-np.array(x_ans))
        logp.append(A.logp)
        errors_b.append(error_b)
        errors_x.append(error_x)
        if i % 50 == 0:
            save(fmetaname,logp,errors_b,errors_x)
            print 'iter: %s, error_b: %s, error_x: %s' % (i,error_b,error_x)
        if error_x <= 0.002:
            print 'logp: %s, error_b: %s, error_x: %s' % (logp, error_b, error_x)
            print "norm(Ax-b): %s" % error_b
            print np.vstack((model['A'].dot(np.array(x_ans)),model['b_obs'][:,0]))

            print "norm(x-x*): %s" % error_x
            print np.vstack((np.array(x_ans),model['x_true'][:,0]))

    save(fmetaname,logp,errors_b,errors_x)

    # print [(x,A.stats()[x]['mean']) for x in A.stats()]

    print "norm(Ax-b): %s" % error_b
    print np.vstack((model['A'].dot(np.array(x_ans)),model['b_obs'][:,0]))

    print "norm(x-x*): %s" % error_x
    print np.vstack((np.array(x_ans),model['x_true'][:,0]))

    return A, logp, errors_b, errors_x

def sample_toy(model,A,fmetaname,iters=100,logp=[],errors_b=[],errors_x=[]):
    print '%s prior' % model['prior']

    for i in range(iters):
        A.sample(iter=1)
        x_ans = sorted([(j,[A.stats()[j]['mean']]) for j in A.stats()], \
                key=lambda x: x[0])
        [x[1].append(1-sum(x[1])) for x in x_ans]
        x_ans = list(chain(*[x[1] for x in x_ans]))
        error_b = (1 - A.stats()['ABp']['mean'])*model['f_AB'] + \
                A.stats()['CAp']['mean']*model['f_CA'] + \
                A.stats()['CBp']['mean']*model['f_CB'] - model['b_obs']
        logp.append(A.logp)
        errors_b.append(error_b)
        errors_x.append(x_ans) # FIXME storing full x_ans, not errors_x
        if i % 50 == 0:
            save(fmetaname,logp,errors_b,errors_x)
            print 'iter: %s, error_b: %s' % (i,error_b)

    save(fmetaname,logp,errors_b,errors_x)

    print 'logp: %s, error_b: %s' % (A.logp, error_b)
    print "norm(Ax-b): %s" % error_b

    print "x:"
    print x_ans

    return A, logp, errors_b, errors_x

def sample_toy_save(model,A,iters=100,verbose=True,save_interval=None):
    if verbose:
        print '%s prior' % model['prior']
    A.sample(iter=iters,save_interval=save_interval)
    x_ans = sorted([(j,np.vstack((A.trace(j).gettrace(),1-np.sum(np.atleast_2d(A.trace(j).gettrace()),axis=0)))) for j in A.stats()], \
            key=lambda x: x[0])
    x_ans = np.vstack((x[1] for x in x_ans))
    errors_b = np.linalg.norm(np.atleast_2d(model['A'].dot(x_ans) - model['b_obs']),axis=0)
    errors_x = x_ans
    if verbose:
        error_b = (1 - A.stats()['ABp']['mean'])*model['f_AB'] + \
                A.stats()['CAp']['mean']*model['f_CA'] + \
                A.stats()['CBp']['mean']*model['f_CB'] - model['b_obs']
        print 'error_b: %s error_b2: %s' % (error_b, errors_b[-1])
        print errors_x[:,-1]
    # logp = logp_trace(A)
    logp = [A.logp]
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

def make_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Traffic model selection')
    parser.add_argument('--type', metavar='type', type=str, default='GRID',
                        help='Type of model: [GRID] or TOY or TOYSAVE')
    parser.add_argument('--prior', metavar='prior', type=str, default='Beta0.5',
                        help='Type of model: [Beta0.5] or Uniform or Normal')
    parser.add_argument('--file', metavar='file', type=str, 
                        default='3_3_3_1_20140421T173515_5_small_graph_OD',
                        help='Data file')
    parser.add_argument('--iters', metavar='iters', type=int, default=0,
                        help='Sample iterations')
    parser.add_argument('--trials', metavar='trials', type=int, default=100,
                        help='Number of trials')
    parser.add_argument('--tau', metavar='tau', type=float, default=10000,
                        help='Noise: tau=1/sigma')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # parse command line arguments
    args = make_parser()
    filename = args.file
    iters = args.iters
    tau = args.tau

    # select appropriate model
    import models
    if args.type == 'GRID':
        model = models.grid_model(filename,tau=tau)
        fname = '%s_%s.pickle' % (filename,tau)
        fmetaname = '%s_%s_meta.pickle' % (filename,tau)
    elif args.type == 'TOY':
        prior = args.prior
        model = models.toy_model(tau=tau,prior=prior)
        fname = '%s_%s_%s.pickle' % (filename,tau,prior)
        fmetaname = '%s_%s_%s_meta.pickle' % (filename,tau,prior)
    elif args.type == 'TOYSAVE':
        prior = args.prior
        model = models.toy_model(tau=tau,prior=prior)
        fname = None
        # fname = '%s_%s_%s_save.pickle' % (filename,tau,prior)

    try:
        # load previous model
        db = pymc.database.pickle.load(fname)
        A = MCMC(model, db=db)
        if args.type != 'TOYSAVE':
            with open(fmetaname,'r') as f:
                logp, errors_b, errors_x = pickle.load(f)
        print "Loaded previous model from %s" % fname
    except (IOError, TypeError) as e:
        # run new simulation
        if fname:
            A = MCMC(model, db='pickle', dbname=fname)
        else:
            A = MCMC(model)
        logp = []
        errors_b = []
        errors_x = []
        print "Created new model at %s" % fname
        A.sample(iter=100)

    if args.type == 'GRID':
        A, logp, errors_b, errors_x = sample(model,A,fmetaname,iters=iters, \
                logp=logp,errors_b=errors_b,errors_x=errors_x)
        plot(logp, errors_b, errors_x)
    elif args.type == 'TOY':
        A, logp, errors_b, errors_x = sample_toy(model,A,fmetaname,iters=iters, \
                logp=logp,errors_b=errors_b,errors_x=errors_x)
        plot(logp, errors_b, errors_x)
    elif args.type == 'TOYSAVE':
        A, logp, errors_b, errors_x  = sample_toy_save(model,A,iters=iters)



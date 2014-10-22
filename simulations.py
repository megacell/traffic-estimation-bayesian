import simulation
from pymc import MCMC
import numpy as np
import models

def run_trials(trials=0,iters=0,tau=10000,prior=None,errort_b=[],Linf=[],sparsity=[],logps=[]):
    for i in range(trials):
        # NOTE need to create new model per iteration, pymc might be using
        # the model instance to seed a random number generator somewhere...
        model = models.toy_model(tau=tau,prior=prior)
        A = MCMC(model)
        A, logp, errors_b, errors_x = simulation.sample_toy_save(model,A, \
                iters=iters,verbose=False)
        logps.append(logp[-1])
        errort_b.append(errors_b[-1])
        Linf.append(np.sum(1/errors_x[:,-1][errors_x[:,-1].argsort()[-3:]]))
        sparsity.append(np.sum(errors_x[:,-1] <= 0.02))

    print ''
    for (i,j) in [('Linf',Linf),('Sparsity',sparsity),('Error_b',errort_b),('logp',logps)]:
        print i
        if i == 'Sparsity':
            print [np.sum(np.array(j) == k) for k in range(4)]
        else:
            (H1,H2) = np.histogram(j)
            print H1
            print H2
        print "Median: %s" % (np.median(j))
    return Linf,sparsity,errort_b,logps

if __name__ == "__main__":
    args = simulation.make_parser()
    filename = args.file
    iters = args.iters
    tau = args.tau
    trials = args.trials
    prior = args.prior

    sparsity = []
    Linf = []
    errort_b = []
    logps = []

    Linf,sparsity,errort_b,logps = run_trials(trials=trials,iters=iters, \
            tau=tau, prior=prior,errort_b=errort_b, Linf=Linf, \
            sparsity=sparsity, logps=logps)

    
    

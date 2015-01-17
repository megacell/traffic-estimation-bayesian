import ipdb
from pymc import DiscreteUniform, Exponential, Poisson, \
        Uniform, Bernoulli, Normal, Beta, Gamma, Dirichlet
import pymc as pm
import numpy as np
from scipy.io import loadmat
from itertools import chain
from scipy.sparse import diags

# dot product for weighted probability distributions
def dot(a,b):
    return sum([x*y for (x,y) in zip(a,b)])

# Clean array wrapper
def array(x):
    return np.atleast_1d(np.squeeze(np.array(x)))

def has_OD(data,OD):
    return OD and 'T' in data and 'd' in data and data['T'] is not None and \
           data['d'] is not None and data['T'].size > 0 and data['d'].size > 0

def has_CP(data,CP):
    return CP and 'U' in data and 'f' in data and data['U'] is not None and \
           data['f'] is not None and data['U'].size > 0 and data['f'].size > 0

def has_LP(data,LP):
    return LP and 'V' in data and 'g' in data and data['V'] is not None and \
           data['g'] is not None and data['V'].size > 0 and data['g'].size > 0

def generate_route_flows_from_incidence_matrix(M,alpha=1):
    """
    Generate blocks of route flows/splits (x) from an incidence matrix
    :param M: incidence matrix
    :return:
    """
    x_blocks = []
    order, block_sizes = [], []
    m,n = M.shape
    assert M.getnnz() == n
    # Construct a Gamma distribution for each row in the incidence matrix
    for i in xrange(m):
        block_ind = M.getrow(i).nonzero()[1]
        order.extend(block_ind)
        size = len(block_ind)
        block_sizes.append(size)

        block = Gamma('x%d' % i,np.array([alpha]*size), 1, shape=size)
        x_blocks.append(block)

    x_blocks_expanded = [[x[xi]/x.sum() for xi in range(i-1)] for (i,x) in \
                         zip(block_sizes,x_blocks)]
    [x.append(1 - sum(x)) for x in x_blocks_expanded]
    x_pri = list(chain(*x_blocks_expanded))
    # reorder
    x_pri = zip(*sorted(zip(x_pri,order),key=lambda x: x[1]))[0]
    return x_pri

def load_model(fname,sparse,full=False,OD=False,CP=False):
    # load data
    # fname = '2_3_3_1_20140421T151732_1_small_graph_OD_dense'
    # fname = '3_3_3_1_20140421T173515_5_small_graph_OD'
    # fname = '4_6_3_1_20140421T155253_1_small_graph_OD'
    data = loadmat(fname)
    return create_model(data,sparse,full=full,OD=OD,CP=CP)

def create_model(AA,bb_obs,EQ,x_true,sparse=False):
    output = {}
    # change variable names
    # sparse = True

    alpha = 0.3 if sparse else 1

    # construct graphical model
    # -----------------------------------------------------------------------------
    import time
    with pm.Model() as model:
        ivar = 10000

        START = time.time()
        # construct sparse prior on all routes
        # Dirichlet distribution doesn't give values that sum to 1 (continuous distribution), so
        # instead we normalize draws from a Gamma distribution
        # CAUTION x_pri is route splits
        x_pri = array(generate_route_flows_from_incidence_matrix(EQ,alpha=alpha))

        # construct skinny normal distributions with observations
        #FIXME sparse dot product (i.e. Mscale.dot(x_pri)) gives error:
        # TypeError: no supported conversion for types: (dtype('float64'), dtype('O'))
        mus_bb = array(AA.todense().dot(x_pri))
        bb = [Normal('b%s' % i, mu=mu, tau=ivar, observed=obsi) for \
             (i,(mu,obsi)) in enumerate(zip(mus_bb,bb_obs))]

        print 'Time to build model: %ds' % (time.time()-START)

    return model,alpha,x_pri

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="Model file (.mat)",
                        default="data/2_3_3_1_20140421T151732_1_small_graph_OD_dense.mat")
    parser.add_argument('--sparse', type=bool, help='Sparse?', default=False)
    args = parser.parse_args()

    model = create_model(args.model, args.sparse)
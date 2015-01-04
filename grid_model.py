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

def create_model(data,sparse,full=False,OD=False,CP=False,EQ='OD'):
    # change variable names
    # sparse = True

    if sparse == True:
        alpha = 0.3
    else:
        alpha = 1

    # Link-route
    if full and 'A_full' in data and 'b_full' in data:
        A, b_obs = data['A_full'], data['b_full'].squeeze()
    else:
        A, b_obs = data['A'], data['b'].squeeze()

    # OD-route
    if OD and 'T' in data and 'd' in data:
        T,d_obs = data['T'], data['d'].squeeze()
    # Cellpath-route
    if CP and 'U' in data and 'f' in data:
        U,f_obs = data['U'], data['f'].squeeze()

    # Route flows
    # TODO need to distinguish between route split and route flows here
    x_true = data['x_true'].squeeze()

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
        if EQ=='OD' and OD:
            N, M = T, U
        elif EQ=='CP' and CP:
            N, M = U, T

        scaling = array(diags([N.dot(x_true)],[0]).dot(N).sum(axis=0))
        nz = scaling.nonzero()[0]
        z = np.where(scaling == 0)[0]
        x_pri = array(generate_route_flows_from_incidence_matrix(N[:,nz],
                                                            alpha=alpha))
        scaling = scaling[nz]

        # construct skinny normal distributions with observations
        # mus = np.array([dot(a,x_pri) for a in A])
        Dx = diags([scaling],[0])
        Ascale, Mscale = Dx.dot(A[:,nz].T).T, Dx.dot(M[:,nz].T).T
        # Dx = diags([x_true],[0])
        # Ascale, Uscale = Dx.dot(A.T).T, Dx.dot(U.T).T
        #FIXME sparse dot product (i.e. Mscale.dot(x_pri)) gives error:
        # TypeError: no supported conversion for types: (dtype('float64'), dtype('O'))
        mus_b, mus_m = Ascale.dot(x_pri), Mscale.todense().dot(x_pri)
        b = [Normal('b%s' % i, mu=mu, tau=ivar, observed=obsi) for \
             (i,(mu,obsi)) in enumerate(zip(mus_b,b_obs))]

        if EQ=='OD' and OD:
            f = [Normal('f%s' % i, mu=mu, tau=ivar, observed=obsi) for \
                 (i,(mu,obsi)) in enumerate(zip(array(mus_m),f_obs))]
        elif EQ=='CP' and CP:
            d = [Normal('f%s' % i, mu=mu, tau=ivar, observed=obsi) for \
                 (i,(mu,obsi)) in enumerate(zip(array(mus_m),d_obs))]

        print 'Time to build model: %ds' % (time.time()-START)

    if OD and CP:
        model.data = type('data',(object,),dict(A=A,b_obs=b_obs,x_true=x_true,
                                                T=T,d=d_obs,U=U,f=f_obs,
                                                sparse=sparse,alpha=alpha))
    elif OD:
        model.data = type('data',(object,),dict(A=A,b_obs=b_obs,x_true=x_true,
                                                T=T,d=d_obs,sparse=sparse,
                                                alpha=alpha))

    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="Model file (.mat)",
                        default="data/2_3_3_1_20140421T151732_1_small_graph_OD_dense.mat")
    parser.add_argument('--sparse', type=bool, help='Sparse?', default=False)
    args = parser.parse_args()

    model = create_model(args.model, args.sparse)
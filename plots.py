import numpy as np
from itertools import chain
from matplotlib import pyplot as plt

plot = 3

keys = ['Beta0.25', 'Beta0.5', 'Uniform', 'Beta2']
labels = { 'Beta0.25':  'Dir(.25,.25)',
           'Beta0.5':   'Dir(.5,.5)', 
           'Beta2':     'Dir(2,2)', 
           'Uniform':   'Dir(1,1)' }

# 1: TOY EXAMPLE PLOT (iterations | 5e-2 median error over 500 trials, tau=1000)
# -----------------------------------------------------------------------------
if plot == 1:
    iterations = {}
    error_b = {}

    iterations['Beta0.25'] = [100,200,200,200,300,400,500,600,700,900,1000,1200,1500,1500,2000,3000,3000,4000,10000,15000][1:-2]
    error_b['Beta0.25'] = [0.772516742398,0.276881503785,0.232909079915,0.183427382195,0.1370544599,0.0836404551013,0.0685366598249,0.0529058185754,0.0464342624286,0.0436589773205,0.0420679592406,0.03601986035,0.0333101696796,0.0313822167442,0.0313528559212,0.0226317810362,0.0250782336334,0.0239491029495,0.0244307192305,0.0229277707392][1:-2]
    
    iterations['Beta0.5'] = [100,200,200,200,300,400,600,700,725,725,730,735,737,740,750,775,800,900,1000,1500,2000,3000,4000][1:]
    error_b['Beta0.5'] = [0.681623193246,0.17384367673,0.21090271096,0.179442586499,0.126975764903,0.09007584227,0.0659174150722,0.0521048090564,0.0507495730519,0.0519824018794,0.0503060968414,0.0521176986468,0.0550987853492,0.0451922132524,0.0489344656963,0.041875841297,0.0450483228706,0.0462957823395,0.0472659528832,0.0290561927993,0.0269675489676,0.0223419004686,0.0222932248784][1:]
    
    iterations['Uniform'] = [200,400,600,800,900,1000,1025,1030,1031,1033,1035,1050,1100,2000,3000,4000]
    error_b['Uniform'] = [0.301965758752,0.120241711932,0.0919477651099,0.0654998831757,0.0701055704467,0.05458669974,0.0561581797206,0.0572948503443,0.0541415580769,0.0483023412327,0.0479711223111,0.0436755234095,0.043617781336,0.0279527480325,0.0229206583982,0.0213791779972]
    
    iterations['Beta2'] = [200,400,600,800,1000,1200,1200,1215,1217,1218,1220,1225,1250,1300,1500,2000,3000,4000]
    error_b['Beta2'] = [0.338142289951,0.161319129575,0.106643128828,0.083821631449,0.0701327727482,0.0520181644772,0.0524719565683,0.0530891276948,0.052852042438,0.0463487650544,0.0439466976466,0.04586270677,0.0425690761619,0.0464749679205,0.0411034544811,0.0318449064482,0.0225671197244,0.0205472812533]
    
    plt.clf()
    plt.figure(1)
    plt.title('Iterations vs norm error (tau=1000)')
    for i in keys:
        x = iterations[i]
        y = error_b[i]
        line, = plt.loglog(x,y,'.')
        logx = np.log(x)
        logy = np.log(y)
        coeffs = np.polyfit(logx,logy,deg=2)
        poly = np.poly1d(coeffs)
        yfit = lambda x: np.exp(poly(np.log(x)))
        plt.hold(True)
        new_x = np.linspace(x[0],x[-1],100)
        plt.loglog(new_x,yfit(new_x),color=line.get_color(),label=labels[i])
        plt.ylabel('Median norm error')
        plt.xlabel('Iterations sampled')
    plt.loglog([100,10000],[5e-2,5e-2],'-',label='5e-2 norm error')
    plt.loglog([100,10000],[2.3e-2,2.3e-2],'-',label='2e-2 norm error')
    plt.legend(loc='best')
    plt.show()

# 2: TOY EXAMPLE PLOT (norm error vs noise | fixed iterations )
# -----------------------------------------------------------------------------
elif plot == 2:
    # iters = 3000, trials = 500
    sparsity = {}
    tau = {}
    error_b = {}
    Linf = {}

    tau['Beta0.25'] = [10,100,500,1000,5000,10000]
    error_b['Beta0.25'] = [0.240124731159,0.0776185006915,0.0340892879798,0.0250782336334,0.0125679878323,0.00772820638083]
    Linf['Beta0.25'] = [3.42726746436,3.43235092438,3.44992993403,3.44178839465,3.44298121179,3.43667993353]
    sparsity['Beta0.25'] = [[124, 200, 154, 22],[112, 231, 145, 12],[130, 193, 165, 12],[127, 218, 134, 21],[138, 204, 138, 20]]
 
    tau['Beta0.5'] = [10,100,500,1000,5000,10000]
    error_b['Beta0.5'] = [0.224259897383,0.0673011191353,0.0289372859655,0.0223419004686,0.0105740545606,0.00764495562504]
    Linf['Beta0.5'] = [3.43153513853,3.42845304587,3.42733037321,3.42571154145,3.4269016833,3.43008965264]
    sparsity['Beta0.5'] = [[204, 214, 81, 1],[223, 201, 76, 0],[207, 194, 97, 2],[217, 202, 80, 1],[197, 218, 84, 1],[198, 214, 85, 3]]
 
    tau['Uniform'] = [10,100,500,1000,5000,10000]
    error_b['Uniform'] = [0.227409713576,0.0718294836837,0.0270671978541,0.0229206583982,0.0101172891676,0.00892673619705]
    Linf['Uniform'] = [3.40104791563,3.4077065517,3.39128135863,3.41060152062,3.41112377265,3.40030176294]
    sparsity['Uniform'] = [[246, 215, 39, 0],[246, 210, 44, 0],[283, 179, 38, 0],[270, 186, 44, 0],[261, 199, 40, 0],[278, 188, 34, 0]]
 
    tau['Beta2'] = [10,100,500,1000,5000,10000]
    error_b['Beta2'] = [0.218358058339,0.066356963061,0.031230571556,0.0225671197244,0.0106341332323,0.00747702283357]
    Linf['Beta2'] = [3.39698745811,3.38280184143,3.39274470959,3.39793344334,3.40120070743,3.37456714221]
    sparsity['Beta2'] = [[295, 186, 19, 0],[296, 180, 24, 0],[299, 180, 21, 0],[289, 187, 24, 0],[286, 190, 24, 0],[310, 170, 20, 0]]

    plt.clf()
    plt.figure(1)
    plt.title('Noise vs norm error (iter=3000)')
    for i in keys:
        x = tau[i]
        y = error_b[i]
        line, = plt.loglog(x,y,'.')
        logx = np.log(x)
        logy = np.log(y)
        coeffs = np.polyfit(logx,logy,deg=2)
        poly = np.poly1d(coeffs)
        yfit = lambda x: np.exp(poly(np.log(x)))
        plt.hold(True)
        new_x = np.linspace(x[0],x[-1],100)
        plt.loglog(new_x,yfit(new_x),color=line.get_color(),label=labels[i])
        plt.ylabel('Median norm error')
        plt.xlabel('Inv variance')
    plt.legend(loc='best')
    plt.show()
    # variance affects all priors the same way at iter=3000, so we may compare the priors at different variances

    plt.clf()
    plt.figure(1)
    plt.title('Noise vs sparsity (iter=3000)')
    for i in keys:
        x = tau[i]
        y = Linf[i]
        logx = np.log(x)
        line, = plt.plot(logx,y,'.')
        coeffs = np.polyfit(logx,y,deg=1)
        poly = np.poly1d(coeffs)
        yfit = lambda x: poly(np.log(x))
        plt.hold(True)
        new_x = np.linspace(x[0],x[-1],100)
        plt.plot(np.log(new_x),yfit(new_x),color=line.get_color(),label=labels[i])
        plt.ylabel('1/Linf norm error')
        plt.xlabel('Log inv variance')
    plt.plot([2,10],[3.55,3.55],'-',label='min L2 solution')
    plt.plot([2,10],[3.25,3.25],'-',label='min block inv Linf soln')
    (ylim_lower,ylim_upper) = plt.ylim()
    plt.ylim(ylim_lower-.02,ylim_upper+.02)
    plt.legend(loc='best')
    plt.show()
    # this is counter to the theory of minimizing the block inv Linf norm

    plt.clf()
    plt.figure(1)
    plt.title('Noise vs sparsity (iter=3000)')
    colors = 'crgbykm'
    for n,i in enumerate(keys[::-1]):
        x = np.array(range(4))
        y = np.mean(np.array(sparsity[i]),axis=0)
        std = np.std(np.array(sparsity[i]),axis=0)
        plt.bar(x + (n-2)/5.,y,label=labels[i],width=0.15,color=colors[n],yerr=std)
        plt.hold(True)
        plt.ylabel('Count')
        plt.xlabel('# zeros in x')
    plt.gca().set_xticks(x)
    plt.gca().set_xticklabels(x)
    plt.legend(loc='best')
    plt.show()

# 3: GRID NETWORK PLOT (iterations for logp or Ax-b to converge vs noise | 2_3 network, 3_3 network)
# -----------------------------------------------------------------------------
elif plot == 3:
    import models
    import pymc
    from pymc import MCMC
    import cPickle as pickle
    taus = [1.0,10.0,100.0,1000.0,10000.0]
    filenames = ['2_3_3_1_20140421T151732_1_small_graph_OD','2_3_3_1_20140421T151732_1_small_graph_OD_dense','3_3_3_1_20140421T173515_5_small_graph_OD','3_3_3_1_20140421T173515_5_small_graph_OD_dense','4_6_3_1_20140421T155253_1_small_graph_OD','4_6_3_1_20140421T155253_1_small_graph_OD_dense']
    labels = {'2_3_3_1_20140421T151732_1_small_graph_OD':'2x3 sparse',
            '2_3_3_1_20140421T151732_1_small_graph_OD_dense':'2x3 dense',
            '3_3_3_1_20140421T173515_5_small_graph_OD':'3x3 sparse',
            '3_3_3_1_20140421T173515_5_small_graph_OD_dense':'3x3 dense',
            '4_6_3_1_20140421T155253_1_small_graph_OD':'4x6 sparse',
            '4_6_3_1_20140421T155253_1_small_graph_OD_dense':'4x6 dense'}
    # filenames = ['2_3_3_1_20140421T151732_1_small_graph_OD']
    # filenames = ['2_3_3_1_20140421T151732_1_small_graph_OD_dense']
    # filenames = ['3_3_3_1_20140421T173515_5_small_graph_OD']
    # filenames = ['3_3_3_1_20140421T173515_5_small_graph_OD_dense']
    # filenames = ['4_6_3_1_20140421T155253_1_small_graph_OD']
    # filenames = ['4_6_3_1_20140421T155253_1_small_graph_OD_dense']

    for filename in filenames:
        print filename
        plt.figure(1)
        plt.suptitle('%s grid network' % labels[filename])
        for tau in taus:
            model = models.grid_model(filename,tau=tau)
            fname = '%s_%s.pickle' % (filename,tau)
            fmetaname = '%s_%s_meta.pickle' % (filename,tau)
            try:
                # load previous model
                db = pymc.database.pickle.load(fname)
                A = MCMC(model, db=db)
                with open(fmetaname,'r') as f:
                    logp, errors_b, errors_x = pickle.load(f)

                plt.subplot(221)
                plt.plot(range(len(logp)),logp,label=tau)
                plt.title('Log likelihood')
                plt.ylabel('Log likelihood')
                plt.xlabel('Sample')
                plt.legend(loc='best')
                plt.subplot(222)
                plt.plot(range(len(errors_b)),errors_b,label=tau)
                plt.title('Objective')
                plt.ylabel('norm(Ax-b)')
                plt.xlabel('Sample')
                plt.subplot(223)
                plt.plot(range(len(errors_x)),errors_x,label=tau)
                plt.title('Recovery')
                plt.ylabel('norm(x-x*)')
                plt.xlabel('Sample')
                plt.hold(True)
            except (IOError, TypeError, ValueError, EOFError) as e:
                print e
        plt.gcf()
        plt.show()

    # For 3_3_dense, tau=10000 has super fast convergence

# 4: GRID NETWORK PLOT (iterations vs network size | tau=1000, tau=100) 
# -----------------------------------------------------------------------------
elif plot == 4:
    import models
    import pymc
    from pymc import MCMC
    import cPickle as pickle
    taus = [1.0,10.0,100.0,1000.0,10000.0]
    filenames = ['2_3_3_1_20140421T151732_1_small_graph_OD','2_3_3_1_20140421T151732_1_small_graph_OD_dense','3_3_3_1_20140421T173515_5_small_graph_OD','3_3_3_1_20140421T173515_5_small_graph_OD_dense','4_6_3_1_20140421T155253_1_small_graph_OD','4_6_3_1_20140421T155253_1_small_graph_OD_dense']
    labels = {'2_3_3_1_20140421T151732_1_small_graph_OD':'2x3 sparse',
            '2_3_3_1_20140421T151732_1_small_graph_OD_dense':'2x3 dense',
            '3_3_3_1_20140421T173515_5_small_graph_OD':'3x3 sparse',
            '3_3_3_1_20140421T173515_5_small_graph_OD_dense':'3x3 dense',
            '4_6_3_1_20140421T155253_1_small_graph_OD':'4x6 sparse',
            '4_6_3_1_20140421T155253_1_small_graph_OD_dense':'4x6 dense'}

    for tau in taus:
        plt.figure(1)
        for filename in filenames:
            model = models.grid_model(filename,tau=tau)
            fname = '%s_%s.pickle' % (filename,tau)
            fmetaname = '%s_%s_meta.pickle' % (filename,tau)
            try:
                # load previous model
                db = pymc.database.pickle.load(fname)
                A = MCMC(model, db=db)
                with open(fmetaname,'r') as f:
                    logp, errors_b, errors_x = pickle.load(f)

                plt.subplot(221)
                plt.plot(range(len(logp)),logp,label=labels[filename])
                plt.title('Log likelihood')
                plt.ylabel('Log likelihood')
                plt.xlabel('Sample')
                plt.subplot(222)
                plt.plot(range(len(errors_b)),errors_b,label=labels[filename])
                plt.title('Objective')
                plt.ylabel('norm(Ax-b)')
                plt.xlabel('Sample')
                plt.subplot(223)
                plt.plot(range(len(errors_x)),errors_x,label=labels[filename])
                plt.title('Recovery')
                plt.ylabel('norm(x-x*)')
                plt.xlabel('Sample')
                plt.hold(True)
            except (IOError, TypeError, ValueError, EOFError) as e:
                print e
        plt.gcf()
        plt.legend(loc='best')
        plt.show()

# 5: GRID NETWORK PLOT (iterations vs network size | tau=1000, tau=100) 
# -----------------------------------------------------------------------------
elif plot == 5:
    import models
    import pymc
    from pymc import MCMC
    import cPickle as pickle
    taus = [1.0,10.0,100.0,1000.0,10000.0]
    filenames = ['2_3_3_1_20140421T151732_1_small_graph_OD','2_3_3_1_20140421T151732_1_small_graph_OD_dense','3_3_3_1_20140421T173515_5_small_graph_OD','3_3_3_1_20140421T173515_5_small_graph_OD_dense','4_6_3_1_20140421T155253_1_small_graph_OD','4_6_3_1_20140421T155253_1_small_graph_OD_dense']
    labels = {'2_3_3_1_20140421T151732_1_small_graph_OD':'2x3 sparse',
            '2_3_3_1_20140421T151732_1_small_graph_OD_dense':'2x3 dense',
            '3_3_3_1_20140421T173515_5_small_graph_OD':'3x3 sparse',
            '3_3_3_1_20140421T173515_5_small_graph_OD_dense':'3x3 dense',
            '4_6_3_1_20140421T155253_1_small_graph_OD':'4x6 sparse',
            '4_6_3_1_20140421T155253_1_small_graph_OD_dense':'4x6 dense'}

    plt.figure(1)
    for tau in taus:
        for filename in filenames:
            model = models.grid_model(filename,tau=tau)
            fname = '%s_%s.pickle' % (filename,tau)
            fmetaname = '%s_%s_meta.pickle' % (filename,tau)
            try:
                # load previous model
                db = pymc.database.pickle.load(fname)
                A = MCMC(model, db=db)
                with open(fmetaname,'r') as f:
                    logp, errors_b, errors_x = pickle.load(f)

                plt.subplot(221)
                plt.plot(range(len(logp)),logp,label='%s %s' % (labels[filename],tau))
                plt.title('Log likelihood')
                plt.ylabel('Log likelihood')
                plt.xlabel('Sample')
                plt.subplot(222)
                plt.plot(range(len(errors_b)),errors_b,label='%s %s' % (labels[filename],tau))
                plt.title('Objective')
                plt.ylabel('norm(Ax-b)')
                plt.xlabel('Sample')
                plt.subplot(223)
                plt.plot(range(len(errors_x)),errors_x,label='%s %s' % (labels[filename],tau))
                plt.title('Recovery')
                plt.ylabel('norm(x-x*)')
                plt.xlabel('Sample')
                plt.hold(True)
            except (IOError, TypeError, ValueError, EOFError) as e:
                print e
    plt.gcf()
    plt.legend(loc='best')
    plt.show()

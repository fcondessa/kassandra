""" kassandra.py: Kassandra Prediction Engine

Tensor-based prediction using (recursive) convolutional bayesian networks.

"""

import numpy as np
from numpy.fft import fftn, ifftn
from operator import mul
from multiprocessing import Pool
from numpy.linalg import lstsq

__author__ = "Filipe Condessa"
__copyright__ = "Filipe Condessa, 2017"
__maintainer__ = "Filipe Condessa"
__email__ = "fcondessa@gmail.com"
__status__ = "prototype"

# functions that mplement parallel solving of linear problems
def funct(arg1):
    '''Wrapper for least squares so that works with single input'''
    return lstsq(arg1[0],arg1[1])[0]

def solve_problem(problem):
    '''Solves sequentially list of linear problems using funct wrapper for least squares'''
    solved_problem = [[] for _ in range(len(problem))]
    for iter_i in range(len(problem)):
        solved_problem[iter_i]  = funct(problem[iter_i])
    return solved_problem

def solve_parallel_problem(problem,num_processes = 4):
    '''Solves in parallel list of linear problems using funct wrapper for least squares'''
    try:
        pool = Pool(processes=num_processes)
        result = pool.map(funct, problem)
        pool.close()
        return result
    except OSError:
        try:
            pool.close()
        except:
            pass
        part_solved = [solve_problem(elem) for elem in chunks(problem, len(problem) / 2)]
        return [val for sublist in part_solved for val in sublist]

def chunks(l, n):
    '''Yield successive n-sized chunks from l.'''
    # code copied from stackoverflow
    for i in range(0, len(l), n):
        yield l[i:i + n]

class CBM_TENSOR():
    ''' Class that implements the Convolutional Bayesian Tensor'''
    def __init__(self):
        '''Initialization of class'''
        self.data = np.array([])
        self.fixed = {}
        self.num_filters = 0
        self.shape_filter = []
        self.normalize = True
        self.num_iters = 0
        self.act_initialization_th = 0.8
        self.th = 0
        self.n_pts = 0
        self.filt_norm = []
        self.robust_noise = False
        self.robust_noise_value = 1E-10
        self.hotstart = False
        self.norm_factor = 1E-30
        self.diff = []
        self.flip_activations = False
        self.percentile_th = 90
        self.project_filters_to_real = False
        self.flip_th = 0.85
        self.VERBOSE = False
        self.PARALLEL = True
    def fit(self, data, num_filters=4, shape_filter=[5,5,5], normalize=True, hotstart=False, inp_filters=np.array([]),
            num_iters=10, th=0.1):
        ''' Populate the CBM tensors'''
        self.data = data
        self.shape = data[0].shape
        self.sparsify_filter = False
        self.filter_th = 1E-2
        self.dim = len(data)
        self.num_filters = num_filters
        self.shape_filter = shape_filter
        self.normalize = normalize
        self.num_iters = num_iters
        self.th = th
        self.filt_norm = []
        self.n_pts = reduce(mul,self.shape ,1)
        self.n_pts_filter = reduce(mul, self.shape_filter, self.dim)
        self.hotstart = False
        if self.normalize:
            self.normalize_input()
        self.spectral_data = [fftn(elem) for elem in data]
        self.vectorized_spectral_data = [self.spectral_data[iter_i].reshape((self.n_pts)) for iter_i in range(self.dim)]
        self.precomp_data =  [np.array([self.vectorized_spectral_data[i][pos] for i in range(self.dim)]) for pos in range(self.n_pts)]
    def normalize_input(self):
        # extend this for tensor stuff
        self.norm_factor = [self.n_pts*np.linalg.norm(elem)+1E-100 for elem in self.data]
        for i in range(self.dim):
            self.data[i] = self.data[i] / self.norm_factor[i]
    def initialize_filters(self):
        if self.hotstart:
            raise NotImplementedError
        else:
            self.filters = [[np.random.randn(reduce(mul,self.shape_filter ,1)).reshape(self.shape_filter) for _ in range(self.dim)] for _ in range(self.num_filters)]
        self.spectral_filters = [[fftn(self.filters[filt_id][dim],s = self.shape) for dim in range(self.dim)] for filt_id in range(self.num_filters)]
        self.vectorized_spectral_filters = [[self.spectral_filters[filt_id][dim].reshape(self.n_pts) for dim in range(self.dim)] for filt_id in range(self.num_filters)]
    def initialize_activation(self,zero=True):
        if zero:
            self.activations = [np.zeros(self.shape) for _ in range(self.num_filters)]
            self.spectral_activations = [np.zeros(self.shape) for _ in range(self.num_filters)]
            self.vectorized_spectral_activations = [self.spectral_activations[i].reshape(self.n_pts) for i in range(len(self.spectral_activations))]
        else:
            self.activations = [np.random.rand(self.n_pts).reshape(self.shape) for _ in range(self.num_filters)]
            self.activations = [activation*(activation>self.act_initialization_th) for activation in self.activations]
            self.spectral_activations = [fftn(self.activations[filt_id]) for filt_id in range(self.num_filters)]
            self.vectorized_spectral_activations = [self.spectral_activations[filt_id].reshape(self.n_pts) for filt_id in range(self.num_filters)]
    def solve_activation(self):
        # build problem
        b = self.precomp_data
        if self.robust_noise:
            A = [np.array(
                [[self.vectorized_spectral_filters[filt][dim][pos] + self.robust_noise_value*np.random.randn() for filt in range(self.num_filters)] for dim in
                 range(self.dim)]) for pos in range(self.n_pts)]
        else:
            A = [np.array(
                [[self.vectorized_spectral_filters[filt][dim][pos] for filt in range(self.num_filters)] for dim in
                 range(self.dim)]) for pos in range(self.n_pts)]
        self.problem = [(A[iter_i],b[iter_i]) for iter_i in range(self.n_pts)]
        # run the solver
        #ld('assembling problem list for activations')
        self.vectorized_spectral_activations = np.array(self.solve_problem()).T
        # reassemble results
        self.spectral_activations = [activation.reshape(self.shape) for activation in self.vectorized_spectral_activations]
        self.activations = [np.real(ifftn(spectral_activation)) for spectral_activation in self.spectral_activations]
        # this flips the activation
        if self.flip_activations:
            for filt_id in range(self.num_filters):
                if np.mean(self.activations[filt_id]>0)<self.flip_th:
                    self.activations[filt_id] = -self.activations[filt_id]
                    self.vectorized_spectral_activations[filt_id,:] = -self.vectorized_spectral_activations[filt_id,:]
                    self.spectral_activations[filt_id] = -self.spectral_activations[filt_id]
                    for dim in range(self.dim):
                        self.filters[filt_id][dim] = -self.filters[filt_id][dim]
                        self.spectral_filters[filt_id][dim] = -self.spectral_filters[filt_id][dim]
                        self.vectorized_spectral_filters[filt_id][dim] = -self.vectorized_spectral_filters[filt_id][dim]
    def soft(self,v,th):
        vabs = np.abs(v)
        mask = vabs >= th
        return np.sign(v) * (vabs-th)*mask
    def solve_filter(self):
        b = self.precomp_data
        #ld('assembling problem list for filters')
        A = [ self.aux_solve_filter(np.array([self.vectorized_spectral_activations[iter_i][pos] for iter_i in range(self.num_filters)])) for pos in range(self.n_pts) ]
        self.problem = [(A[iter_i],b[iter_i]) for iter_i in range(self.n_pts)]
        aux = self.solve_problem()
        vaux = [g.reshape((self.dim,self.num_filters)) for g in aux]
        self.vectorized_spectral_filters = [[np.array([vaux[pos][dim,filt_id]  for pos in range(self.n_pts)]) for dim in range(self.dim)] for filt_id in range(self.num_filters)]
        self.spectral_filters = [[self.vectorized_spectral_filters[filt_id][dim].reshape(self.shape) for dim in range(self.dim)] for filt_id in range(self.num_filters)]
        self.filters = [[self.project_filter_dim(ifftn(self.spectral_filters[filt_id][dim])) for dim in range(self.dim)] for filt_id in range(self.num_filters)]
        self.filt_norm = np.array([np.linalg.norm(np.array(self.filters[filt_id]))  for filt_id in range(self.num_filters)])
        self.filters = [[self.filters[filt_id][dim] / self.filt_norm[filt_id] for dim in range(self.dim)] for filt_id in
                        range(self.num_filters)]
        if self.sparsify_filter:
            self.filters =[[self.soft(self.filters[filt_id][dim],self.filter_th)for dim in range(self.dim)] for filt_id in range(self.num_filters)]
        for key in self.fixed.keys():
            self.filters[key[0]][key[1]] = self.fixed[key]
        self.spectral_filters = [[fftn(self.filters[filt_id][dim], s=self.shape) for dim in range(self.dim)] for filt_id in range(self.num_filters)]
        self.vectorized_spectral_filters = [[self.spectral_filters[filt_id][dim].reshape(self.n_pts) for dim in range(self.dim)] for filt_id in range(self.num_filters)]
    def spectralize_filters(self,filters):
        self.filters = filters
        self.spectral_filters = [[fftn(self.filters[filt_id][dim], s=self.shape) for dim in range(self.dim)] for filt_id in range(self.num_filters)]
        self.vectorized_spectral_filters = [[self.spectral_filters[filt_id][dim].reshape(self.n_pts) for dim in range(self.dim)] for filt_id in range(self.num_filters)]
    def project_filter_dim(self,elem):
        indices = tuple([slice(0, len_dim) for len_dim in self.shape_filter])
        out = elem[indices]
        if self.project_filters_to_real:
            out= np.real(out)
        return out
    def prior_activation(self):
        self.activations = [(activation - self.th) * (activation > self.th) for activation in self.activations]
        self.spectral_activations = [fftn(activation) for activation in self.activations]
        self.vectorized_spectral_activations = [activation.reshape(self.n_pts) for activation in self.spectral_activations]
    def aux_solve_filter(self,elem):
        output = np.zeros((self.dim,self.dim*self.num_filters))
        for iter_i in range(self.dim):
            output[iter_i,(iter_i)*self.num_filters:((iter_i+1)*self.num_filters)] = elem
        return output
    def solve_problem(self):
        #ld('solving problem list; len of problem list: ' + str(len(self.problem)) + '; shape of each problem: ' + str(self.problem[0][0].shape) + ' ' + str(self.problem[0][1].shape))
        #ld('solving mode PARALLEL: ' + str(self.PARALLEL))
        # extend this one to solve in parallel
        if self.PARALLEL:
            solved_problem = solve_parallel_problem(self.problem, 8)
        else:
            solved_problem = solve_problem(self.problem)
        return solved_problem
    # def rec(self):
    #     # special test case now
    #     return ifft(np.array(self.spectral_activations)[:,np.newaxis,:] * np.array(self.spectral_filters),axis=2).sum(axis=0).T
    # def diff_c(self):
    #     d = np.array(self.data).T - self.rec()
    #     self.diff.append(np.abs(d).sum(axis=0))
    def project_filters_low_rank(self,filters):
        # TODO
        # self.filters = projection(self.filters,list_of_dimensions)
        #self.spectralize_filters(self.filters)
        # update spectral filters
        # update vectorized spectral filters

def projection(single_filter,list_of_dimensions,max_dims):
    ''' projection of a single filter'''
    result = []
    possible_dims = set(range(max_dims))
    for dimensions in list_of_dimensions:
        target_dims = tuple(possible_dims.difference(dimensions))
        auxa =  np.array(single_filter.mean(axis=target_dims),ndmin=max_dims)
        for iter_dim in range(len(dimensions)):
            auxa = np.swapaxes(auxa,dimensions[iter_dim],max_dims-len(dimensions)+iter_dim)
        result.append(auxa)
    output = reduce(lambda x, y: y*x,result)
    output = output/output.sum()*single_filter.sum()
    return output

def projection_all_filters(filters,list_of_dimensions,max_dims):
    ''' does the projection for all the filters'''
    out_filters = []
    for filt_id in range(len(filters)):
        res = []
        for dim_filter in filters[filt_id]:
            res.append(projection(dim_filter),list_of_dimensions,max_dims)
        out_filters.append(res)
    return out_filters


def fconv(X,Y):
    ''' performs multidimensional convolution in spectral domain
    convolves X and Y by computing the n-dimensional fourier transforms of
    X with the size of Y '''
    # check if X and Y have the same number of dimensions
    assert len(X.shape) == len(Y.shape)
    # check if size of X is never larger than the size of Y in all dimensions
    assert all([X.shape[i] <= Y.shape[i] for i in range(len(X.shape))])
    b = fftn(X,s=Y.shape)*fftn(Y)
    return np.real(ifftn(b))

class Prediction_Engine:
    def __init__(self):
        self.shape_filter = [5, 5, 5]
        self.num_filters = 24
        self.filter_th = 1E-1
        self.th_val = 0E-2
        self.n_iters =  2
        self.verbose = True
        self.parallel = True
    def load_train_data(self,ori_data,new_data,data_inpt,fake_inpt_data):
        #ld('loading data to initialize CBM')
        self.ori_data = ori_data
        self.new_data = new_data
        self.data_inpt = data_inpt
        self.fake_inpt_data = fake_inpt_data
        #ld('loading data to initialize CBM: done')
    def fit(self, Xtrain,Ytrain):
        self.ori_data = Xtrain
        self.data_inpt = Ytrain
        #ld('initializing CBM level 0')
        self.analysis = CBM_TENSOR()
        self.analysis.PARALLEL = self.parallel
        self.analysis.filter_th = self.filter_th
        self.analysis.fit(self.ori_data, shape_filter=self.shape_filter, th=self.th_val, num_filters=self.num_filters,
                          normalize=True)
        #ld('initializing CBM level 0: done')
        #ld('initializing CBM level 0 filters and activations')
        self.analysis.flip_activations = True
        self.analysis.initialize_activation(zero=True)
        self.analysis.initialize_filters()
        #ld('initializing CBM level 0 filters and activations: done')
        # initialize synthesis filters
        #ld('initializing CBM level 1')
        self.synthesis = CBM_TENSOR()
        self.synthesis.PARALLEL = self.parallel
        self.synthesis.filter_th = self.filter_th
        self.synthesis.fit(self.data_inpt, shape_filter=self.shape_filter, th=self.th_val, num_filters=self.num_filters,
                           normalize=True)
        self.synthesis.flip_activations = False
        # SOLVE FILTER AND ACTIVATION FOR XTRAIN
        for iter_i in range(self.n_iters):
            #ld('solving activation for level 0')
            self.analysis.solve_activation()
            #ld('applying prior to activation for level 0')
            self.analysis.prior_activation()
            #ld('solving analysis filters')
            self.analysis.solve_filter()
        # IMPOSE ACTIVATIONS FOR YTRAIN
        #ld('copying activations from level 0 to level 1')
        self.synthesis.activations = self.analysis.activations
        self.synthesis.spectral_activations = self.analysis.spectral_activations
        self.synthesis.vectorized_spectral_activations = self.analysis.vectorized_spectral_activations
        # SOLVE FILTER FOR YTRAIN
        self.synthesis.solve_filter()
    def predict(self, Xtest):
        ''' Compute activation of Xtest (from analysis filter)'''
        compute_analysis = CBM_TENSOR()
        compute_analysis.PARALLEL = self.parallel
        compute_analysis.fit(Xtest, shape_filter=self.analysis.shape_filter,
                                  num_filters=self.analysis.num_filters, normalize=True)
        compute_analysis.spectralize_filters(self.analysis.filters)
        compute_analysis.solve_activation()
        # Synthetize output from activation of Xtest
        return np.array([[fconv(self.synthesis.filters[filt_id][dim], compute_analysis.activations[filt_id]) for
                              dim in range(self.synthesis.dim)] for filt_id in range(self.num_filters)]).sum(axis=0)


# class CBM():
#     def __init__(self):
#         self
        
# test = np.ones((5,5,5,5)) * np.eye(5,5)[:,:,np.newaxis,np.newaxis]
# dims_to_project = [[0,1],[2],[3]]
# res = projection(test,dims_to_project,len(test.shape))
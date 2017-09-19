from __future__ import print_function
import argparse, gzip, cPickle, sys, time, itertools

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
import autograd.scipy.stats.dirichlet as dirichlet
from autograd.scipy.misc import logsumexp
from autograd.util import flatten_func, flatten
from autograd import grad, primitive
from autograd.numpy.numpy_grads import unbroadcast

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn import mixture

from  autograd.scipy.special import gammaln, digamma, gamma
from scipy import linalg
from scipy import stats, integrate

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import pandas as pd
import six


class LinearRegression(object):
    def __init__(self, data, scale):
        self.data = data             
        self.scale = scale
        self.N = data['x'].shape[0]
        self.D = data['x'].shape[1]        
    
    def p_log_prob(self, idx, z):
        x, y = self.data['x'][idx], self.data['y'][idx] 
        w, tau = z['w'], z['tau']       
        log_prior = 0.
        log_prior += np.sum(gamma_logpdf(tau, 1e-3, 1e-3) + np.log(jacobian_softplus(z['tau'])))        
        #log_prior += np.sum(gamma_logpdf(alpha, 1e-3, 1e-3) + np.log(jacobian_softplus(z['alpha'])))        
        log_prior += np.sum(norm.logpdf(w, 0, 1.)) 
        log_lik = np.sum(norm.logpdf(y, np.matmul(x, w), 1./np.sqrt(tau)))
        return self.scale * log_lik + log_prior        
         
    def q_log_prob(self, means, log_sigmas, z):
        q_w = np.sum(norm.logpdf(z, means, np.exp(log_sigmas)))        
        return q_w
      
    def q_log_prob_sep(self, means, log_sigmas, z):
        q_w = norm.logpdf(z, means, np.exp(log_sigmas))        
        return q_w

    def phi_log_prob_sep(self, eps):
        phi_w = norm.logpdf(eps, 0., 1.)        
        return phi_w

    def sample_q(self, means, log_sigmas, d):        
        eps = npr.randn(d)        
        q_s = np.exp(log_sigmas) * eps + means
        return (q_s, eps)
        
    def grad_params(self, dp_log_prob, eps, log_sigmas):                
        grad_means = dp_log_prob
        grad_log_sigmas = dp_log_prob*eps*np.exp(log_sigmas) + 1                
        return np.concatenate([grad_means, grad_log_sigmas])
        
    def calc_eps(self, means, log_sigma, z):        
        eps  = (z - means)/np.exp(log_sigma)
        return eps            


#Parameters
N = 50000
K = 500
D = 1
x = 5 * npr.randn(N*K).reshape([N,K])
alpha = np.ones(K)
w = npr.normal(0, 1./np.sqrt(alpha))
y = np.matmul(x, w) + npr.randn(N)
data = {}
data['x'] = x
data['y'] = y
data['x'].shape


batch_size = 5000
seed = 1234
learning_rate = 0.1
samples = 1
epochs = 50


model = LinearRegression(data, N/batch_size) 
sns.set_style(style='white')

npr.seed(seed)    
params = {}
params['means'] = {'w': npr.randn(K), 'tau': 1e2 * npr.randn(1)} #, 'alpha':  npr.randn(K), 'tau': npr.randn(1)}
params['log_sigmas'] = {'w': npr.randn(K), 'tau': 1e-2 * npr.randn(1)} #, 'alpha': .1 * npr.randn(K), 'tau': npr.randn(1)}
inference = Inference(model, params)  
inference.run(13, batch_size, samples, learning_rate, 'SGD')
plt.plot(np.cumsum(inference.time), -inference.F, color = colors[1], label = 'SGD')


npr.seed(seed)    
params = {}
params['means'] = {'w': npr.randn(K), 'tau': 1e2 * npr.randn(1)} #, 'alpha':  npr.randn(K), 'tau': npr.randn(1)}
params['log_sigmas'] = {'w': npr.randn(K), 'tau': 1e-2 * npr.randn(1)} #, 'alpha': .1 * npr.randn(K), 'tau': npr.randn(1)}
inference = Inference(model, params)  
inference.run(80, batch_size, samples, learning_rate, 'iSGD')
plt.plot(np.cumsum(inference.time), -inference.F, color = colors[0], label = 'I-SGD')


fig = plt.gcf()
fig.set_size_inches(4,3)
plt.ylabel('ELBO', fontsize = 15)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel('Time[s]', fontsize = 15)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.legend(loc = 4, fontsize = 15)
#plt.savefig('/home/sakaya/MUPI/papers/uai17importance/linear.png', dpi=300, bbox_inches= 'tight')


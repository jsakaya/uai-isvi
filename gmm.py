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


color_names = ["maroon",               
               "gold",
               "royal blue"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
color_iter = itertools.cycle(colors)

def plot_results(ax, X, Y, means, covariances, index, title):    
    for i, (mean, covar, color) in enumerate(zip(
             means, covariances, color_iter)):
        v, w = linalg.eigh(np.diag(np.full([2], covar)))
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])      

        if not np.any(Y == i):
            continue
        ax.scatter(X[Y == i, 0], X[Y == i, 1], 2., color=color)

        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        #ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
            
class GMM(object):
    def __init__(self, data, clusters, scale):
        self.data = data     
        self.clusters = clusters
        self.scale = scale
        self.N = data.shape[0]
        self.D = data.shape[1]        
    
    def p_log_prob(self, idx, z):
        x = self.data[idx]        
        mu, tau, pi = z['mu'], softplus(z['tau']), stick_breaking(z['pi'])
        matrix = []  
        log_prior = 0.
        log_prior += np.sum(gamma_logpdf(tau, 1e-5, 1e-5) + np.log(jacobian_softplus(z['tau'])))        
        log_prior += np.sum(norm.logpdf(mu, 0, 1.))
        log_prior += dirichlet.logpdf(pi, 1e3 * np.ones(self.clusters)) + np.log(jacobian_stick_breaking(z['pi']))
        for k in range(self.clusters):
            matrix.append(np.log(pi[k]) + np.sum(norm.logpdf(x, mu[(k * self.D):((k + 1) * self.D)],
                                np.full([self.D], 1./np.sqrt(tau[k]))), 1))
        matrix  = np.vstack(matrix)
        vector = logsumexp(matrix, axis=0)
        log_lik = np.sum(vector)        
        return self.scale * log_lik + log_prior        
    
    def predict(self, z):
        x = self.data
        mu, tau, pi = z['mu'], softplus(z['tau']), stick_breaking(z['pi'])        
        matrix = []                
        for k in range(self.clusters):
            matrix.append(np.log(pi[k]) + np.sum(norm.logpdf(x, mu[(k * self.D):((k + 1) * self.D)],
                                 np.full([self.D], 1./np.sqrt(tau[k]))), 1))
        matrix  = np.vstack(matrix)                
        return np.argmax(matrix, 0)    
    
 
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


N = n_samples = 10000
epochs = 10
samples = 1
learning_rate = 0.1
K = clusters = 25
seed = 222
batch_size = 2000
D = 2


npr.seed(seed)
c =  5 * npr.randn(K * D).reshape([K,D])
m = np.tile(c, (N/K,1))
X =  npr.randn(N * D).reshape([N,D]) + m


batch_size = 500
epochs = 50
samples = 1
learning_rate = .1
seed = 111
data = X
    
N = data.shape[0]
D = data.shape[1]
model = GMM(data, K, N/batch_size)

        
#f, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(30,30))


npr.seed(seed)    
params = {}
params['means'] = {'mu': np.ones(K*D), 'tau': 1e-2 * npr.randn(K), 'pi':  npr.randn(K-1)}
params['log_sigmas'] = {'mu': 1e-2 * npr.randn(K*D), 'tau': 1e-2 * npr.randn(K), 'pi': 1e-2 * npr.randn(K-1)}
inference = Inference(model, params)
inference.run(40, batch_size, samples, learning_rate, 'SGD')
plt.plot(np.cumsum(inference.time), -inference.F, color = colors[1],  label = "SGD")
# p = model.predict(inference.params['means'])
# means_ = inference.params['means']['mu'].reshape([clusters, D])
# covariances_ = 1/np.sqrt(np.exp(inference.params['means']['tau']))
# pi_ = stick_breaking(inference.params['means']['pi'])
# plot_results(ax2, X, p, means_, covariances_, 0, 'Bayesian GMM')



npr.seed(seed)    
params = {}
params['means'] = {'mu': np.ones(K*D), 'tau':  1e-2 * npr.randn(K), 'pi':  npr.randn(K-1)}
params['log_sigmas'] = {'mu': 1e-2 * npr.randn(K*D), 'tau': 1e-2 * npr.randn(K), 'pi': 1e-2 * npr.randn(K-1)}
inference = Inference(model, params)
inference.run(310, batch_size, samples, learning_rate, 'iSGD')
plt.plot(np.cumsum(inference.time), -inference.F, color = colors[0], label = "I-SGD")
# p = model.predict(inference.params['means'])
# means_ = inference.params['means']['mu'].reshape([clusters, D])
# covariances_ = 1/np.sqrt(np.exp(inference.params['means']['tau']))
# pi_ = stick_breaking(inference.params['means']['pi'])
# plot_results(ax3, X, p, means_, covariances_, 0, 'Bayesian GMM')


fig = plt.gcf()
fig.set_size_inches(4,3)
plt.ylabel('ELBO', fontsize = 15)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel('Time[s]', fontsize = 15)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.legend(loc = 4, fontsize = 15)
plt.savefig('/home/sakaya/MUPI/papers/uai17importance/gmm.png', dpi=300, bbox_inches= 'tight')

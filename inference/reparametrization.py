import autograd.numpy as np
import autograd.numpy.random as npr

""" Implementation of the Adam optimizer. Borrowed from autograd's version."""

class Adam(object):
    def __init__(self, dparam, b1=0.9, b2=0.999, eps=10**-8,
                         decay_rate=0.9, decay_steps=100):                    
        self.b1 = b1;
        self.b2 = b2;
        self.eps = eps        
        self.m = np.zeros(dparam)
        self.v = np.zeros(dparam)
        self.i = 0
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
    
    def update(self, gradients, params, learning_rate = 0.1):        
        self.i = self.i+1
        step_size = learning_rate * self.decay_rate**(self.i/self.decay_steps)        
        self.m = (1 - self.b1) * gradients + self.b1 * self.m
        self.v = (1 - self.b2) * (gradients**2) + self.b2 * self.v
        mhat = self.m / (1 - self.b1**(self.i))
        vhat = self.v / (1 - self.b2**(self.i))                        
        params = params + step_size*mhat/(np.sqrt(vhat) + self.eps)        
        return np.split(params,2)

"""	Inference class. 
	SGD performs standard stochastic variational inference with the Monte Carlo estimate of the gradient
	iSGD performs importance sampled stochastic variational inference with the Monte Carlo estimate of the gradient """

class Inference(object):      
    def __init__(self, model, params):
        self.model = model
        self.params = params

    def run(self, epochs, batch_size, samples, learning_rate, algorithm = 'SGD', optimizer = 'adam'):
        epochs = epochs
        batches = self.model.N/batch_size
        batch_size = batch_size
        samples = samples
        learning_rate = learning_rate        
        
        means, unflatten = flatten(self.params['means'])
        log_sigmas, unflatten = flatten(self.params['log_sigmas'])        
        D =len(means)

        self.F = np.zeros(epochs * batches)
        self.time = np.zeros(epochs * batches)
        adam = Adam(2*D)
        f = 0
        
        grad_p_log_prob = grad(model.p_log_prob, argnum = 1)
        grad_q_log_prob = grad(model.q_log_prob, argnum = 1)
        
        if algorithm == 'SGD':
            for e in range(epochs):
                for b in range(batches): 
                    start = time.clock()
                    losses = 0.
                    d_elbo = 0.
                    idx = np.random.choice(np.arange(self.model.N), batch_size, replace=False)                    
                    d_elbo = 0.

                    for s in range(samples):
                        eps = npr.randn(D)        
                        z = np.exp(log_sigmas) * eps + means                                            
                        p_log_prob = model.p_log_prob(idx, unflatten(z))                        
                        dp_log_prob, _ = flatten(grad_p_log_prob(idx, unflatten(z)))
                        g =  model.grad_params(dp_log_prob, eps, log_sigmas)                        
                        d_elbo += g
                        q_log_prob = model.q_log_prob(means, log_sigmas, z)                                         
                        losses +=  (p_log_prob - q_log_prob)                    
                    loss = losses/samples
                    d_elbo /= samples   
                    means_old, log_sigmas_old = means, log_sigmas
                    means, log_sigmas = adam.update(d_elbo, np.concatenate([means, log_sigmas]), learning_rate)
                    if np.sum(np.isnan(means)) > 0 or np.sum(np.isnan(log_sigmas)) > 0:
                        means, log_sigmas = means_old, log_sigmas_old
                        learning_rate = learning_rate * .1                        
                    self.F[f] =  -loss                

                    stop = time.clock()
                    self.time[f] = stop - start
                    f+=1
                if e % 1 == 0:
                    pstate = 'Epoch = ' + "{0:0=3d}".format(e) + ': Loss = {0:.3f}'.format(self.F[f-1])
                    print (pstate, end = '\r')
                    sys.stdout.flush()   
                                                                                                              
        if algorithm == 'iSGD':
            n = 1.  
            z_old = [0.] * samples
            dp_log_prob_old = [0.] * samples
            phi_log_prob_old = [0.] * samples            

            for e in range(epochs):
                for b in range(batches): 
                    start = time.clock()
                    losses = 0.
                    d_elbo = 0.
                    idx = np.random.choice(np.arange(self.model.N), batch_size, replace=False)                                        
                    """ Choice of when to use the importance sampled estimate of the gradient dependent on n = npr.uniform()
						Here, the inference uses the importance sampled estimates 90% of the time. """
                    if n > .9:
                        for s in range(samples):
                            eps = npr.randn(D)        
                            z = np.exp(log_sigmas) * eps + means
                            p_log_prob = model.p_log_prob(idx, unflatten(z))
                            q_log_prob = model.q_log_prob_sep(means, log_sigmas, z)                                         
                            dp_log_prob, _ = flatten(grad_p_log_prob(idx, unflatten(z)))
                            g =  model.grad_params(dp_log_prob, eps, log_sigmas)                        
                            d_elbo += g
                            losses +=  (p_log_prob - np.sum(q_log_prob))
                                        
                            z_old[s] = z
                            dp_log_prob_old[s] = dp_log_prob
                            phi_log_prob_old[s] = model.phi_log_prob_sep(eps)                                                                     
                        loss = losses/samples
                        d_elbo /= samples                                            
                    else:                         
                        for s in range(samples):                            
                            eps = (z_old[s] - means)/np.exp(log_sigmas)                            
                            phi_log_prob = model.phi_log_prob_sep(eps)                                         
                            w = np.exp(phi_log_prob - phi_log_prob_old[s])                            
                            g =  model.grad_params(w * dp_log_prob_old[s], eps, log_sigmas)                        
                            d_elbo += g                            
                        d_elbo /= samples                                             
                    n = npr.uniform()
                    means_old, log_sigmas_old = means, log_sigmas
                    means, log_sigmas = adam.update(d_elbo, np.concatenate([means, log_sigmas]), learning_rate)
                    if np.sum(np.isnan(means)) > 0 or np.sum(np.isnan(log_sigmas)) > 0:
                        means, log_sigmas = means_old, log_sigmas_old
                        learning_rate = learning_rate * .9   
                        n = 1.
                    self.F[f] =  -loss                
                    stop = time.clock()
                    self.time[f] = stop - start
                    f+=1                                        
                if e % 1 == 0:
                    pstate = 'Epoch = ' + "{0:0=3d}".format(e) + ': Loss = {0:.3f}'.format(self.F[f-1])
                    print (pstate, end = '\r')
                    sys.stdout.flush()  
        self.params = {'means': unflatten(means), 'log_sigmas': unflatten(log_sigmas)}

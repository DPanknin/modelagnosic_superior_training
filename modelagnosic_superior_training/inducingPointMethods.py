"""
This module implements several inducing point selection methods.
"""

import numpy as np
import scipy as sp
import torch
from .svgd import SVGD
from .myKMeans import myKMeans
from .AL_base import myScientificFormat
import gpytorch
from gpytorch.constraints import Positive

class SVGD_inducing_points():
    
    def __init__(self, X, inputSpaceBounds, refSizeNumIP, refSig, stepsize, terminalStepsize, repulsionPropToDensity, version, n_iter, pTrainOfX = None, pTargetOfX = None, pTrain = None, pTarget = None, intrinsicDim = None):
        self.X = X
        if pTrainOfX is None:
            pTrainOfX = pTrain(X)
        if pTargetOfX is None:
            pTargetOfX = pTarget(X)
        self.pTarget = pTarget
        resamplingWeights = pTargetOfX / pTrainOfX
        resamplingWeights /= np.sum(resamplingWeights)
        self.resamplingWeights = resamplingWeights
        if intrinsicDim is None:
            self.intrinsicDim = self.X.shape[1]
        else:
            self.intrinsicDim = intrinsicDim
        self.inputSpaceBounds = inputSpaceBounds
        self.refSizeNumIP = refSizeNumIP
        self.refSig = refSig
        
        self.stepsize = stepsize
        self.terminalStepsize = terminalStepsize
        self.repulsionPropToDensity = repulsionPropToDensity
        self.version = version
        self.n_iter = n_iter
        
    def __call__(self, m):
        sigSVGD = self.refSig*(m/self.refSizeNumIP)**(-1/self.intrinsicDim)
        print('apply SVGD sigma', myScientificFormat(sigSVGD))
        
        inxes = np.random.choice(np.arange(len(self.X)), m, replace = False, p = self.resamplingWeights)
        inducing_points = SVGD().update(self.X[inxes], fixedIndices = None, prob = self.pTarget, lnprob = None, inputSpaceBounds = self.inputSpaceBounds, bandwidth = sigSVGD, stepsize = self.stepsize, terminalStepsize = self.terminalStepsize, repulsionPropToDensity = self.repulsionPropToDensity, version=self.version, n_iter = self.n_iter)
        return inducing_points

class kMeans_inducing_points():
    
    def __init__(self, X, init_centers, distributionalClustering, max_iter, pTrainOfX = None, pTrain = None, intrinsicDim = None):
        self.X = X
        self.init_centers = init_centers
        self.distributionalClustering = distributionalClustering
        self.max_iter = max_iter
        if pTrainOfX is None:
            pTrainOfX = pTrain(X)
        self.pTrainOfX = pTrainOfX
        if intrinsicDim is None:
            self.intrinsicDim = self.X.shape[1]
        else:
            self.intrinsicDim = intrinsicDim
    
    def __call__(self, m):
        inxes,_ = myKMeans(self.X, n_centers = m, densityX = self.pTrainOfX, init_centers = self.init_centers, distributionalClustering = self.distributionalClustering, max_iter = self.max_iter, intrinsicDim = self.intrinsicDim)
        inducing_points = torch.tensor(self.X[inxes])
        return inducing_points

class random_inducing_points():

    def __init__(self, X, pTrainOfX = None, pTargetOfX = None, pTrain = None, pTarget = None):
        self.X = X
        if pTrainOfX is None:
            pTrainOfX = pTrain(X)
        if pTargetOfX is None:
            pTargetOfX = pTarget(X)
        resamplingWeights = pTargetOfX / pTrainOfX
        resamplingWeights /= np.sum(resamplingWeights)
        self.resamplingWeights = resamplingWeights
        
    def __call__(self, m):
        inxes = np.random.choice(np.arange(len(self.X)), m, replace = False, p = self.resamplingWeights)
        inducing_points = torch.tensor(self.X[inxes])
        return inducing_points

class GFF_inducing_points():
    
    def __init__(self, X, y, lam, scales, noiseLevel, informationThreshold = 1e-2, doublePrecision = False):
        if np.isscalar(scales):
            ard_num_dims = None
        else:
            ard_num_dims = len(scales)

        kFun = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims, lengthscale_constraint=Positive(transform=torch.exp, inv_transform=torch.log)), outputscale_constraint=Positive(transform=torch.exp, inv_transform=torch.log))
        kFun._set_outputscale(lam)
        kFun.base_kernel._set_lengthscale(scales)
        if doublePrecision:
            kFun = kFun.double()
        self.kFun = kFun
        self.X = X
        self.y = y
        self.noiseLevel = noiseLevel
        self.informationThreshold = informationThreshold
        
    def __call__(self, m, lam = None, scales = None, noiseLevel = None):
        
        if not lam is None:
            self.kFun._set_outputscale(lam)
        if not scales is None:
            self.kFun.base_kernel._set_lengthscale(scales)
        if not noiseLevel is None:
            self.noiseLevel = noiseLevel
        
        inxes,_ = seegerFastForwardIPselection(torch.tensor(self.X).type(self.kFun.dtype), self.y, np.sqrt(self.noiseLevel), m, K = None, kFun = self.kFun, sofarIPinx = [], informationThreshold = self.informationThreshold)
        inducing_points = torch.tensor(self.X[inxes])
        print('found ', len(inxes), ' of up to ' + str(m) + ' inducing points at threshold ', myScientificFormat(self.informationThreshold), sep='')
        return inducing_points

def seegerFastForwardIPselection(x, y, noise_stdDev, numIP, K = None, kFun = None, sofarIPinx = [], kernelResemblanceThreshold = 0., informationThreshold = 0.):
    n = len(y)
    if K is None:
        with torch.no_grad():
            diagK = kFun(x, diag=True).numpy()
    else:
        diagK = np.diag(K)
    if numIP > n:
        print('Cannot draw ', numIP,' IPs from ', n,' data points according to GFF. Reducing number of IPs to ', n, sep='')
        numIP = n
    y = y.squeeze()
    L = np.zeros([numIP,numIP])
    V = np.empty([numIP,n])
    p = np.zeros(n)
    q = np.zeros(n)
    LM = np.zeros([numIP,numIP])
    beta = np.empty(numIP)
    mu = np.zeros(n)
    if len(sofarIPinx) == 0:
        # initialize an IP at random
        i = np.random.choice(n)
        sofarIPinx.append(i)
    #TODO: Initialize all variables reasonably
    i = sofarIPinx[0]
    li = np.sqrt(diagK[i])
    L[0,0] = li
    if K is None: 
        with torch.no_grad():
            Ki = kFun(x, x[[i]]).numpy().ravel()
    else:
        Ki = K[:,i]
    v = 1/li * Ki
    V[0,:] = v
    p += v**2
    #M = noise_stdDev**2 + np.sum(v**2)
    lMi = np.sqrt(noise_stdDev**2 + np.sum(v**2))
    LM[0,0] = lMi # np.sqrt(M)
    
    w = 1/lMi * v
    q += w**2
    beta[0] = 1/lMi * np.sum(v * y)
    mu += beta[0] * w
    
    for d in range(1,numIP):
        l = np.sqrt(np.maximum(diagK - p, 0.))
        l[sofarIPinx] = 0.
        # if everything is represented, we cannot go on sampling by any means. Else we set the next diag entry of L to zero.
        if np.max(l) <= kernelResemblanceThreshold:
            return sofarIPinx, mu
        if d >= len(sofarIPinx):
            # propose a new inx
            xi = 1 / ((noise_stdDev/l)**2 + 1 - q)
            kappa = xi*(1 + 2*(noise_stdDev/l)**2)
            informationGain = -np.log(noise_stdDev/l) - 0.5*(np.log(xi) + xi*(1-kappa)/noise_stdDev**2*(y-mu)**2 - kappa + 2)
            # some values of l can become zero, even though the index is not drawn yet. I think that these should be treated as well-represented, and therefore should not be added. Thus, set their informationGain value to a negative value, but larger than -np.Inf to distinguish them from already drawn points
            informationGain[l==0.] = -np.Inf
            if any(np.isnan(informationGain)):
                print('nan in informationGain')
            i = np.argmax(informationGain)
            if informationGain[i] <= informationThreshold:
                return sofarIPinx, mu
        else:
            # special case, where we use further pre-determined inxes
            i = sofarIPinx[d]
        
        # update the variables
        li = l[i]
        vi = V[:d,i]
        L[d,:d] = vi
        L[d,d] = li
        if K is None: 
            with torch.no_grad():
                Ki = kFun(x, x[[i]]).numpy().ravel()
        else:
            Ki = K[:,i]
        v = 1/li * (Ki - V[:d,:].T.dot(vi))
        V[d,:] = v
        p += v**2
        #lM = LM[:d,:d].solve(V[:d,:] @ v)
        lM = sp.linalg.solve_triangular(LM[:d,:d], V[:d,:].dot(v), lower=True, check_finite = False)
        lMi = np.sqrt(np.maximum(noise_stdDev**2 + np.sum(v**2) - np.sum(lM**2), 0.))
        LM[d,:d] = lM
        LM[d,d] = lMi
        
        #w = 1/lMi * (v - V[:d,:].T @ LM[:d,:d].T.solve(lM))
        w = 1/lMi * (v - V[:d,:].T.dot(sp.linalg.solve_triangular(LM[:d,:d], lM, trans='T', lower=True, check_finite = False)))
        q += w**2
        beta[d] = 1/lMi * (np.sum(v * y) - np.sum(beta[:d] * lM))
        mu += beta[d] * w
    
        sofarIPinx.append(i)
    
    return sofarIPinx, mu

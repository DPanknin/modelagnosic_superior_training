import numpy as np
from scipy.spatial.distance import pdist, squareform

class SVGD():

    def __init__(self):
        pass
        
    def svgd_kernel(self, theta, h = -1, importanceWeights = None, fixedIndices = None, lnpgrad = None, probs = None, version='v2'):      
        if h < 0: # if h < 0, using median trick
            sq_dist = pdist(theta)
            pairwise_dists = squareform(sq_dist)**2
            h = np.median(pairwise_dists)  
            h = np.sqrt(0.5 * h / np.log(theta.shape[0]+1))
            
        if not importanceWeights is None:
            importanceWeights = importanceWeights.reshape(-1,1)
            
        if fixedIndices is None:
            iterateOver = range(len(theta))
        else:
            iterateOver = np.where(~fixedIndices)[0]
        if not probs is None:
            inputDim = theta.shape[1]
            probs = probs.reshape(-1,1) ** (1/inputDim)
        grad = []
        for i in iterateOver:
            if probs is None:
                KxiX = np.exp(-0.5*np.linalg.norm(theta - theta[i], axis = 1)**2/h**2).reshape(-1,1)
                if not importanceWeights is None:
                    KxiX = KxiX * importanceWeights
                g = (KxiX.sum()*theta[i] - np.sum(KxiX*theta,axis=0) ) / h**2
            else:
                if version == 'v1':
                    KxiX = np.exp(-0.5*np.linalg.norm(theta - theta[i], axis = 1).reshape(-1,1)**2/(h/probs)**2)
                    if not importanceWeights is None:
                        KxiX = KxiX * importanceWeights
                    g = np.sum(KxiX/(h/probs)**2)*theta[i] - np.sum(KxiX/(h/probs)**2*theta,axis=0)
                else:
                    KxiX = np.exp(-0.5*np.linalg.norm(theta - theta[i], axis = 1)**2/(h/probs[i])**2).reshape(-1,1)
                    if not importanceWeights is None:
                        KxiX = KxiX * importanceWeights
                    g = (KxiX.sum()*theta[i] - np.sum(KxiX*theta,axis=0) ) / (h/probs[i])**2
                
            if not lnpgrad is None:
                g += np.sum(KxiX*lnpgrad,axis=0)
            grad.append(g)
        grad = np.asarray(grad)
        grad /= len(theta)
        return grad
 
    def update(self, theta, fixedIndices = None, prob = None, lnprob = None, inputSpaceBounds = None, bandwidth = -1, stepsize = 1e-2, terminalStepsize = 1e-4, repulsionPropToDensity = True, version='v2', alpha = 0.9, n_iter = 200, fudge_factor = 1e-6, empiricalVersion = True):
        # Check input
        if theta is None or (lnprob is None and prob is None):
            raise ValueError('theta or density info cannot be None!')
        n = len(theta)
        probs = None
        lnpgrad = None
        importanceWeights = None
        # adagrad with momentum
        gradOld = 0
        for iter in range(n_iter):
            eps = 10**(np.log10(stepsize) - iter/(n_iter-1)*np.log10(stepsize / terminalStepsize))
            if repulsionPropToDensity and not prob is None or lnprob is None:
                # update probs
                if probs is None or fixedIndices is None:
                    probs = prob(theta)
                else:
                    probs[~fixedIndices] = prob(theta[~fixedIndices])

            if not lnprob is None:
                # update lnpgrad
                if lnpgrad is None or fixedIndices is None:
                    lnpgrad = lnprob(theta)
                else:
                    lnpgrad[~fixedIndices] = lnprob(theta[~fixedIndices])
            else:
                importanceWeights = 1 / probs
                importanceWeights /= importanceWeights.sum() / n # sum of IW should be n

            # calculating the gradient
            gradNew = self.svgd_kernel(theta, h = bandwidth, importanceWeights = importanceWeights, fixedIndices = fixedIndices, lnpgrad = lnpgrad, probs = (probs if repulsionPropToDensity else None), version=version)
            if lnprob is None and not empiricalVersion:
                if not fixedIndices is None:
                    importanceWeights = importanceWeights[~fixedIndices]
                    #importanceWeights /= importanceWeights.sum() / n # or len(importanceWeights)
                gradNew *= importanceWeights.reshape(-1,1)
            # adagrad 
            if iter == 0:
                gradOld = gradOld + gradNew ** 2
            else:
                gradOld = alpha * gradOld + (1 - alpha) * (gradNew ** 2)
            adj_grad = np.divide(gradNew, fudge_factor+np.sqrt(gradOld))
            if not fixedIndices is None:
                theta[~fixedIndices] += eps * adj_grad
            else:
                theta += eps * adj_grad
            if not inputSpaceBounds is None:
                #theta = np.minimum(np.maximum(theta, inputSpaceBounds[0]), inputSpaceBounds[1])
                reflectTheta = 2*(inputSpaceBounds[0] - theta)
                theta[theta < inputSpaceBounds[0]] += reflectTheta[theta < inputSpaceBounds[0]]
                reflectTheta = 2*(inputSpaceBounds[1] - theta)
                theta[theta > inputSpaceBounds[1]] += reflectTheta[theta > inputSpaceBounds[1]]
        return theta

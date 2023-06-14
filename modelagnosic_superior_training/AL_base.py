"""
This module contains some utility functions for the AL framework. That is, standard sampling methods and density functions, density and noise estimators and result file handling. 
"""

import numpy as np
import os
import torch
from sklearn.neighbors import NearestNeighbors
from .myKMeans import myKMeans

def randInputUnif(size, inputSpaceBounds, xTrain = []):
    inputDim = inputSpaceBounds.shape[1]
    U = np.random.uniform(0,1,size = size*inputDim).reshape(-1,inputDim)
    U = inputSpaceBounds[0] + U * (inputSpaceBounds[1] - inputSpaceBounds[0])
    return U

def inputDensityUnif(x, inputSpaceBounds):
    vol = (inputSpaceBounds[1] - inputSpaceBounds[0]).prod()
    return np.ones(len(x))/vol
    
def equidistantInput(size, inputSpaceBounds, xTrain = []):
    inputDim = inputSpaceBounds.shape[1]
    
    dimWiseSize = int(size**(1/inputDim))
    inputs = np.linspace(0+1/dimWiseSize/2,1-1/dimWiseSize/2,dimWiseSize)

    ll = [inputs] * inputDim
    inputs = np.meshgrid(*ll, indexing='ij')
    inputs = np.asarray(inputs).reshape(inputDim,-1).T
    inputs = inputSpaceBounds[0] + inputs * (inputSpaceBounds[1] - inputSpaceBounds[0])
    
    if len(xTrain) == size:
        # treat special case of first update, where we need to shift the proposed new inputs
        if inputs[[0]] in xTrain:
            inputs -= 1/dimWiseSize/2
    
    return(inputs)
    
def myRejectionSampling(p, inputSpaceBounds, pMax, size):
    samples = []
    while len(samples) < size:
        y = np.random.uniform(inputSpaceBounds[0], inputSpaceBounds[1], size = [1,inputSpaceBounds.shape[1]])
        u = np.random.uniform(0, pMax, size = 1)
        if u < p(y):
            samples.append(y)
    return(np.row_stack(samples))
    
def myPoolSampling(size, pTargetOfPool, pPoolofPool, invalidPoolInx = None, replace=False):
    poolResamplingWeights = pTargetOfPool/pPoolofPool
    if not invalidPoolInx is None:
        poolResamplingWeights[invalidPoolInx] = 0.
    poolResamplingWeights /= np.sum(poolResamplingWeights)
    inx = np.random.choice(len(poolResamplingWeights), size, replace=replace, p = poolResamplingWeights)
    return(inx)
    
def kMeansSamplingFromPool(size, candidatePool, pTargetOfPool, pPoolofPool, invalidPoolInx = None, perms = None, featureFunction = None, n_init = 10, init_centers = 'k-means++', distributionalClustering = True, max_iter = 300, intrinsicDim = None, fixedCenters = None, preSampleSizeFactor = 10):
    # from the remaining pool, draw training data
    inx = myPoolSampling(size*preSampleSizeFactor, pTargetOfPool, pPoolofPool, invalidPoolInx = invalidPoolInx)

    X = candidatePool[inx]
    if not featureFunction is None:
        XFeat = featureFunction(X)
    else:
        XFeat = X
    inxes,_ = myKMeans(XFeat, n_centers = size, densityX = pTargetOfPool[inx], perms = perms, init_centers = init_centers, distributionalClustering = distributionalClustering, max_iter = max_iter, intrinsicDim = intrinsicDim, fixedCenters = fixedCenters)
    return(inx[inxes])
    
def kMeansSamplingFromDensity(size, p, inputSpaceBounds, pMax, perms = None, featureFunction = None, init_centers = 'k-means++', distributionalClustering = True, max_iter = 300, intrinsicDim = None, fixedCenters = None, preSampleSizeFactor = 10):
    # draw a pseudo pool
    X = myRejectionSampling(p, inputSpaceBounds, pMax, size*preSampleSizeFactor)
    if not featureFunction is None:
        XFeat = featureFunction(X)
    else:
        XFeat = X
    inxes,_ = myKMeans(XFeat, n_centers = size, densityX = p(X), perms = perms, init_centers = init_centers, distributionalClustering = distributionalClustering, max_iter = max_iter, intrinsicDim = intrinsicDim, fixedCenters = fixedCenters)
    return(X[inxes])

# this one applies in 1-d, so far
def constructDyadicLayer(n):
    layer = int(np.log2(n))
    if layer == 0:
        return(np.zeros([1,1]))
    x = np.arange(1,2**layer,2)/2**layer
    return(x.reshape(-1,1))
    
def initDyadicGrid(n):
    terminalLayer = int(np.log2(n))
    xOut = []
    for layer in range(terminalLayer+1):
        xOut.append(constructDyadicLayer(int(2**layer)))
    return(np.row_stack(xOut))
    
def myDensityFitSampling(p, refX, inputSpaceBounds, size):
    currentX = np.sort(refX.reshape(-1))
    currentX = np.concatenate((inputSpaceBounds[0], currentX, inputSpaceBounds[1], inputSpaceBounds[1]))
    candidates = 0.5*(currentX[1:] + currentX[:-1])
    interDistances = currentX[1:] - currentX[:-1]
    pCandidate = p(candidates.reshape(-1,1), inputSpaceBounds).reshape(-1)
    scores = pCandidate * interDistances
    candidates = candidates.tolist()
    scores = scores.tolist()
    interDistances = interDistances.tolist()
    #pCandidate = pCandidate.tolist()
    currentX = currentX.tolist()
    samples = []
    while len(samples) < size:
        inx = np.argmax(scores)
        newX = candidates[inx]
        currentX.insert(inx+1, newX)
        leftSuccessor = 0.5*(currentX[inx+1] + currentX[inx])
        rightSuccessor = 0.5*(currentX[inx+1] + currentX[inx+2])

        samples.append(newX)
        # update all arrays
        # remove candidates[inx] from candidates
        candidates.pop(inx)
        # add left and right new candidates instead
        candidates.insert(inx, leftSuccessor)
        candidates.insert(inx+1, rightSuccessor)
        newDist = interDistances[inx]/2
        interDistances.pop(inx)
        interDistances.insert(inx, newDist)
        interDistances.insert(inx+1, newDist)
        #pCandidate.pop(inx)
        pLeft = p(np.array([[leftSuccessor]]), inputSpaceBounds)[0,0]
        pRight = p(np.array([[rightSuccessor]]), inputSpaceBounds)[0,0]
        #pCandidate.insert(inx, pLeft)
        #pCandidate.insert(inx+1, pRight)
        scores.pop(inx)
        scores.insert(inx, pLeft*newDist)
        scores.insert(inx, pRight*newDist)
    return(np.row_stack(samples))
    
def myNoiseEstimate(X, y, evalX, numAveragingNbrs, numDiffNbrs = None):
    n, d = X.shape
    nEval = evalX.shape[0]
    if numDiffNbrs is None:
        numDiffNbrs = 2*d
    
    Xnbrs = NearestNeighbors(n_neighbors=numDiffNbrs+1, algorithm='ball_tree').fit(X)
    _, indices = Xnbrs.kneighbors(X)
    differences = np.abs(y - y.squeeze()[indices[:,1:]])
    if numAveragingNbrs >= n:
        localNoiseStd = np.ones(nEval) * np.median(differences) / (np.sqrt(2)*0.6745)
    else:
        Xnbrs = NearestNeighbors(n_neighbors=numAveragingNbrs, algorithm='ball_tree').fit(X)
        _, indices = Xnbrs.kneighbors(evalX)
        localNoiseStd = np.empty(nEval)
        for i in range(nEval):
            localNoiseStd[i] = np.median(differences[indices[i]]) / (np.sqrt(2)*0.6745)
    
    return(localNoiseStd**2)

def numpy_fillna(data):
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:,None]
    
    # get dtype of entries
    dd = data
    while isinstance(dd, list):
        dd = dd[0]
    dtype=dd.dtype

    # Setup output array and put elements from data into masked positions
    out = out = np.nan * np.empty(mask.shape, dtype=dtype)
    out[mask] = np.concatenate(data)
    return out

def myScientificFormat(x, digits=2):
    if digits is None:
        x = "{:E}".format(x)
    else:
        x = ("{:."+str(digits)+"E}").format(x)
    return(x.split('E')[0].rstrip('0').rstrip('.') + 'E' + x.split('E')[1])

class resultList():
    
    def __init__(self, resultFile = None, poolBased = False):

        resultLists = {}
        if os.path.isfile(resultFile):
            result = torch.load(resultFile)
            if 'allXs' in result:
                resultLists['allXs'] = list(result['allXs'])
                resultLists['allYs'] = list(result['allYs'])

            if 'allValXs' in result:
                resultLists['allValXs'] = list(result['allValXs'])
                resultLists['allValYs'] = list(result['allValYs'])

            if 'allTestXs' in result:
                resultLists['allTestXs'] = list(result['allTestXs'])
                resultLists['allTestYs'] = list(result['allTestYs'])

            if 'allTrainInx' in result:
                resultLists['allTrainInx'] = list(result['allTrainInx'])

            if 'allValInx' in result:
                resultLists['allValInx'] = list(result['allValInx'])

            if 'allTestInx' in result:
                resultLists['allTestInx'] = list(result['allTestInx'])

            resultLists['allDensitiesInformation'] = [list(rr) for rr in result['allDensitiesInformation']]
            resultLists['allSABERmodelsStateDict'] = [list(rr) for rr in result['allSABERmodelsStateDict']]
            resultLists['allPlotInformation'] = [list(rr) for rr in result['allPlotInformation']]
            resultLists['allTestErrors'] = [list(rr) for rr in result['allTestErrors']]
            resultLists['allGlobalModelsStateDict'] = [list(rr) for rr in result['allGlobalModelsStateDict']]
            resultLists['allGlobalTestErrors'] = [list(rr) for rr in result['allGlobalTestErrors']]
        else:
            if poolBased:
                resultLists['allTrainInx'] = []
                resultLists['allValInx'] = []
                resultLists['allTestInx'] = []
            else:
                resultLists['allXs'] = []
                resultLists['allYs'] = []
                resultLists['allValXs'] = []
                resultLists['allValYs'] = []
                resultLists['allTestXs'] = []
                resultLists['allTestYs'] = []
            resultLists['allDensitiesInformation'] = []
            resultLists['allSABERmodelsStateDict'] = []
            resultLists['allPlotInformation'] = []
            resultLists['allTestErrors'] = []
            resultLists['allGlobalModelsStateDict'] = []
            resultLists['allGlobalTestErrors'] = []
        self.resultLists = resultLists
        self.resultFile = resultFile
        
    def __getitem__(self, key):
        return self.resultLists.get(key, None)

    def __setitem__(self, key, value):
        self.resultLists.setdefault(key, value)
        
    def update(self, rep, X = None, y = None, trainInx = None, densitiesInformation = None, valX = None, valY = None, testX = None, testY = None, valInx = None, testInx = None, SABERmodel_stateDict = None, plotInformation = None, testErrors = None, globalModel_stateDict = None, globalTestErrors = None):

        if 'allXs' in self.resultLists and not X is None:
            if rep < len(self.resultLists['allXs']):
                self.resultLists['allXs'][rep] = X
                self.resultLists['allYs'][rep] = y
            else:
                self.resultLists['allXs'].append(X)
                self.resultLists['allYs'].append(y)

        if 'allTrainInx' in self.resultLists and not trainInx is None:
            if rep < len(self.resultLists['allTrainInx']):
                self.resultLists['allTrainInx'][rep] = trainInx
            else:
                self.resultLists['allTrainInx'].append(trainInx)

        if not densitiesInformation is None:
            if rep < len(self.resultLists['allDensitiesInformation']):
                self.resultLists['allDensitiesInformation'][rep] = densitiesInformation
            else:
                self.resultLists['allDensitiesInformation'].append(densitiesInformation)

        if 'allValXs' in self.resultLists and not valX is None:
            if rep < len(self.resultLists['allValXs']):
                self.resultLists['allValXs'][rep] = valX
                self.resultLists['allValYs'][rep] = valY
                self.resultLists['allTestXs'][rep] = testX
                self.resultLists['allTestYs'][rep] = testY
            else:
                self.resultLists['allValXs'].append(valX)
                self.resultLists['allValYs'].append(valY)
                self.resultLists['allTestXs'].append(testX)
                self.resultLists['allTestYs'].append(testY)

        if 'allValInx' in self.resultLists and not valInx is None:
            if rep < len(self.resultLists['allValInx']):
                self.resultLists['allValInx'][rep] = valInx
                self.resultLists['allTestInx'][rep] = testInx
            else:
                self.resultLists['allValInx'].append(valInx)
                self.resultLists['allTestInx'].append(testInx)

        if not SABERmodel_stateDict is None:
            if rep >= len(self.resultLists['allSABERmodelsStateDict']):
                self.resultLists['allSABERmodelsStateDict'].append([])
            self.resultLists['allSABERmodelsStateDict'][rep].append(SABERmodel_stateDict)

        if not plotInformation is None:
            if rep >= len(self.resultLists['allPlotInformation']):
                self.resultLists['allPlotInformation'].append([])
            self.resultLists['allPlotInformation'][rep].append(plotInformation)

        if not testErrors is None:
            if rep >= len(self.resultLists['allTestErrors']):
                self.resultLists['allTestErrors'].append([])
            self.resultLists['allTestErrors'][rep].append(testErrors)
            
        if not globalModel_stateDict is None:
            if rep >= len(self.resultLists['allGlobalModelsStateDict']):
                self.resultLists['allGlobalModelsStateDict'].append([])
            self.resultLists['allGlobalModelsStateDict'][rep].append(globalModel_stateDict)
            
        if not globalTestErrors is None:
            if rep >= len(self.resultLists['allGlobalTestErrors']):
                self.resultLists['allGlobalTestErrors'].append([])
            self.resultLists['allGlobalTestErrors'][rep].append(globalTestErrors)


    def save(self):
        res = {'allTestErrors': self.resultLists['allTestErrors'], 'allDensitiesInformation': self.resultLists['allDensitiesInformation'], 'allPlotInformation': self.resultLists['allPlotInformation'], 'allSABERmodelsStateDict': self.resultLists['allSABERmodelsStateDict'], 'allGlobalModelsStateDict': self.resultLists['allGlobalModelsStateDict'], 'allGlobalTestErrors': self.resultLists['allGlobalTestErrors']}
        if 'allXs' in self.resultLists:
            res['allXs'] = self.resultLists['allXs']
            res['allYs'] = self.resultLists['allYs']

        if 'allValXs' in self.resultLists:
            res['allValXs'] = self.resultLists['allValXs']
            res['allValYs'] = self.resultLists['allValYs']

        if 'allTestXs' in self.resultLists:
            res['allTestXs'] = self.resultLists['allTestXs']
            res['allTestYs'] = self.resultLists['allTestYs']

        if 'allTrainInx' in self.resultLists:
            res['allTrainInx'] = self.resultLists['allTrainInx']

        if 'allValInx' in self.resultLists:
            res['allValInx'] = self.resultLists['allValInx']

        if 'allTestInx' in self.resultLists:
            res['allTestInx'] = self.resultLists['allTestInx']

        torch.save(res, self.resultFile)
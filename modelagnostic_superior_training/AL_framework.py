"""
This module contains the definitions and routines of the model-agnostic, asymptotically superior AL framework in a pool-based or query synthesis AL scenario.

Details on this framework can be found in "Local Function Complexity for Active Learning via Mixture of Gaussian Processes" by (Panknin et. al, 2022) and "Optimal Sampling Density for Nonparametric Regression" by (Panknin et. al, 2021).
https://arxiv.org/abs/1902.10664, https://arxiv.org/abs/2105.11990
"""

from .AL_base import *

class activeLearner():
    
    def __init__(self, expPars, alPars, testDensity, randTestDistribution, randPInit, pInit, randUniform, labelOracle = None, xPool = None, yPool = None, densityPool = None, invalidPoolInx = None, testDensityPool = None, pInitPool = None, xRef = None):
        self.alPars = alPars
        outputDim = 0
        if expPars['labelModel']:
            outputDim += 1
        if expPars['derivativeModel']:
            outputDim += self.alPars['inputDim']
        self.outputDim = outputDim
        
        self.X = np.ones([0,self.alPars['inputDim']])
        self.y = np.ones([0,self.outputDim])
        self.trainInx = np.ones([0]).astype(int)
        
        self.importanceWeights = None
        self.randTestDistribution = randTestDistribution
        self.randPCurrent = randPInit
        self.pInit = pInit
        self.randUniform = randUniform
        self.testDensity = testDensity

        self.xRef = xRef
        if not self.xRef is None:
            self.oneNNXRef = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.xRef)
        else:
            self.oneNNXRef = None
            
        self.xPool = xPool
        self.yPool = yPool
        self.densityPool = densityPool
        self.invalidPoolInx = invalidPoolInx
        self.testDensityPool = testDensityPool
        self.pInitPool = pInitPool
            
        self.densitiesInformation = []
        self.localNoiseVariance = None
        self.localBandwidthFunction = None
        self.labelOracle = labelOracle
        
    def importPreviousData(self, X, y, trainInx = None, densitiesInformation = []):
        self.X = X
        self.y = y
        if not trainInx is None:
            self.trainInx = trainInx
        self.densitiesInformation = densitiesInformation
        self.updateTrainImportanceWeights()
        
    def updateTrainImportanceWeights(self):
        if self.xPool is None:
            self.importanceWeights = self.testDensity(self.X, self.alPars['inputSpaceBounds']) / self.densitiesInformation[-1]['trainPCurrent']
        else:
            self.importanceWeights = self.testDensityPool[self.trainInx] / self.densitiesInformation[-1]['trainPCurrent']
            
        self.importanceWeights *= len(self.importanceWeights)/np.sum(self.importanceWeights)
        
    def oneNNtoXRefPredictor(self, x, propertyRef):
        _, indices = self.oneNNXRef.kneighbors(x)
        return(propertyRef[indices.ravel()].ravel())
            
    def sampleBatch(self, size, trainInx = None):
        if self.xPool is None:
            if self.randPCurrent is None:
                xNew = self.randTestDistribution(size, self.alPars['inputSpaceBounds'], self.X)
            else:
                xNew = self.randPCurrent(size, self.alPars['inputSpaceBounds'], self.X)
            yNew = self.labelOracle(xNew)
            inxNew = None
        else:
            # case that we already have precalculated the trainInx
            if not trainInx is None:
                inxNew = trainInx
            else:
                if self.randPCurrent is None:
                    inxNew = self.randTestDistribution(size, self.invalidPoolInx)
                else:
                    inxNew = self.randPCurrent(size, self.invalidPoolInx)
            xNew = self.xPool[inxNew].reshape(-1, self.alPars['inputDim'])
            yNew = self.yPool[inxNew]
        return xNew, yNew, inxNew
            
    def growTrainingData(self, batchSize, trainInx = None):
        xNew, yNew, inxNew = self.sampleBatch(batchSize, trainInx)
        
        self.X = np.row_stack((self.X, xNew))
        self.y = np.row_stack((self.y, yNew))
        if not self.xPool is None:
            self.trainInx = np.concatenate((self.trainInx, inxNew))
            self.invalidPoolInx = np.concatenate((self.invalidPoolInx, inxNew))
        
        # set up a dictionary for later reconstruction of local properties from intermediate iterations
        densityInformation = {}
        densityInformation['globalNoiseLevel'] = None
        densityInformation['vRef'] = None
        densityInformation['sigmaRef'] = None
        densityInformation['complexityRef'] = None
        densityInformation['proposalDensityRef'] = None
        if self.xRef is None:
            densityInformation['trainingDensityRef'] = None
        else:
            # before appending density information, the newest pNext is the actual training density at this point; after appending, pNext describes the next density
            densityInformation['trainingDensityRef'] = self.pNext(self.xRef)
        densityInformation['trainSize'] = len(self.X)
        densityInformation['proposalDensityNormalization'] = 1.
        densityInformation['vOfPool'] = None
        densityInformation['sigmaOfPool'] = None
        densityInformation['complexityOfPool'] = None
        densityInformation['proposalDensityOfPool'] = None
        if self.xPool is None:
            densityInformation['trainingDensityOfPool'] = None
        else:
            # before appending density information, the newest pNext is the actual training density at this point; after appending, pNext describes the next density
            densityInformation['trainingDensityOfPool'] = self.pNext()
        densityInformation['densityWeight'] = None
        densityInformation['trainPCurrent'] = self.pNext(self.X, self.trainInx)
        self.densitiesInformation.append(densityInformation)
        
        
    #this one is the new density after obtaining the new batch
    def pNext(self, x = None, inxes = None):
        if self.xPool is None or (not x is None and inxes is None):
            dens = self.pInit(x, self.alPars['inputSpaceBounds'])
            if not self.alPars['updateTrainingDensity']:
                return dens
            if len(self.densitiesInformation) > 0:
                if not self.xRef is None:
                    _, indices = self.oneNNXRef.kneighbors(x)
                for i in range(len(self.densitiesInformation)):
                    w = self.densitiesInformation[i]['densityWeight']
                    if not self.xRef is None:
                        dens = dens * w + self.densitiesInformation[i]['proposalDensityRef'][indices.flat] * (1-w)
                    else:
                        dens = dens * w + self.proposalDensityEstimate(x, fromIteration = i) * (1-w)
        else:
            if inxes is None:
                dens = self.pInitPool
            else:
                dens = self.pInitPool[inxes]
            if not self.alPars['updateTrainingDensity']:
                return dens
            if len(self.densitiesInformation) > 0:
                for i in range(len(self.densitiesInformation)):
                    w = self.densitiesInformation[i]['densityWeight']
                    if inxes is None:
                        dens = dens * w + self.densitiesInformation[i]['proposalDensityOfPool'] * (1-w)
                    else:
                        dens = dens * w + self.densitiesInformation[i]['proposalDensityOfPool'][inxes] * (1-w)
            
        return(dens)
        
    # here, we calculate pCurrent at the desired iteration
    def pCurrent(self, x = None, inxes = None, fromIteration = -1):
        if self.xPool is None or (not x is None and inxes is None):
            if len(self.densitiesInformation) == 0 or len(self.densitiesInformation) < fromIteration + 1:
                # there are no samples of this iteration, so far
                return None
            dens = self.pInit(x, self.alPars['inputSpaceBounds']) # pCurrent(x) of iteration 0
            if not self.alPars['updateTrainingDensity']:
                return dens
            if len(self.densitiesInformation) > 1 and not fromIteration == 0:
                if fromIteration == -1:
                    fromIteration = len(self.densitiesInformation) - 1
                if not self.xRef is None:
                    _, indices = self.oneNNXRef.kneighbors(x)
                for i in range(fromIteration):
                    # construct pCurrent(x) of iteration i+1 from pCurrent(x) of iteration i, beginning with i = 0
                    w = self.densitiesInformation[i]['densityWeight']
                    if not self.xRef is None:
                        dens = dens * w + self.densitiesInformation[i]['proposalDensityRef'][indices.flat] * (1-w)
                    else:
                        dens = dens * w + self.proposalDensityEstimate(x, fromIteration = i, pCurrentOfX = dens) * (1-w) # # proposalDensityEstimate(x) of iteration i requires pCurrent(x) of iteration i
        else:
            # estimate current density on xPool
            if len(self.densitiesInformation) == 0 or len(self.densitiesInformation) < fromIteration + 1:
                # there are no samples of this iteration, so far
                return None
            if inxes is None:
                dens = self.pInitPool
            else:
                dens = self.pInitPool[inxes]
            if not self.alPars['updateTrainingDensity']:
                return dens
            if len(self.densitiesInformation) > 1 and not fromIteration == 0:
                if fromIteration == -1:
                    fromIteration = len(self.densitiesInformation) - 1
                for i in range(fromIteration):
                    w = self.densitiesInformation[i]['densityWeight']
                    if inxes is None:
                        dens = dens * w + self.densitiesInformation[i]['proposalDensityOfPool'] * (1-w)
                    else:
                        dens = dens * w + self.densitiesInformation[i]['proposalDensityOfPool'][inxes] * (1-w)
        return(dens)
            
# in order to estimate self.proposalDensityEstimate(x, fromIteration = i), we require pCurrent(self, x, inputSpaceBounds, fromIteration = i)
# recursively, to calculate pCurrent(self, x, inputSpaceBounds, fromIteration = i+1), we require self.proposalDensityEstimate(x, fromIteration = i)
# we can however pass the earlier estimates of pCurrent to the respective proposalDensityEstimate to break the recursion
        
    def getMaxOfDensity(self, p):
        if not self.xRef is None:
            return np.max(p(self.xRef))
        else:
            maxP = -np.Inf
            maxCounter = 10
            counter = 0
            batchSize = 1000
            threshold = 1e-3
            while counter < maxCounter:
                oldMax = maxP
                batch = self.randUniform(batchSize, self.alPars['inputSpaceBounds'])
                maxP = max(maxP, np.max(p(batch)))
                if (maxP - oldMax) / maxP < threshold:
                    counter += 1
                else:
                    counter = 0
            return maxP
                
                    
    def updateTrainingDistribution(self):
        if len(self.X) > 0:
            if self.xPool is None:
                if len(self.densitiesInformation) > 0 and self.alPars['updateTrainingDensity']:
                    if self.alPars['samplingAlgorithm'] == 'Random':
                        # propose new density that adapts fast to the new proposal density:
                        def pAdvanced(x):
                            return(np.maximum(2*self.pNext(x) - self.pCurrent(x),0))
                        pAdvancedMax = self.getMaxOfDensity(pAdvanced)
                        def randPCurrent(size, inputSpaceBounds, xTrain):
                            return(myRejectionSampling(p=pAdvanced, inputSpaceBounds=inputSpaceBounds, pMax = pAdvancedMax, size = size))
                    elif self.alPars['samplingAlgorithm'] == 'KMeans':

                        pMax = self.getMaxOfDensity(self.pNext)
                        feature_perms = None
                        featureFunction = None
                        preSampleSizeFactor = 10
                        def randPCurrent(size, inputSpaceBounds, xTrain):
                            return(kMeansSamplingFromDensity(size = size, p = self.pNext, inputSpaceBounds = inputSpaceBounds, pMax = pMax, fixedCenters = xTrain))
                        
                    elif self.alPars['samplingAlgorithm'] == 'SVGD':
                        raise NotImplementedError
                    else:
                        raise NotImplementedError
                    self.randPCurrent = randPCurrent
            else:
                if len(self.densitiesInformation) > 0 and self.alPars['updateTrainingDensity']:
                    # prepare to sample from pool according to proposed density
                    pAdvancedOfPool = np.maximum(2*self.pNext() - self.pCurrent(),0)
                    
                    if self.alPars['samplingAlgorithm'] == 'Random':
                        def randPCurrent(size, invalidPoolInx):
                            return(myPoolSampling(size, pAdvancedOfPool, self.densityPool, invalidPoolInx))
                    elif self.alPars['samplingAlgorithm'] == 'KMeans':
                        # we can also go on sampling via kMeans clustering
                        
                        #preSampleSizeFactor = 10
                        #maxDensityRatioForSampleSizeFactor = np.max(self.densitiesInformation[fromIteration]['trainingDensityOfPool'] / self.densitiesInformation[fromIteration]['proposalDensityOfPool'])
                        #maxDensityRatio = max(preSampleSizeFactor+1, maxDensityRatio)
                        #newWeight = max(0, (1/(preSampleSizeFactor+1) - 1/maxDensityRatio) / (1 - 1/maxDensityRatio))
                        pAdvancedForSampleSizeFactor = self.pNext() # TODO that's not accurate, but probably close enough
                        
                        feature_perms = None
                        featureFunction = None
                        
                        def randPCurrent(size, pars, invalidPoolInx):
                            return(kMeansSamplingFromPool(size, self.MDDataset, pAdvancedForSampleSizeFactor, self.densityPool, invalidPoolInx, perms = feature_perms, featureFunction = featureFunction, init_centers = self.alPars['KMeansInit'], distributionalClustering = self.alPars['KMeansDistributional'], max_iter = self.alPars['KMeansMaxIter'], intrinsicDim = self.alPars['intrinsicDim'], fixedCenters = self.X, preSampleSizeFactor = preSampleSizeFactor))
                    else:
                        raise NotImplementedError
                    
                    self.randPCurrent = randPCurrent
                    
    def noiseLevelEstimate(self, x = None, inxes = None, fromIteration = -1):
        if self.xPool is None or (not x is None and inxes is None):
            if self.densitiesInformation[fromIteration]['vRef'] is None:
                return self.localNoiseVariance(x)
            else:
                return self.oneNNtoXRefPredictor(x, self.densitiesInformation[fromIteration]['vRef']).ravel()
        else:

            if self.densitiesInformation[fromIteration]['vOfPool'] is None:
                if inxes is None:
                    return self.localNoiseVariance(self.xPool)
                else:
                    return self.localNoiseVariance(self.xPool[inxes])
            else:
                if inxes is None:
                    return self.densitiesInformation[fromIteration]['vOfPool']
                else:
                    return self.densitiesInformation[fromIteration]['vOfPool'][inxes]
                
# if xRef is not specified: Summarize, what information we require to evaluate the training densities over the several iterations in hindsight:
# we require self.noiseLevelEstimate(x), which can be calculated from (self.X, self.trainResiduals) at training size n_iter, or self.globalNoiseLevel if homoscedastic
# we require self.localVolumeEstimate(x), which can be calculated from the forest: ['child_nodes'], ['split_vars'], ['split_values'] and ['nodeVolumes']
# we require self.pCurrent(x), which can be constructed from the previous pCurrent, noiseLevelEstimate and localVolumeEstimate
                
    def localBandwidthEstimate(self, x = None, inxes = None, fromIteration = -1):
        if self.xPool is None or (not x is None and inxes is None):
            if self.densitiesInformation[fromIteration]['sigmaRef'] is None:
                return self.LOBestimate(x)
            else:
                return self.oneNNtoXRefPredictor(x, self.densitiesInformation[fromIteration]['sigmaRef']).ravel()
        else:
            if self.densitiesInformation[fromIteration]['sigmaOfPool'] is None:
                if inxes is None:
                    return self.LOBestimate(self.xPool)
                else:
                    return self.LOBestimate(self.xPool[inxes])
            else:
                if inxes is None:
                    return self.densitiesInformation[fromIteration]['sigmaOfPool']
                else:
                    return self.densitiesInformation[fromIteration]['sigmaOfPool'][inxes]
            
    def localFunctionComplexityEstimate(self, x = None, inxes = None, fromIteration = -1, pCurrentOfX = None, bandwidthOfX = None):
        if self.alPars['intrinsicDim'] is None:
            d = self.alPars['inputDim']
        else:
            d = self.alPars['intrinsicDim']
        if self.xPool is None or (not x is None and inxes is None):
            if self.densitiesInformation[fromIteration]['complexityRef'] is None:
                if pCurrentOfX is None:
                    pCurrentOfX = self.pCurrent(x, fromIteration=fromIteration)
                if bandwidthOfX is None:
                    bandwidthOfX = self.localBandwidthEstimate(x, fromIteration=fromIteration)
                # distinguish between cases of the specification of the sigma-function
                if len(bandwidthOfX.shape) == 1:
                    # isotropic
                    bandwidthVolumeOfX = bandwidthOfX**d
                if len(bandwidthOfX.shape) == 2:
                    if bandwidthOfX.shape[1] == 1:
                        # isotropic:
                        bandwidthVolumeOfX = bandwidthOfX**d
                    else:
                        # anisotropic:
                        bandwidthVolumeOfX = bandwidthOfX.prod(1)
                if len(bandwidthOfX.shape) == 3:
                    # full matrix:
                    bandwidthVolumeOfX = np.array([np.linalg.det(sss) for sss in bandwidthOfX])
                if not self.alPars['noiseFree']:
                    if self.alPars['correctLFCforNoise']:
                        v = self.noiseLevelEstimate(x, fromIteration=fromIteration)
                    else:
                        # assumption: heteroscedastic GP already treats v in the model. Therefore it will not be present in LOB, and thus LFC requires no correction wrt. v
                        v = 1.
                else:
                    v = 1.
                return (v/pCurrentOfX)**(d/(2*(self.alPars['Q']+1)+d))/bandwidthVolumeOfX
            else:
                return self.oneNNtoXRefPredictor(x, self.densitiesInformation[fromIteration]['complexityRef']).ravel()
        else:
            if self.densitiesInformation[fromIteration]['complexityOfPool'] is None:
                pCurrentOfX = self.pCurrent(inxes=inxes, fromIteration=fromIteration)
                bandwidthOfX = self.localBandwidthEstimate(inxes=inxes, fromIteration=fromIteration)
                # distinguish between cases of the specification of the sigma-function
                if len(bandwidthOfX.shape) == 1:
                    # isotropic
                    bandwidthVolumeOfX = bandwidthOfX**d
                if len(bandwidthOfX.shape) == 2:
                    if bandwidthOfX.shape[1] == 1:
                        # isotropic:
                        bandwidthVolumeOfX = bandwidthOfX**d
                    else:
                        # anisotropic:
                        bandwidthVolumeOfX = bandwidthOfX.prod(1)
                if len(bandwidthOfX.shape) == 3:
                    # full matrix:
                    bandwidthVolumeOfX = np.array([np.linalg.det(sss) for sss in bandwidthOfX])
                if not self.alPars['noiseFree']:
                    if self.alPars['correctLFCforNoise']:
                        v = self.noiseLevelEstimate(inxes=inxes, fromIteration=fromIteration)
                    else:
                        # assumption: heteroscedastic GP already treats v in the model. Therefore it will not be present in LOB, and thus LFC requires no correction wrt. v
                        v = 1.
                else:
                    v = 1.
                return (v/pCurrentOfX)**(d/(2*(self.alPars['Q']+1)+d))/bandwidthVolumeOfX
            else:
                if inxes is None:
                    return self.densitiesInformation[fromIteration]['complexityOfPool']
                else:
                    return self.densitiesInformation[fromIteration]['complexityOfPool'][inxes]
  
    def proposalDensityEstimate(self, x = None, inxes = None, fromIteration = -1, pCurrentOfX = None, q = None, lfc = None):
        if self.alPars['intrinsicDim'] is None:
            d = self.alPars['inputDim']
        else:
            d = self.alPars['intrinsicDim']
        if self.xPool is None or (not x is None and inxes is None):
            if self.densitiesInformation[fromIteration]['proposalDensityRef'] is None:
                if lfc is None:
                    lfc = self.localFunctionComplexityEstimate(x, fromIteration=fromIteration, pCurrentOfX=pCurrentOfX)
                if q is None:
                    q = self.testDensity(x, self.alPars['inputSpaceBounds'])
                if not self.alPars['noiseFree']:
                    v = self.noiseLevelEstimate(x, fromIteration=fromIteration)
                else:
                    v = 1.
                if self.alPars['Q'] == np.Inf:
                    return (lfc*q)**(0.5)*v**(0.5) / self.densitiesInformation[fromIteration]['proposalDensityNormalization']
                else:
                    return (lfc*q)**((2*(self.alPars['Q']+1)+d)/(4*(self.alPars['Q']+1)+d))*v**(2*(self.alPars['Q']+1)/(4*(self.alPars['Q']+1)+d)) / self.densitiesInformation[fromIteration]['proposalDensityNormalization']
            else:
                return self.oneNNtoXRefPredictor(x, self.densitiesInformation[fromIteration]['proposalDensityRef']).ravel()
        else:
            if self.densitiesInformation[fromIteration]['proposalDensityOfPool'] is None:
                lfc = self.localFunctionComplexityEstimate(inxes=inxes, fromIteration=fromIteration, pCurrentOfX=pCurrentOfX)
                if inxes is None:
                    q = self.testDensityPool
                else:
                    q = self.testDensityPool[inxes]
                if not self.alPars['noiseFree']:
                    v = self.noiseLevelEstimate(inxes=inxes, fromIteration=fromIteration)
                else:
                    v = 1.
                if self.alPars['Q'] == np.Inf:
                    return (lfc*q)**(0.5)*v**(0.5) / self.densitiesInformation[fromIteration]['proposalDensityNormalization']
                else:
                    return (lfc*q)**((2*(self.alPars['Q']+1)+d)/(4*(self.alPars['Q']+1)+d))*v**(2*(self.alPars['Q']+1)/(4*(self.alPars['Q']+1)+d)) / self.densitiesInformation[fromIteration]['proposalDensityNormalization']
                raise NotImplementedError
            else:
                if inxes is None:
                    return self.densitiesInformation[fromIteration]['proposalDensityOfPool']
                else:
                    return self.densitiesInformation[fromIteration]['proposalDensityOfPool'][inxes]
                
            
    def calculateProposalDensityNorm(self, fromIteration = -1):
        if self.densitiesInformation[fromIteration]['proposalDensityRef'] is None and self.densitiesInformation[fromIteration]['proposalDensityOfPool'] is None:
            self.densitiesInformation[fromIteration]['proposalDensityNormalization'] = 1.
            # do uniform sampling over input space to estimate norm (monte carlo integration)
            # like below in uniform case, we obtain p(x) = volX * P(x), which is determined up to a constant that does not depend on p!
            norm = 0.
            n = 0
            batchSize = 1000
            threshold = 1e-3
            while True:
                nOld = n
                normOld = norm
                batch = self.randUniform(batchSize, self.alPars['inputSpaceBounds'])
                n += 1000
                norm = (normOld*nOld + self.proposalDensityEstimate(batch, fromIteration = fromIteration).sum())/n
                if np.abs(norm-normOld) / norm < threshold:
                    break
            self.densitiesInformation[fromIteration]['proposalDensityNormalization'] = norm
        elif not self.densitiesInformation[fromIteration]['proposalDensityRef'] is None:
            # assume we are given p(x) = a * P(x), where P is a probability density and a is the unknown constant
            #we can normalize P, given p and U(x) as follows:
            #estimate c = mean_{x in unifX} p(x) = int_z p(z) U(z) dz = a / volX
            #then set p(x) <- p(x) / c such that p(x) = volX * P(x) is determined up to a constant that does not depend on p!
            self.densitiesInformation[fromIteration]['proposalDensityNormalization'] = np.mean(self.densitiesInformation[fromIteration]['proposalDensityRef'])
            self.densitiesInformation[fromIteration]['proposalDensityRef'] /= self.densitiesInformation[fromIteration]['proposalDensityNormalization']
        else:
            #even more generally, given samples x ~ Q with an unnormalized density estimate q(x) = b * Q(x), we can do importance sampling
            #estimate c = mean_{x ~ Q} p(x) / q(x) =  int_z p(z) / q(z) * Q(z) dz = a/b
            #by setting p(x) <- p(x) / c, it is p(x) = b * P(x) etermined up to a constant that does not depend on p!
            # we reproduce the uniform case by setting q(x) = 1. Then it follows b = volX
            self.densitiesInformation[fromIteration]['proposalDensityNormalization'] = np.mean(self.densitiesInformation[fromIteration]['proposalDensityOfPool'] / self.densityPool)
            self.densitiesInformation[fromIteration]['proposalDensityOfPool'] /= self.densitiesInformation[fromIteration]['proposalDensityNormalization']
        
    # if all training and proposal density estimates have the same (unnormalized) constant p(x) = a * P(x), q(x) = a * Q(x), then their ratio W(x) = p(x)/q(x) = P(X)/Q(X) is directly normalized.
    # however, if we cannot guarantee this we can go on as follows:
    #assume p(x) = a * P(x), q(x) = b * Q(x), and let w(x) = p(x) / q(x)
    #let: c = mean_{x ~ Q} w(x) = int_z p(z) / q(z) * Q(z) = int_z P(z) * a / b = a / b
    #then W(x) = w(x)/c = P(x) / Q(x) are the true importance weights
    # Current implementation assumes the first case!!!!!!!!!!
    def updateMaxCurrentToProposalDensityRatio(self, fromIteration = -1):
        if self.densitiesInformation[fromIteration]['proposalDensityRef'] is None and self.densitiesInformation[fromIteration]['proposalDensityOfPool'] is None:
            maxDensityRatio = -np.Inf
            maxCounter = 10
            counter = 0
            batchSize = 1000
            threshold = 1e-3
            while counter < maxCounter:
                oldMaxRatio = maxDensityRatio
                batch = self.randUniform(batchSize, self.alPars['inputSpaceBounds'])
                pCurrent = self.pCurrent(batch, fromIteration = fromIteration)
                pProposal = self.proposalDensityEstimate(batch, fromIteration = fromIteration)
                maxDensityRatio = max(maxDensityRatio, np.max(pCurrent / pProposal))
                if (maxDensityRatio - oldMaxRatio) / maxDensityRatio < threshold:
                    counter += 1
                else:
                    counter = 0
        elif not self.densitiesInformation[fromIteration]['proposalDensityRef'] is None:
            maxDensityRatio = np.max(self.densitiesInformation[fromIteration]['trainingDensityRef'] / self.densitiesInformation[fromIteration]['proposalDensityRef'])
        else:
            maxDensityRatio = np.max(self.densitiesInformation[fromIteration]['trainingDensityOfPool'] / self.densitiesInformation[fromIteration]['proposalDensityOfPool'])
        # ratios below 2 can always be eliminated completely
        maxDensityRatio = max(2, maxDensityRatio)
        newWeight = max(0, (0.5 - 1/maxDensityRatio) / (1 - 1/maxDensityRatio))
        self.densitiesInformation[fromIteration]['densityWeight'] = newWeight
        
    def updateLocalProperties(self, localBandwidthFunction, localNoiseVariance):
        
        self.LOBestimate = localBandwidthFunction
        self.localNoiseVariance = localNoiseVariance
        
        # set adaptive hyperparameters
        ##########################################
        
        # the SABER model provides an explicit noise function
    
        # update noise level estimate
        if not self.alPars['noiseFree']:
            
            if not self.xRef is None:
                print('estimate reference point noise levels')
                self.densitiesInformation[-1]['vRef'] = self.noiseLevelEstimate(self.xRef)
            if not self.xPool is None:
                self.densitiesInformation[-1]['vOfPool'] = self.noiseLevelEstimate()
        
        if not self.xRef is None:
            print('estimate reference point LOB')
            # update local volume estimate
            self.densitiesInformation[-1]['sigmaRef'] = self.localBandwidthEstimate(self.xRef)

            print('estimate reference point LFC')
            # update proposal density estimate
            self.densitiesInformation[-1]['complexityRef'] = self.localFunctionComplexityEstimate(self.xRef)
            
            print('estimate reference point pOpt')
            # we need to recalculate the density normalization constant of the proposal density
            self.densitiesInformation[-1]['proposalDensityRef'] = self.proposalDensityEstimate(self.xRef)
        if not self.xPool is None:
            self.densitiesInformation[-1]['sigmaOfPool'] = self.localBandwidthEstimate()
            self.densitiesInformation[-1]['complexityOfPool'] = self.localFunctionComplexityEstimate()
            self.densitiesInformation[-1]['proposalDensityOfPool'] = self.proposalDensityEstimate()
            
        self.calculateProposalDensityNorm()
        self.updateMaxCurrentToProposalDensityRatio()
        
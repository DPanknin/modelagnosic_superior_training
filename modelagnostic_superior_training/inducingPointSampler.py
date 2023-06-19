"""
This module defined a meta function that sets up the inducing point construction function, based on so far obtained information from the AL framework.
"""

from .inducingPointMethods import SVGD_inducing_points, kMeans_inducing_points, random_inducing_points, GFF_inducing_points

def initIPsamplerPars(ipSelectionPars = None, alPars = None, modelPars = None):
    pars = {}
    pars['ipComplexityExponent'] = 0.
    
    pars['svgdIters'] = 200
    pars['svgdRepulsionPropToDensity'] = True
    pars['svgdVersion'] = 'v2'
    pars['svgdIPrefSize'] = None
    pars['svgdSigma'] = 1e0
    pars['svgdInitStepSize'] = 1e-1
    pars['svgdFinalStepSize'] = 1e-3

    pars['KMeansInit'] = 'k-means++' # 'random', 'k-means++'
    pars['KMeansMaxIter'] = 300
    pars['KMeansDistributional'] = True
    
    pars['GFFthreshold'] = 1e-2
    
    pars['inputDim'] = None
    pars['intrinsicDim'] = None
    pars['inputSpaceBounds'] = None
    
    pars['applyARD'] = False
    pars['doublePrecision'] = False
    
    if not ipSelectionPars is None:
        pars['ipComplexityExponent'] = ipSelectionPars['ipComplexityExponent']
        
        pars['svgdIPrefSize'] = ipSelectionPars['svgdIPrefSize']
        pars['svgdSigma'] = ipSelectionPars['svgdSigma']
        pars['svgdInitStepSize'] = ipSelectionPars['svgdInitStepSize']
        pars['svgdFinalStepSize'] = ipSelectionPars['svgdFinalStepSize']
        pars['svgdRepulsionPropToDensity'] = ipSelectionPars['svgdRepulsionPropToDensity']
        pars['svgdVersion'] = ipSelectionPars['svgdVersion']
        pars['svgdIters'] = ipSelectionPars['svgdIters']
        
        pars['KMeansInit'] = ipSelectionPars['KMeansInit']
        pars['KMeansDistributional'] = ipSelectionPars['KMeansDistributional']
        pars['KMeansMaxIter'] = ipSelectionPars['KMeansMaxIter']
        
        pars['GFFthreshold'] = ipSelectionPars['GFFthreshold']
        
    if not alPars is None:
        pars['inputDim'] = alPars['inputDim']
        pars['intrinsicDim'] = alPars['intrinsicDim']
        pars['inputSpaceBounds'] = alPars['inputSpaceBounds']
        
    if not modelPars is None:
        pars['applyARD'] = modelPars['applyARD']
        pars['doublePrecision'] = modelPars['doublePrecision']
    
    return pars
        
class inducingPointSampler():
    
    def __init__(self, saberAL, pars = None):
        
        if pars is None:
            self.pars = initIPsamplerPars()
        else:
            self.pars = pars
        
        self.pTrain = saberAL.pCurrent
        self.pTrainOfX = saberAL.densitiesInformation[-1]['trainPCurrent']
        self.X = saberAL.X
        self.y = saberAL.y
        self.trainInx = saberAL.trainInx
        
        if self.pars['ipComplexityExponent'] > 0 and len(saberAL.densitiesInformation) > 1:
            def pTarget(x, inputSpaceBounds = None):
                return(saberAL.localFunctionComplexityEstimate(x, fromIteration=-2)**self.pars['ipComplexityExponent'] * saberAL.pCurrent(x)**(1-self.pars['ipComplexityExponent']))
            # try to estimate pTargetOfX
            # for this, we need their LFC estimates
            if saberAL.xPool is None:
                LFCofX = saberAL.localFunctionComplexityEstimate(self.X, fromIteration=-2)
            else:
                LFCofX = saberAL.localFunctionComplexityEstimate(inxes = self.trainInx, fromIteration=-2)
            pTargetOfX = LFCofX**self.pars['ipComplexityExponent'] * self.pTrainOfX**(1-self.pars['ipComplexityExponent'])
        else:
            pTarget = saberAL.pCurrent
            pTargetOfX = self.pTrainOfX
            
        self.pTarget = pTarget
        self.pTargetOfX = pTargetOfX
        
    def __call__(self, IPstrategy, trainingSubInx = None):

        if trainingSubInx is None:
            X = self.X
            y = self.y
            pTrainOfX = self.pTrainOfX
            pTargetOfX = self.pTargetOfX
        else:
            X = self.X[trainingSubInx]
            y = self.y[trainingSubInx]
            pTrainOfX = self.pTrainOfX[trainingSubInx]
            pTargetOfX = self.pTargetOfX[trainingSubInx]
        
        if IPstrategy == 'Random':
            inducing_point_method = random_inducing_points(X, pTrainOfX = pTrainOfX, pTargetOfX = pTargetOfX, pTrain = self.pTrain, pTarget = self.pTarget)
            
        if IPstrategy == 'SVGD':
            inducing_point_method = SVGD_inducing_points(X, self.pars['inputSpaceBounds'], self.pars['svgdIPrefSize'], self.pars['svgdSigma'], self.pars['svgdInitStepSize'], self.pars['svgdFinalStepSize'], self.pars['svgdRepulsionPropToDensity'], self.pars['svgdVersion'], self.pars['svgdIters'], pTrainOfX = pTrainOfX, pTargetOfX = pTargetOfX, pTrain = self.pTrain, pTarget = self.pTarget, intrinsicDim = self.pars['intrinsicDim'])
            
        if IPstrategy == 'KMeans':
            inducing_point_method = kMeans_inducing_points(X, self.pars['KMeansInit'], self.pars['KMeansDistributional'], self.pars['KMeansMaxIter'], pTrainOfX = pTrainOfX, pTrain = self.pTrain, intrinsicDim = self.pars['intrinsicDim'])
            
        if IPstrategy == 'GFF':
            inducing_point_method = GFF_inducing_points(X, y, 1e0, (torch.ones([1,self.pars['inputDim']]) if self.pars['applyARD'] else 1e0), 1e0, informationThreshold = self.pars['GFFthreshold'], doublePrecision = self.pars['doublePrecision'])
        
        return inducing_point_method
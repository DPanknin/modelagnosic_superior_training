"""
This module contains all definitions and routines of the sparse mixture of Gaussian processes regression model.

Details on this model can be found in "Local Function Complexity for Active Learning via Mixture of Gaussian Processes" by (Panknin et. al, 2022)
https://arxiv.org/abs/1902.10664

The model is built using the GPyTorch library.

Details on this library can be found in "GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration" by (Gardner et. al, 2018)
https://proceedings.neurips.cc/paper/2018/file/27e8e17134dd7083b050476733207ea1-Paper.pdf
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import Tensor
import gpytorch
from gpytorch.constraints import Positive
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls._approximate_mll import _ApproximateMarginalLogLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.utils.memoize import clear_cache_hook, _add_to_cache_ignore_args, _is_in_cache_ignore_args
import numpy as np
from .AL_base import myScientificFormat
import functools
import math
from .globalModel import *
from .moe import MoE, SparseDispatcher
from copy import copy

def initSABERpars(expPars = None, modelPars = None, alPars = None):
    pars = {}
    pars['splitRatio'] = 1
    pars['labelModel'] = True
    pars['derivativeModel'] = False
    pars['doublePrecision'] = False
    pars['kernel'] = 'gaussian'
    pars['jitter'] = 0.
    pars['cholJitter'] = 1e-16
    pars['noiseFree'] = False
    pars['homoscedastic'] = True
    pars['has_global_noise'] = True
    pars['has_task_noise'] = False
    
    pars['applyARD'] = False
    pars['initARDscales'] = None
    
    pars['validationFrequency'] = 1
    pars['subValidationFrequency'] = 1
    pars['patienceFactor'] = None
    pars['weightDecay'] = 0.
    pars['LR'] = 1e-1
    pars['LR_pre'] = 1e0
    pars['minLR'] = 1e-3
    pars['lrFactorGate'] = 1e0
    pars['lrFactorGPRmean'] = 1.
    pars['lrFactorGPRhyperparameters'] = 2e-1
    pars['lrFactorIPlocations'] = 1e-2
    pars['minEPOCH'] = 0
    pars['maxEPOCH'] = np.Inf
    
    # expert model hyperpars
    pars['expertModel'] = 'exactGP'
    pars['preTrainEpochsExpert'] = int(2**6)
    
    pars['numIPsExpert'] = None
    pars['batchSizeExpert'] = None
    
    pars['initMeanExpert'] = None
    pars['initLambdaExpert'] = 1e0
    pars['initNoiseLevelExpert'] = 1e0
    
    pars['fixedNoiseExpert'] = False
    pars['fixedMeanExpert'] = False
    pars['fixedLambdaExpert'] = True
    pars['fixedIPlocationsExpert'] = True
    pars['inducingCovarTypeExpert'] = 'scalar' # 'scalar', 'diag' or 'full'
    
    # gate model hyperpars
    pars['numIPsGate'] = None
    pars['batchSizeGate'] = None

    pars['initSigmaGate'] = 1e0
    pars['initLambdaGate'] = 1e0

    pars['fixedSigmaGate'] = True
    pars['fixedLambdaGate'] = True
    pars['fixedIPlocationsGate'] = True
    pars['gateOutputType'] = 'independent' #'independent', 'dependent'
    pars['inducingCovarTypeGate'] = 'scalar' # 'scalar', 'diag' or 'full'

    # MoE model hyperpars
    pars['fixedExperts'] = False
    pars['expertBandwidths'] = []
    pars['smallBandwidthPenalty'] = 0.
    pars['expertHardSparsity'] = len(pars['expertBandwidths']) # the final maximal number of experts the gate chooses; note that we loose continuity of the regressor, iff chosen too small
    pars['sparsifyAfterValiationIterations'] = 2
    pars['initSparsificationAfterValiationIterations'] = 5
    pars['noisy_gating'] = True
    pars['noisy_gating_decaying'] = 0.5 #0.
    pars['noisy_gating_b'] = False
    pars['noisy_gating_w'] = False
    pars['gate_noise_stdDev'] = 1e-1
    
    if not expPars is None:
        pars['splitRatio'] = expPars['splitRatio']
        pars['labelModel'] = expPars['labelModel']
        pars['derivativeModel'] = expPars['derivativeModel']

    if not modelPars is None:
        pars['doublePrecision'] = modelPars['doublePrecision']
        pars['kernel'] = modelPars['kernel']
        pars['jitter'] = modelPars['jitter']
        pars['cholJitter'] = modelPars['cholJitter']
        pars['has_global_noise'] = modelPars['has_global_noise']
        pars['has_task_noise'] = modelPars['has_task_noise']

        pars['applyARD'] = modelPars['applyARD']
        pars['initARDscales'] = modelPars['initARDscales']
        
        pars['validationFrequency'] = modelPars['validationFrequency']
        pars['subValidationFrequency'] = modelPars['subValidationFrequency']
        pars['patienceFactor'] = modelPars['patienceFactor']
        pars['weightDecay'] = modelPars['weightDecay']
        pars['LR'] = modelPars['LR']
        pars['LR_pre'] = modelPars['LR_pre']
        pars['minLR'] = modelPars['minLR']
        pars['lrFactorGate'] = modelPars['lrFactorGate']
        pars['lrFactorGPRmean'] = modelPars['lrFactorGPRmean']
        pars['lrFactorGPRhyperparameters'] = modelPars['lrFactorGPRhyperparameters']
        pars['lrFactorIPlocations'] = modelPars['lrFactorIPlocations']
        pars['minEPOCH'] = modelPars['minEPOCH']
        pars['maxEPOCH'] = modelPars['maxEPOCH']
        
        # expert model hyperpars
        pars['expertModel'] = modelPars['expertModel']
        pars['preTrainEpochsExpert'] = modelPars['preTrainEpochsExpert']

        pars['numIPsExpert'] = modelPars['numIPsExpert']
        pars['batchSizeExpert'] = modelPars['batchSizeExpert']

        pars['initMeanExpert'] = modelPars['initMeanExpert']
        pars['initLambdaExpert'] = modelPars['initLambdaExpert']
        pars['initNoiseLevelExpert'] = modelPars['initNoiseLevelExpert']

        pars['fixedNoiseExpert'] = modelPars['fixedNoiseExpert']
        pars['fixedMeanExpert'] = modelPars['fixedMeanExpert']
        pars['fixedLambdaExpert'] = modelPars['fixedLambdaExpert']
        pars['fixedIPlocationsExpert'] = modelPars['fixedIPlocationsExpert']
        pars['inducingCovarTypeExpert'] = modelPars['inducingCovarTypeExpert']
        
        # gate model hyperpars
        pars['numIPsGate'] = modelPars['numIPsGate']
        pars['batchSizeGate'] = modelPars['batchSizeGate']

        pars['initSigmaGate'] = modelPars['initSigmaGate']
        pars['initLambdaGate'] = modelPars['initLambdaGate']

        pars['fixedSigmaGate'] = modelPars['fixedSigmaGate']
        pars['fixedLambdaGate'] = modelPars['fixedLambdaGate']
        pars['fixedIPlocationsGate'] = modelPars['fixedIPlocationsGate']
        pars['gateOutputType'] = modelPars['gateOutputType']
        pars['inducingCovarTypeGate'] = modelPars['inducingCovarTypeGate']

        # MoE model hyperpars
        pars['fixedExperts'] = modelPars['fixedExperts']
        pars['expertBandwidths'] = modelPars['expertBandwidths']
        pars['smallBandwidthPenalty'] = modelPars['smallBandwidthPenalty']
        pars['expertHardSparsity'] = modelPars['expertHardSparsity']
        pars['sparsifyAfterValiationIterations'] = modelPars['sparsifyAfterValiationIterations']
        pars['initSparsificationAfterValiationIterations'] = modelPars['initSparsificationAfterValiationIterations']
        pars['noisy_gating'] = modelPars['noisy_gating']
        pars['noisy_gating_decaying'] = modelPars['noisy_gating_decaying']
        pars['noisy_gating_b'] = modelPars['noisy_gating_b']
        pars['noisy_gating_w'] = modelPars['noisy_gating_w']
        pars['gate_noise_stdDev'] = modelPars['gate_noise_stdDev']
    
    if not alPars is None:
        pars['noiseFree'] = alPars['noiseFree']
        pars['homoscedastic'] = alPars['homoscedastic']
    
    return pars

class SABER(nn.Module):
    
    # split data if (variationalGP and fixedExperts and self.pars['splitRatio'] > 1)
    # or if (exactGP and self.pars['splitRatio'] > 1)
    def splitTrainingData(self, X):
        if (self.pars['expertModel'] == 'exactGP' or not self.pars['fixedExperts']) and self.pars['splitRatio'] > 1:
            # In this case, we need to need to prevent from overfitting by splitting expert and gate training data
            randInx = np.random.choice(len(X), len(X), replace=False)
            trainSizeExpert = len(X)//self.pars['splitRatio']
            trainSubInxExpert = randInx[:trainSizeExpert]
            trainSubInxGate = randInx[trainSizeExpert:]
            return trainSubInxExpert, trainSubInxGate
        else:
            return None, None
    
    def __init__(self, X, y, iwTrain = None, lossFunction = None, pars = None, xVal = None, yVal = None, iwVal = None, valLoss = None, localNoiseVariance = None):
        
        super(SABER, self).__init__()
        
        if pars is None:
            self.pars = initSABERpars()
        else:
            self.pars = pars
        # in case that we assume no noise, set a dummy noise level to 1e-12
        if self.pars['noiseFree']:
            self.pars['initNoiseLevel'] = 1e-12
        # in case that we assume no noise, keep noise parameter fixed
        self.pars['fixedNoiseExpert'] = self.pars['fixedNoiseExpert'] or self.pars['noiseFree']
        
        # if required, split the training data into expert and gate training subsets
        self.expertTrainingSubInx, self.gateTrainingSubInx = self.splitTrainingData(X)
        
        if self.expertTrainingSubInx is None:
        
            self.yExpert = torch.from_numpy(y)
            if self.pars['doublePrecision']:
                self.yExpert = self.yExpert.double()
            self.xExpert = torch.from_numpy(X).type(self.yExpert.dtype)
            if iwTrain is None:
                iwTrain = np.ones(len(X))
            iwTrain = torch.from_numpy(iwTrain).type(self.yExpert.dtype)
            self.iwTrainExpert = iwTrain * len(iwTrain) / sum(iwTrain).item()
            
            self.yGate = self.yExpert
            self.xGate = self.xExpert
            self.iwTrainGate = self.iwTrainExpert
        else:
            self.yExpert = torch.from_numpy(y[self.expertTrainingSubInx])
            if self.pars['doublePrecision']:
                self.yExpert = self.yExpert.double()
            self.xExpert = torch.from_numpy(X[self.expertTrainingSubInx]).type(self.yExpert.dtype)
            if iwTrain is None:
                iwTrainExpert = np.ones(len(self.expertTrainingSubInx))
            else:
                iwTrainExpert = iwTrain[self.expertTrainingSubInx]
            iwTrainExpert = torch.from_numpy(iwTrainExpert).type(self.yExpert.dtype)
            self.iwTrainExpert = iwTrainExpert * len(iwTrainExpert) / sum(iwTrainExpert).item()
            
            self.yGate = torch.from_numpy(y[self.gateTrainingSubInx]).type(self.yExpert.dtype)
            self.xGate = torch.from_numpy(X[self.gateTrainingSubInx]).type(self.yExpert.dtype)
            if iwTrain is None:
                iwTrainGate = np.ones(len(self.gateTrainingSubInx))
            else:
                iwTrainGate = iwTrain[self.gateTrainingSubInx]
            iwTrainGate = torch.from_numpy(iwTrainGate).type(self.yExpert.dtype)
            self.iwTrainGate = iwTrainGate * len(iwTrainGate) / sum(iwTrainGate).item()
        
        self.lossFunction = lossFunction
        if self.lossFunction is None:
            if self.pars['noiseFree']:
                self.lossFunction = 'mse'
            else:
                self.lossFunction = 'mll'
                
        self.xVal = torch.from_numpy(xVal).type(self.yExpert.dtype)
        self.yVal = torch.from_numpy(yVal).type(self.yExpert.dtype)
        if iwVal is None:
            if not xVal is None:
                iwVal = np.ones(len(xVal))
        iwVal = torch.from_numpy(iwVal).type(self.yExpert.dtype)
        self.iwVal = iwVal * len(iwVal) / sum(iwVal).item()
        
        self.valLoss = valLoss
        if self.valLoss is None:
            if self.pars['noiseFree']:
                self.valLoss = 'mse'
            else:
                self.valLoss = 'mll'
        
        self.localNoiseVariance = localNoiseVariance
        
        self.inputDim = X.shape[1]
        outputDim = 0
        if self.pars['labelModel']:
            outputDim += 1
        if self.pars['derivativeModel']:
            outputDim += self.inputDim
        self.outputDim = outputDim
        self.loss_str = None
        self.val_loss_str = None
            
    def init(self, ipMetaSampler, ipSelectionPars, modelStateDict = None):
        
        nExpert = len(self.xExpert)
        nGate = len(self.xGate)
        
        inducing_point_method_expert = ipMetaSampler(IPstrategy = ipSelectionPars['IPstrategyExpert'], trainingSubInx = self.expertTrainingSubInx)
        inducing_point_method_gate = ipMetaSampler(IPstrategy = ipSelectionPars['IPstrategyGate'], trainingSubInx = self.gateTrainingSubInx)
        
        ############# set up experts ####################
        print('initialize new experts')
        experts = torch.nn.ModuleList()
        #experts = list()
        # check whether we need inducing points; Share them  across the experts, if possible
        inducing_points_expert = None
        if modelStateDict is None and self.pars['expertModel'] == 'variationalGP' or (not self.pars['numIPsExpert'] is None and self.pars['numIPsExpert'] < nExpert):
            if type(inducing_point_method_expert).__name__ != 'GFF_inducing_points':
                inducing_points_expert = torch.from_numpy(inducing_point_method_expert(self.pars['numIPsExpert'])).type(self.yExpert.dtype)

        for l,sigma in enumerate(self.pars['expertBandwidths']):
            print('expert', l)

            if not modelStateDict is None:
                # if we got a state_dict to load, prepare some fitting dummy IPs here
                if self.pars['expertModel'] == 'excatGP':
                    for l in range(len(self.pars['expertBandwidths'])):
                        if 'model.experts.'+str(l)+'.model.covar_module.inducing_points' in modelStateDict:
                            inducing_points_expert = torch.zeros_like(modelStateDict['model.experts.'+str(l)+'.model.covar_module.inducing_points'])
                        else:
                            inducing_points_expert = None
                if self.pars['expertModel'] == 'variationalGP':
                    for l in range(len(self.pars['expertBandwidths'])):
                        inducing_points_expert = torch.zeros_like(modelStateDict['model.experts.'+str(l)+'.model.variational_strategy.inducing_points'])

            # arrange params of each global GPR model in the MoE
            modelPars = copy(self.pars)
            modelPars['globalModel'] = self.pars['expertModel']
            modelPars['inducingCovarTypeGlobal'] = self.pars['inducingCovarTypeExpert']
            modelPars['numIPsGlobal'] = self.pars['numIPsExpert']

            modelPars['initMeanGlobal'] = self.pars['initMeanExpert']
            modelPars['initSigmaGlobal'] = sigma
            modelPars['initLambdaGlobal'] = self.pars['initLambdaExpert']
            modelPars['initNoiseLevelGlobal'] = self.pars['initNoiseLevelExpert']

            modelPars['fixedNoiseGlobal'] = True
            modelPars['fixedMeanGlobal'] = True
            modelPars['fixedLambdaGlobal'] = True
            modelPars['fixedSigmaGlobal'] = True
            modelPars['fixedIPlocationsGlobal'] = True

            modelPars['smallARDbandwidthPenalty'] = 0.

            modelPars['batchSizeGlobal'] = self.pars['batchSizeExpert']
            modelPars['preTrainEpochsGlobal'] = self.pars['preTrainEpochsExpert']
            expertPars = initGPRpars(modelPars, self.pars)

            expert = globalGPR(self.xExpert, self.yExpert, iwTrain = self.iwTrainExpert, lossFunction = self.lossFunction, pars = expertPars, xVal = self.xVal, yVal = self.yVal, iwVal = self.iwVal, valLoss = self.valLoss, localNoiseVariance = self.localNoiseVariance, labelModel = self.pars['labelModel'], derivativeModel = self.pars['derivativeModel'])

            expert.init(inducing_points = inducing_points_expert, inducing_point_method = inducing_point_method_expert)
            if modelStateDict is None:
                expert.preTrainModel()
            experts.append(expert)

        # we want to share the mean, regularization lambda and the likelihood model across all experts from this point
        likelihood = experts[0].model.likelihood
        mean_module = experts[0].model.mean_module
        if hasattr(experts[0].model.covar_module, 'raw_outputscale'):
            lambdaParam = experts[0].model.covar_module.raw_outputscale
        else:
            lambdaParam = experts[0].model.covar_module.base_kernel.raw_outputscale

        for expert in experts[1:]:
            expert.model.likelihood = likelihood
            expert.model.mean_module = mean_module
            if hasattr(expert.model.covar_module, 'raw_outputscale'):
                expert.model.covar_module.raw_outputscale = lambdaParam
            else:
                expert.model.covar_module.base_kernel.raw_outputscale = lambdaParam

        ############# set up gate #######################
        # we need gate IPs in any case
        if modelStateDict is None:
            print('find gate inducing points')
            inducing_points_gate = inducing_point_method_gate(self.pars['numIPsGate'])

            inducing_points_gate = torch.from_numpy(inducing_points_gate[None].repeat(len(experts),axis=0)).type(self.yExpert.dtype)
        else:
            inducing_points_gate = torch.zeros_like(modelStateDict['model.gate.variational_strategy.base_variational_strategy.inducing_points'])

        gate = MultitaskApproxGPModel(inducing_points_gate, len(experts), initialBandwidth = self.pars['initSigmaGate'], initialLambda = self.pars['initLambdaGate'], fixedInducingPoints = self.pars['fixedIPlocationsGate'], inducingCovarType = self.pars['inducingCovarTypeGate'], outputType = self.pars['gateOutputType'])
        if self.pars['doublePrecision']:
            gate = gate.double()

        ############# set up MoE #####################
        self.model = MoE(self.inputDim, experts, gate, likelihood, noisy_gating=self.pars['noisy_gating'], initialNoiseStdDev=self.pars['gate_noise_stdDev'])
        
        if not modelStateDict is None:
            print('load model from stateDict')
            self.load_state_dict(modelStateDict)
            self.model.sparsityK = min(self.pars['expertHardSparsity'], len(experts))
            
            # in this case, if there is subsequent retraining, we need to ensure no early convergence due to the fact that we start close to a local minimum already
            self.pars['minEPOCH'] = int(5*self.pars['validationFrequency'])
        
        if self.pars['doublePrecision']:
            self.model = self.model.double()
                
        self.eval()
        
    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)
    
    def trainModel(self, printMethod = None, plotFrequencyFactor = 1):
        
        nExpert = len(self.xExpert)
        nGate = len(self.xGate)
        
        sharedX = self.xGate.shape == self.xExpert.shape and torch.equal(self.xGate,self.xExpert)
        
        sparsificationFrequency = int(np.round(self.pars['sparsifyAfterValiationIterations'] * self.pars['validationFrequency']))
        initSparsification = int(np.round(self.pars['initSparsificationAfterValiationIterations'] * self.pars['validationFrequency']))
        
        finalSparsity = len(self.model.experts) == self.pars['expertHardSparsity']
        finalModel = copy(finalSparsity)
        
        if sparsificationFrequency == 0 and initSparsification == 0:
            self.model.sparsityK = self.pars['expertHardSparsity']
            finalSparsity = True
            
        if self.pars['validationFrequency'] % self.pars['subValidationFrequency'] == 0:
            subValidationFrequency = self.pars['subValidationFrequency']
        else:
            subValidationFrequency = 1
        patience = int(self.pars['patienceFactor'] * self.pars['validationFrequency'] / subValidationFrequency)
        plotFrequency = int(plotFrequencyFactor * self.pars['validationFrequency'])
            
        print('Gate parameters: ', 'batchSize ', str(self.pars['batchSizeGate']), ', inducing points ', str(self.pars['numIPsGate']), ', valFreq ', str(self.pars['validationFrequency']), ', subValFreq ', str(subValidationFrequency), ', patience ', str(patience), ', sigmaPen ', myScientificFormat(self.pars['smallBandwidthPenalty']), ', sparsity K ', str(self.pars['expertHardSparsity']), sep='')
        
        gate = self.model.gate
        experts = self.model.experts
        likelihood = self.model.likelihood
        mean_module = experts[0].model.mean_module
        if hasattr(experts[0].model.covar_module, 'raw_outputscale'):
            lambdaParam = experts[0].model.covar_module.raw_outputscale
        else:
            lambdaParam = experts[0].model.covar_module.base_kernel.raw_outputscale
        
        # what are all the data scenarios?
        # (1) variationalGP --> WeightedPredictiveLogLikelihood or weighted mseLoss
        # (1.a) not fixed --> share xGate and xExpert
        # (1.b) fixed:
        # (1.b,i) self.pars['splitRatio'] == 1: still share xGate and xExpert
        # (1.b,ii) self.pars['splitRatio'] > 1: use separate xGate and xExpert
        
        # (2) exactGP:
        # (2.a) looScenario, iff xGate and xExpert are shared --> WeightedLeaveOneOutPseudoLikelihood or WeightedLeaveOneOutMSE
        # (2.b) for xGate != xExpert --> WeightedExactPredictiveLogLikelihood or weighted mseLoss
        
        looScenario = self.pars['expertModel'] == 'exactGP' and sharedX
        
        if self.lossFunction == 'mll':
            expertMLLs = torch.nn.ModuleList()
            for l,expert in enumerate(experts):
                if expert.pars['model'] == 'exactGP':
                    if looScenario:
                        # expert and gate training sets are shared, estimate leave-one-out predictive log likelihood
                        mll = WeightedLeaveOneOutPseudoLikelihood(likelihood, expert.model)
                    else:
                        # expert and gate training sets are individual, estimate explicit predictive log likelihood
                        mll = WeightedExactPredictiveLogLikelihood(likelihood, expert.model)
                else:
                    mll = WeightedPredictiveLogLikelihood(likelihood, expert.model, num_data=nGate)
                expertMLLs.append(mll)
                
        if self.lossFunction == 'mse':
            mseLoss = torch.nn.MSELoss()
                
        if self.valLoss == 'mll':
            expertValidationMLLs = torch.nn.ModuleList()
            for l,expert in enumerate(experts):
                if expert.pars['model'] == 'exactGP':
                    mll = WeightedExactPredictiveLogLikelihood(likelihood, expert.model)
                else:
                    mll = WeightedPredictiveLogLikelihood(likelihood, expert.model, num_data=nGate)
                expertValidationMLLs.append(mll)
                
        #self.train()
        gate.train()
        if not self.pars['fixedExperts']:
            if self.pars['expertModel'] == 'variationalGP' or looScenario:
                experts.train()
        likelihood.train()
        
        allParams = []
        par_lr_thresholds = []
        weight_decay = self.pars['weightDecay']
        
        # all gate-related parameters
        lrGate = self.pars['LR'] * self.pars['lrFactorGate']
        min_lrGate = self.pars['minLR'] * self.pars['lrFactorGate']

        gatePars = list(gate.variational_strategy.base_variational_strategy._variational_distribution.parameters())
        if self.pars['noisy_gating']:
            if self.pars['noisy_gating_b']:
                gatePars += [self.model.b_noise]
            else:
                self.model.b_noise.requires_grad = False
            if self.pars['noisy_gating_w']:
                gatePars += [self.model.w_noise]
            else:
                self.model.w_noise.requires_grad = False
                
        allParams.append({'params': gatePars, 'lr': lrGate, 'weight_decay': weight_decay})
        par_lr_thresholds.append(min_lrGate)

        gateHyperpars = list(gate.mean_module.parameters())
        if self.pars['fixedSigmaGate']:
            # it is computationally more efficient, iff we do not plan to update a variable, then to put its gradient off
            gate.covar_module.base_kernel.raw_lengthscale.requires_grad=False
        else:
            gateHyperpars += list(gate.covar_module.base_kernel.raw_lengthscale)

        if self.pars['fixedLambdaGate']:
            gate.covar_module.raw_outputscale.requires_grad = False
        else:
            gateHyperpars += list(gate.covar_module.raw_outputscale)

        allParams.append({'params': gateHyperpars, 'lr': lrGate*self.pars['lrFactorGPRhyperparameters'], 'weight_decay': weight_decay})
        par_lr_thresholds.append(min_lrGate*self.pars['lrFactorGPRhyperparameters'])

        if not self.pars['fixedIPlocationsGate']:
            parsGateInducingPoints = [gate.variational_strategy.base_variational_strategy.inducing_points]
            allParams.append({'params': parsGateInducingPoints, 'lr': lrGate*self.pars['lrFactorIPlocations'], 'weight_decay': weight_decay})
            par_lr_thresholds.append(min_lrGate*self.pars['lrFactorIPlocations'])
            
        # all expert-related parameters
        if self.pars['fixedExperts']:
            # put off gradients of all expert parameters
            
            # the expert parameters
            if self.pars['expertModel'] == 'variationalGP':
                for expert in experts:
                    for pp in expert.model.variational_strategy._variational_distribution.parameters():
                        pp.requires_grad = False
                        
            # the expert inducing point positions
            if self.pars['expertModel'] == 'variationalGP' and not self.pars['fixedIPlocationsExpert']:
                for expert in experts:
                    expert.model.variational_strategy.inducing_points.requires_grad = False

            if self.pars['expertModel'] == 'exactGP' and hasattr(experts[0].model.covar_module, 'inducing_points') and not self.pars['fixedIPlocationsExpert']:
                for expert in experts:
                    expert.model.covar_module.inducing_points.requires_grad = False
                    
            # the mean estimate
            if self.pars['labelModel']:
                mean_module.constant.requires_grad = False


            # the noise estimate
            if self.lossFunction == 'mll':
                if self.pars['homoscedastic']:
                    if hasattr(likelihood, 'noise'):
                        likelihood.raw_noise.requires_grad=False
                    if hasattr(likelihood, 'task_noises'):
                        likelihood.raw_task_noises.requires_grad=False

            # the lambda estimate
            lambdaParam.requires_grad = False

            # we typically want all experts to have a fixed bandwidth. Set gradient off, here
            for expert in experts:
                if hasattr(expert.model.covar_module.base_kernel, 'raw_lengthscale'):
                    expert.model.covar_module.base_kernel.raw_lengthscale.requires_grad = False
                else:
                    expert.model.covar_module.base_kernel.base_kernel.raw_lengthscale.requires_grad = False
            
        else:
            # add expert parameters, and hyperparameters to a certain degree

            # the expert parameters
            if self.pars['expertModel'] == 'variationalGP':
                expertPars = []
                for expert in experts:
                    expertPars += list(expert.model.variational_strategy._variational_distribution.parameters())
                allParams.append({'params': expertPars, 'lr': self.pars['LR'], 'weight_decay': weight_decay})
                par_lr_thresholds.append(self.pars['minLR'])

            # the expert inducing point positions
            if self.pars['expertModel'] == 'variationalGP' and not self.pars['fixedIPlocationsExpert']:
                parsExpertInducingPoints = []
                for expert in experts:
                    parsExpertInducingPoints.append(expert.model.variational_strategy.inducing_points)
                allParams.append({'params': parsExpertInducingPoints, 'lr': self.pars['LR']*self.pars['lrFactorIPlocations'], 'weight_decay': weight_decay})
                par_lr_thresholds.append(self.pars['minLR']*self.pars['lrFactorIPlocations'])

            if self.pars['expertModel'] == 'exactGP' and hasattr(experts[0].model.covar_module, 'inducing_points') and not self.pars['fixedIPlocationsExpert']:
                parsExpertInducingPoints = []
                for expert in experts:
                    parsExpertInducingPoints.append(expert.model.covar_module.inducing_points)
                allParams.append({'params': parsExpertInducingPoints, 'lr': self.pars['LR']*self.pars['lrFactorIPlocations'], 'weight_decay': weight_decay})
                par_lr_thresholds.append(self.pars['minLR']*self.pars['lrFactorIPlocations'])

            # the mean estimate
            if self.pars['labelModel']:
                if self.pars['fixedMeanExpert']:
                    mean_module.constant.requires_grad = False
                else:
                    mean_module.constant.requires_grad = True
                    allParams.append({'params': list(mean_module.parameters()), 'lr': self.pars['LR']*self.pars['lrFactorGPRmean'], 'weight_decay': weight_decay})
                    par_lr_thresholds.append(self.pars['minLR']*self.pars['lrFactorGPRmean'])

            # the noise estimate
            if self.lossFunction == 'mll':
                if self.pars['homoscedastic']:
                    if self.pars['fixedNoiseExpert']:
                        if hasattr(likelihood, 'noise'):
                            likelihood.raw_noise.requires_grad=False
                        if hasattr(likelihood, 'task_noises'):
                            likelihood.raw_task_noises.requires_grad=False
                    else:
                        if hasattr(likelihood, 'noise'):
                            likelihood.raw_noise.requires_grad=True
                        if hasattr(likelihood, 'task_noises'):
                            likelihood.raw_task_noises.requires_grad=True
            
                        expertHyperparNoise = list(likelihood.parameters())
                        allParams.append({'params': expertHyperparNoise, 'lr': self.pars['LR']*self.pars['lrFactorGPRhyperparameters'], 'weight_decay': weight_decay})
                        par_lr_thresholds.append(self.pars['minLR']*self.pars['lrFactorGPRhyperparameters'])

            # the lambda estimate
            if self.pars['fixedLambdaExpert']:
                lambdaParam.requires_grad = False
            else:
                lambdaParam.requires_grad = True
                allParams.append({'params': lambdaParam, 'lr': self.pars['LR']*self.pars['lrFactorGPRhyperparameters'], 'weight_decay': weight_decay})
                par_lr_thresholds.append(self.pars['minLR']*self.pars['lrFactorGPRhyperparameters'])

            # we typically want all experts to have a fixed bandwidth. Set gradient off, here
            for expert in experts:
                if hasattr(expert.model.covar_module.base_kernel, 'raw_lengthscale'):
                    expert.model.covar_module.base_kernel.raw_lengthscale.requires_grad = False
                else:
                    expert.model.covar_module.base_kernel.base_kernel.raw_lengthscale.requires_grad = False

        optimizer = torch.optim.Adam(allParams)
        if finalSparsity:
            scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience = patience, min_lr=par_lr_thresholds)

            
        if self.pars['fixedExperts']:
            # in this case, we can pre-calculate the expert responses once, as they do not change while updating
            allXGateDistribution = []
            print('estimate precalc preds')
            for expert in experts:
                if not looScenario:
                    gatePredDist = expert(self.xGate, predictionsOnly = (self.lossFunction == 'mse'), jitter = self.pars['jitter'], cholJitter = self.pars['cholJitter'])
                else:
                    trainingLOOPredDist = expert.model.looPredictiveDistribution()
                    gatePredDist = MultivariateNormal(trainingLOOPredDist.mean.detach(), trainingLOOPredDist.covariance_matrix.detach())
                allXGateDistribution.append(gatePredDist)
        else:
            allXGateDistribution = None
            
        if self.pars['fixedExperts']:
            # in this case, we can pre-calculate the expert responses once, as they do not change while updating
            allXValDistribution = []
            print('estimate precalc val preds')
            experts.eval()
            likelihood.eval()
            for expert in experts:
                with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False), gpytorch.settings.fast_pred_var(True):
                    gatePredDist = expert(self.xVal, predictionsOnly = (self.lossFunction == 'mse'), jitter = self.pars['jitter'], cholJitter = self.pars['cholJitter'])
                allXValDistribution.append(gatePredDist)
            #experts.train()
            #likelihood.train()
        else:
            allXValDistribution = None
            
        if allXValDistribution is None:
            val_yPred = None
            val_covar = None
        else:
            val_yPred = []
            if self.lossFunction == 'mse':
                val_covar = None
            else:
                val_covar = []
            for expDist in allXValDistribution:
                val_yPred.append(expDist.mean)
                if not self.lossFunction == 'mse':
                    val_covar.append(expDist.covariance_matrix)

        torch_dataset = Data.TensorDataset(Variable(self.xGate), Variable(self.yGate), self.iwTrainGate, torch.arange(nGate))

        loader = Data.DataLoader(dataset=torch_dataset, batch_size=self.pars['batchSizeGate'], shuffle=True, num_workers=0,)
        print('start MoE training')
        epoch = 0
        while epoch < self.pars['maxEPOCH']:
            if (epoch+1) % plotFrequency == 0:
                if not printMethod is None:
                    printMethod(self)
                    
            if not finalSparsity and epoch >= initSparsification and sparsificationFrequency == 0:
                finalSparsity = True
                self.model.sparsityK = self.pars['expertHardSparsity']

            if not finalSparsity and epoch >= initSparsification and (epoch+1) % sparsificationFrequency == 0:
                self.model.sparsityK -= 1
                finalSparsity = self.model.sparsityK == self.pars['expertHardSparsity']
                
            if not finalModel and finalSparsity:
                # switch to final model, and reset the optimizer
                finalModel = True
                print('finalModel reached')
                scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience = patience, min_lr=par_lr_thresholds)
                
            for step, (batch_x, batch_y, batch_IW, batch_inx) in enumerate(loader): # for each training step
                b_x = Variable(batch_x)
                b_y = Variable(batch_y)
                N_batch = len(batch_y)
                w_B = sum(batch_IW).item()

                all_w_b = batch_IW * N_batch / w_B

                if not self.pars['fixedExperts'] and self.pars['expertModel'] == 'exactGP':
                    for expret in experts:
                        expert.model._clear_cache()
                    if looScenario:
                        allXGateDistribution = []
                        for expert in experts:
                            if not looScenario:
                                gatePredDist = expert(self.xGate, predictionsOnly = (self.lossFunction == 'mse'), jitter = self.pars['jitter'], cholJitter = self.pars['cholJitter'])
                            else:
                                trainingLOOPredDist = expert.model.looPredictiveDistribution()
                                gatePredDist = MultivariateNormal(trainingLOOPredDist.mean, trainingLOOPredDist.covariance_matrix)
                            allXGateDistribution.append(gatePredDist)

                if allXGateDistribution is None:
                    batch_yPred = None
                    batch_covar = None
                else:
                    batch_yPred = []
                    if self.lossFunction == 'mse':
                        batch_covar = None
                    else:
                        batch_covar = []
                    for expDist in allXGateDistribution:
                        batch_yPred.append(expDist.mean[batch_inx])
                        if not self.lossFunction == 'mse':
                            batch_covar.append(expDist.covariance_matrix[batch_inx][:,batch_inx])

                if self.lossFunction == 'mll':

                    #if looScenario:
                    #    # cases: If we have precalculated LOO responses, we do not need to recombine new estimates of LOO again
                    #    if self.pars['fixedExperts']:
                    #        predictions, expert_output_distributions, gateValues, inxes, smallBandwidth_loss = self(b_x, iw = all_w_b, combine = False, precalcExpertY = batch_yPred, precalcExpertCovar = batch_covar)
                    #    else:
                    #        raise NotImplementedError
                    #        _, _, gateValues, inxes, smallBandwidth_loss = self(b_x, iw = all_w_b, predictionsOnly = True, combine = False, precalcExpertY = batch_yPred, precalcExpertCovar = batch_covar, returnLabelPredictions = False)
                    #        # recalculate LOO expert_output_distributions and combine
                    #else: 
                    #    predictions, expert_output_distributions, gateValues, inxes, smallBandwidth_loss = self(b_x, iw = all_w_b, combine = False, precalcExpertY = batch_yPred, precalcExpertCovar = batch_covar)
                    predictions, expert_output_distributions, gateValues, inxes, smallBandwidth_loss = self(b_x, iw = all_w_b, combine = False, precalcExpertY = batch_yPred, precalcExpertCovar = batch_covar)
                    loss = 0.
                    # merge each experts weighted standard-MLL loss
                    for i in range(self.model.num_experts):
                        if len(inxes[i]) == 0:
                            # no data assigned to this expert, delete it from loss
                            continue
                        # gateValues tell what expert takes which share of each sample in the training batch. However, the importance of the samples are not treated
                        # in order to treat importance, we need to multiply the gateValues with the importance weights
                        all_positive_v_l_b = gateValues[i].squeeze() * all_w_b[inxes[i]]
                        # hack weights: Since gpytorch divides the log-likelihood-term by len(all_positive_v_l_b) rather than w_B, we adjust for this here to achieve the desired outcome 
                        weights = all_positive_v_l_b * len(all_positive_v_l_b) / w_B
                        
                        if looScenario:
                            if self.pars['homoscedastic']:
                                if batch_yPred is None:
                                    loss -= expertMLLs[i](b_y[inxes[i]].squeeze(-1), weights = weights, subInx = batch_inx[inxes[i]])
                                else:
                                    loss -= expertMLLs[i](b_y[inxes[i]].squeeze(-1), mu = batch_yPred[i][inxes[i]], sigma2 = batch_covar[i].diag()[inxes[i]], weights = weights)
                            else:
                                localBatchNoise = self.localNoiseVariance(b_x)
                                if batch_yPred is None:
                                    loss -= expertMLLs[i](b_y[inxes[i]].squeeze(-1), weights = weights, subInx = batch_inx[inxes[i]], noise = localBatchNoise[inxes[i]])
                                else:
                                    loss -= expertMLLs[i](b_y[inxes[i]].squeeze(-1), mu = batch_yPred[i][inxes[i]], sigma2 = batch_covar[i].diag()[inxes[i]], weights = weights, noise = localBatchNoise[inxes[i]])
                        else:
                            v_B_l = all_positive_v_l_b.sum().item()
                            # hack num_data for correct weighting of the KL-term
                            n_l = nGate * w_B / v_B_l
                            expertMLLs[i].num_data = n_l
                            if self.pars['homoscedastic']:
                                loss -= expertMLLs[i](expert_output_distributions[i], b_y[inxes[i]].squeeze(-1), weights = weights)
                            else:
                                localBatchNoise = self.localNoiseVariance(b_x)
                                loss -= expertMLLs[i](expert_output_distributions[i], b_y[inxes[i]].squeeze(-1), weights = weights, noise = localBatchNoise[inxes[i]])

                if self.lossFunction == 'mse':
                    # implement weighted MSE loss
                    root_all_w_b = all_w_b.pow(0.5).reshape(-1,1)
                    if looScenario:
                        # cases: If we have precalculated LOO responses, we do not need to recombine new estimates of LOO again
                        if self.pars['fixedExperts']:
                            predictions, smallBandwidth_loss = self(b_x, iw = all_w_b, predictionsOnly = True, precalcExpertY = batch_yPred, precalcExpertCovar = batch_covar)
                        else:
                            _, _, gateValues, inxes, smallBandwidth_loss = self(b_x, iw = all_w_b, predictionsOnly = True, combine = False, precalcExpertY = batch_yPred, precalcExpertCovar = batch_covar, returnLabelPredictions = False)
                            gateFull = torch.zeros([len(b_x), self.model.num_experts]).type(b_x.dtype)
                            for i in range(self.model.num_experts):
                                if len(inxes[i]) == 0:
                                    continue
                                gateFull[inxes[i],i] = gateValues[i].squeeze()
                            dispatcher = SparseDispatcher(self.model.num_experts, gateFull)
                            # get loo-predictions of each individual expert
                            expert_outputs = []
                            for i,expert in enumerate(experts):
                                if len(inxes[i]) > 0:
                                    if batch_yPred is None:
                                        looExpertPred = expert.model.looPredictiveDistribution().mean[batch_inx[inxes[i]]]
                                    else:
                                        looExpertPred = batch_yPred[i][inxes[i]]
                                    if looExpertPred.dim() == 1:
                                        looExpertPred = looExpertPred.unsqueeze(-1)
                                else:
                                    looExpertPred = torch.zeros([0,self.model.num_experts])
                                expert_outputs.append(looExpertPred)
                            predictions = dispatcher.combine(expert_outputs)
                    else:
                        predictions, smallBandwidth_loss = self(b_x, iw = all_w_b, predictionsOnly = True, precalcExpertY = batch_yPred, precalcExpertCovar = batch_covar)     # input x and predict based on x
                    if predictions.ndim < 2:
                        predictions = predictions.unsqueeze(-1)
                    if self.pars['labelModel']:
                        loss = mseLoss(predictions * root_all_w_b, b_y * root_all_w_b)
                    else:
                        loss = mseLoss(predictions * root_all_w_b, b_y[:,1:] * root_all_w_b)

                objective_loss = loss.item()
                loss_str = myScientificFormat(objective_loss) + ' (objective)'
                allLosses = (objective_loss,)

                # log likelihood scales with n, smallBandwidth_loss and load_loss are free of scaling
                if self.pars['smallBandwidthPenalty'] > 0:
                    loss += self.pars['smallBandwidthPenalty']*smallBandwidth_loss
                    bwLoss = loss.item() - sum(allLosses)
                    allLosses += (bwLoss,)
                    loss_str += ' + ' + myScientificFormat(bwLoss) + ' (bandwidth penalty)'

                loss_str = 'Loss: ' + myScientificFormat(loss) + ' = ' + loss_str
                self.loss_str = loss_str

                optimizer.zero_grad()   # clear gradients for next train
                loss.backward()         # backpropagation, compute gradients
                optimizer.step()        # apply gradients
                
            if self.pars['noisy_gating_decaying'] > 0 and epoch > 0 and ((epoch+1) % self.pars['validationFrequency'] == 0):
                # statically reduce noisy gating here, note that b_noise is given in log-scale
                self.model.b_noise.data -= self.pars['noisy_gating_decaying']*np.log(2.)
                
            if (finalModel and ((epoch+1) % subValidationFrequency == 0)) or ((epoch+1) % self.pars['validationFrequency'] == 0):
                # do validation
                gate.eval()
                if not self.pars['fixedExperts']:
                    experts.eval()
                    likelihood.eval()

                with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False), gpytorch.settings.fast_pred_var(True):
                    ####################################
                    valSetWeight = sum(self.iwVal).item()
                    if self.valLoss == 'mll':
                        valPred, expert_output_distributions, gateValues, inxes = self(self.xVal, combine = False, precalcExpertY = val_yPred, precalcExpertCovar = val_covar)     # input x and predict based on x
                        valLoss = 0.

                        # merge each experts weighted standard-MLL loss
                        for i in range(len(experts)):
                            if expert_output_distributions[i] == []:
                                # no data assigned to this expert, delete it from loss
                                continue
                            # in order to treat importance, we need to multiply the gateValues with the importance weights
                            raw_weights = gateValues[i].squeeze() * self.iwVal[inxes[i]]
                            # we need to adjust raw_weights as follows, because gpytorch divides the weights by its length, rather than its weight sum
                            if len(gateValues[i]) == 1:
                                weights = raw_weights / valSetWeight
                            else:
                                weights = raw_weights * len(raw_weights) / valSetWeight

                            # hack num_data for correct weighting of the KL-term
                            N_expert = len(self.iwVal) * valSetWeight / raw_weights.sum().item()
                            expertValidationMLLs[i].num_data = N_expert
                            if self.pars['homoscedastic']:
                                if self.pars['labelModel']:
                                    valLoss -= expertValidationMLLs[i](expert_output_distributions[i], self.yVal[inxes[i]].squeeze(), weights = weights)
                                else:
                                    valLoss -= expertValidationMLLs[i](expert_output_distributions[i], self.yVal[:,1:][inxes[i]].squeeze(), weights = weights)
                            else:
                                localValNoise = self.localNoiseVariance(self.xVal)
                                if self.pars['labelModel']:
                                    valLoss -= expertValidationMLLs[i](expert_output_distributions[i], self.yVal[inxes[i]].squeeze(), weights = weights, noise = localValNoise[inxes[i]])
                                else:
                                    valLoss -= expertValidationMLLs[i](expert_output_distributions[i], self.yVal[:,1:][inxes[i]].squeeze(), weights = weights, noise = localValNoise[inxes[i]])
                        valLoss = valLoss.item()
                    if self.valLoss == 'mse':
                        weights = self.iwVal * len(self.iwVal) / sum(self.iwVal).item()
                        valPred, = self(self.xVal, predictionsOnly = True, precalcExpertY = val_yPred, precalcExpertCovar = val_covar)     # input x and predict based on x

                        # implement weighted MSE loss
                        rootWeights = weights.pow(0.5).reshape(-1,1)
                        if valPred.ndim < 2:
                            valPred = valPred.unsqueeze(-1)
                        valLoss = mseLoss(valPred * rootWeights, self.yVal * rootWeights).item()**0.5
                    ##########################################

                gate.train()
                if not self.pars['fixedExperts']:
                    experts.train()
                    likelihood.train()

            if finalModel and ((epoch+1) % subValidationFrequency == 0):
                # if validation loss stagnates, reduce learning rate of optimizer
                scheduler.step(valLoss)
            if (epoch+1) % self.pars['validationFrequency'] == 0:
                #clear_output()
                if self.pars['labelModel']:
                    printMean = myScientificFormat(mean_module.constant.item())
                if not self.pars['noiseFree']:
                    if self.pars['homoscedastic']:
                        printNoise = ', noise level ' + myScientificFormat(likelihood.noise.item(), 2)
                    else:
                        printNoise = ', heteroscedastic'
                else:
                    printNoise = ''

                if hasattr(experts[0].model.covar_module, 'raw_outputscale'):
                    printLambda = myScientificFormat(experts[0].model.covar_module.outputscale.item())
                else:
                    printLambda = myScientificFormat(experts[0].model.covar_module.base_kernel.outputscale.item())
                print('Iter '+str(epoch+1), (', expert mean ' + printMean if self.pars['labelModel'] else ''), printNoise, ', lambda ', printLambda, ', gate sigma ', myScientificFormat(gate.covar_module.base_kernel.lengthscale[0].item()), ', lambda ', myScientificFormat(gate.covar_module.outputscale[0].item()), ' noisy gating ', ([myScientificFormat(torch.exp(b).item()) for b in self.model.b_noise[0]] if self.pars['noisy_gating_b'] else myScientificFormat(torch.exp(self.model.b_noise[0,0]).item())), sep='')
                print(loss_str)
                
                if self.valLoss == 'mse':
                    val_loss_str = 'current validation RMSE: ' + str(valLoss)
                if self.valLoss == 'mll':
                    val_loss_str = 'current validation MLL: ' + str(valLoss)
                self.val_loss_str = val_loss_str
                    
                print(val_loss_str)
                print('current learning rate: ', myScientificFormat(optimizer.param_groups[0]['lr']))

            epoch += 1
            if (epoch >= self.pars['maxEPOCH'] or optimizer.param_groups[0]['lr'] == par_lr_thresholds[0]):
                break
            
        self.eval()
        
    def localBandwidthFunction(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).type(self.yExpert.dtype)
        if x.ndim == 3:
            x = x.flatten(-2)
        with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False), gpytorch.settings.fast_pred_var(True):
            _, sigma = self(x, returnLabelPredictions = False, returnSigmaPredictions = True)
        return sigma.detach().numpy()
    
    def localNoiseVarianceEstimate(self, x):
        if self.pars['homoscedastic']:
            return(np.ones(len(x))*self.model.likelihood.noise.item())
        else:
            return(self.localNoiseVariance(x))
    
class MultitaskApproxGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, outputDim, likelihood = None, initialBandwidth = None, initialLambda = None, fixedInducingPoints = False, inducingCovarType = 'scalar', outputType = 'independent', applyARD = False):
    
        self.likelihood = likelihood
        
        inputDim = inducing_points[0,0].numel()
        if applyARD:
            ard_num_dims = inputDim
        else:
            ard_num_dims = None
        
        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        if inducingCovarType == 'full':
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(-2), batch_shape=torch.Size([outputDim]))
        if inducingCovarType == 'diag':
            variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(inducing_points.size(-2), batch_shape=torch.Size([outputDim]))
        if inducingCovarType == 'scalar':
            variational_distribution = gpytorch.variational.DeltaVariationalDistribution(inducing_points.size(-2), batch_shape=torch.Size([outputDim]))
        
        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        if outputType == 'dependent':
            variational_strategy = gpytorch.variational.LMCVariationalStrategy(
                gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=not fixedInducingPoints),
                num_tasks=outputDim,
                num_latents=outputDim,
                latent_dim=-1
            ).type(inducing_points.dtype)
        if outputType == 'independent':
            variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
                 gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=not fixedInducingPoints),
                 num_tasks=outputDim,
             ).type(inducing_points.dtype)
            
        super().__init__(variational_strategy)
    
        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([outputDim])).type(inducing_points.dtype)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims, lengthscale_constraint=Positive(transform=torch.exp, inv_transform=torch.log), batch_shape=torch.Size([outputDim])),
            batch_shape=torch.Size([outputDim]), outputscale_constraint=Positive(transform=torch.exp, inv_transform=torch.log)
        )
        
        if not initialLambda is None:
            self.covar_module._set_outputscale(initialLambda)
        if not initialBandwidth is None:
            self.covar_module.base_kernel._set_lengthscale(initialBandwidth)

    def forward(self, x, y = None, **kwargs):
        if not y is None:
            mean_x = self.mean_module(y)
            covar_x = self.covar_module(x, y)
        else:
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
"""
This module contains all definitions and routines of the core Gaussian process regression model.

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

from linear_operator.operators import DiagLinearOperator, BlockDiagLinearOperator, TriangularLinearOperator

import numpy as np
from .AL_base import myScientificFormat
import functools
import math

def initGPRpars(modelPars = None, alPars = None):
    pars = {}
    pars['doublePrecision'] = False
    pars['kernel'] = 'gaussian'
    pars['model'] = 'exactGP'
    pars['inducingCovarType'] = 'scalar'
    pars['jitter'] = 0.
    pars['cholJitter'] = 1e-16
    pars['numIPs'] = None
    pars['noiseFree'] = False
    pars['homoscedastic'] = True
    pars['has_global_noise'] = True
    pars['has_task_noise'] = False
    
    pars['initMean'] = None
    pars['initSigma'] = 1e0
    pars['initLambda'] = 1e0
    pars['initNoiseLevel'] = 1e0
    
    pars['fixedNoise'] = False
    pars['fixedMean'] = False
    pars['fixedLambda'] = False
    pars['fixedSigma'] = False
    pars['fixedIPlocations'] = True
    
    pars['applyARD'] = False
    pars['initARDscales'] = None
    pars['smallARDbandwidthPenalty'] = 0.
    
    pars['batchSize'] = int(2**6)
    pars['validationFrequency'] = 1
    pars['subValidationFrequency'] = 1
    pars['patienceFactor'] = None
    pars['weightDecay'] = 0.
    pars['LR'] = 1e-1
    pars['LR_pre'] = 1e0
    pars['minLR'] = 1e-3
    pars['lrFactorGPRmean'] = 1.
    pars['lrFactorGPRhyperparameters'] = 2e-1
    pars['lrFactorIPlocations'] = 1e-2
    pars['preTrainEpochs'] = 0
    pars['minEPOCH'] = 0
    pars['maxEPOCH'] = np.Inf
    if not modelPars is None:
        pars['doublePrecision'] = modelPars['doublePrecision']
        pars['kernel'] = modelPars['kernel']
        pars['model'] = modelPars['globalModel']
        pars['inducingCovarType'] = modelPars['inducingCovarTypeGlobal']
        pars['jitter'] = modelPars['jitter']
        pars['cholJitter'] = modelPars['cholJitter']
        pars['numIPs'] = modelPars['numIPsGlobal']
        pars['has_global_noise'] = modelPars['has_global_noise']
        pars['has_task_noise'] = modelPars['has_task_noise']
        
        pars['initMean'] = modelPars['initMeanGlobal']
        pars['initSigma'] = modelPars['initSigmaGlobal']
        pars['initLambda'] = modelPars['initLambdaGlobal']
        pars['initNoiseLevel'] = modelPars['initNoiseLevelGlobal']

        pars['fixedNoise'] = modelPars['fixedNoiseGlobal']
        pars['fixedMean'] = modelPars['fixedMeanGlobal']
        pars['fixedLambda'] = modelPars['fixedLambdaGlobal']
        pars['fixedSigma'] = modelPars['fixedSigmaGlobal']
        pars['fixedIPlocations'] = modelPars['fixedIPlocationsGlobal']

        pars['applyARD'] = modelPars['applyARD']
        pars['initARDscales'] = modelPars['initARDscales']
        pars['smallARDbandwidthPenalty'] = modelPars['smallARDbandwidthPenalty']
        
        pars['batchSize'] = modelPars['batchSizeGlobal']
        pars['validationFrequency'] = modelPars['validationFrequency']
        pars['subValidationFrequency'] = modelPars['subValidationFrequency']
        pars['patienceFactor'] = modelPars['patienceFactor']
        pars['weightDecay'] = modelPars['weightDecay']
        pars['LR'] = modelPars['LR']
        pars['LR_pre'] = modelPars['LR_pre']
        pars['minLR'] = modelPars['minLR']
        pars['lrFactorGPRmean'] = modelPars['lrFactorGPRmean']
        pars['lrFactorGPRhyperparameters'] = modelPars['lrFactorGPRhyperparameters']
        pars['lrFactorIPlocations'] = modelPars['lrFactorIPlocations']
        pars['preTrainEpochs'] = modelPars['preTrainEpochsGlobal']
        pars['minEPOCH'] = modelPars['minEPOCH']
        pars['maxEPOCH'] = modelPars['maxEPOCH']
        
    if not alPars is None:
        pars['noiseFree'] = alPars['noiseFree']
        pars['homoscedastic'] = alPars['homoscedastic']
    return pars

class globalGPR(nn.Module):
    
    def __init__(self, X, y, iwTrain = None, lossFunction = None, pars = None, xVal = None, yVal = None, iwVal = None, valLoss = None, localNoiseVariance = None, labelModel = True, derivativeModel = False):
        
        super(globalGPR, self).__init__()
        
        if pars is None:
            self.pars = initGPRpars()
        else:
            self.pars = pars
        # in case that we assume no noise, set a dummy noise level to 1e-12
        if self.pars['noiseFree']:
            self.pars['initNoiseLevel'] = 1e-12
        # in case that we assume no noise, keep noise parameter fixed
        self.pars['fixedNoise'] = self.pars['fixedNoise'] or self.pars['noiseFree']
        
        self.y = y
        if not torch.is_tensor(self.y):
            self.y = torch.from_numpy(self.y)
            if self.pars['doublePrecision']:
                self.y = self.y.double()
            
        self.X = X
        if not torch.is_tensor(self.X):
            self.X = torch.from_numpy(self.X).type(self.y.dtype)
        if iwTrain is None:
            iwTrain = np.ones(len(X))
        if not torch.is_tensor(iwTrain):
            iwTrain = torch.from_numpy(iwTrain).type(self.y.dtype)
        self.iwTrain = iwTrain * len(iwTrain) / sum(iwTrain).item()
        
        self.lossFunction = lossFunction
        if self.lossFunction is None:
            if self.pars['noiseFree']:
                self.lossFunction = 'mse'
            else:
                self.lossFunction = 'mll'
                
        self.xVal = xVal
        if not self.xVal is None and not torch.is_tensor(self.xVal):
            self.xVal = torch.from_numpy(self.xVal).type(self.y.dtype)
        self.yVal = yVal
        if not self.xVal is None and not torch.is_tensor(self.yVal):
            self.yVal = torch.from_numpy(self.yVal).type(self.y.dtype)
        if iwVal is None:
            if not self.xVal is None:
                iwVal = np.ones(len(xVal))
        if not iwVal is None and not torch.is_tensor(iwVal):
            iwVal = torch.from_numpy(iwVal).type(self.y.dtype)
        if not iwVal is None:
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
        if labelModel:
            outputDim += 1
        if derivativeModel:
            outputDim += self.inputDim
        self.outputDim = outputDim
        self.labelModel = labelModel
        self.derivativeModel = derivativeModel
        self.loss_str = None
        self.val_loss_str = None
            
    def init(self, inducing_points = None, inducing_point_method = None, modelStateDict = None):
        
        n = len(self.X)
        
        # check whether we need inducing points
        if self.pars['model'] == 'variationalGP' or (not self.pars['numIPs'] is None and self.pars['numIPs'] < n):
        
            if not inducing_points is None:
                m = len(inducing_points)
                if self.pars['numIPs'] != m:
                    print(f"The specified number of inducing points {numIPs} does not match the number of specified inducing points {m}!")
                    print(f'Adjusting the specified number of inducing points to {m}')
                    numIPs = m
                    self.pars['numIPs'] = numIPs

            # check if we need to find IPs
            if inducing_points is None:

                print('find inducing points')
                inducing_points = torch.from_numpy(inducing_point_method(self.pars['numIPs'])).type(self.y.dtype)
            
        if self.pars['model'] == 'exactGP':
            self.initExactGP(noiseLevel = self.pars['initNoiseLevel'], lam = self.pars['initLambda'], meanValue = self.pars['initMean'], sig = self.pars['initSigma'], ardScales = self.pars['initARDscales'], inducing_points = inducing_points)
        
        if self.pars['model'] == 'variationalGP':
            self.initVariationalGP(noiseLevel = self.pars['initNoiseLevel'], lam = self.pars['initLambda'], meanValue = self.pars['initMean'], sig = self.pars['initSigma'], ardScales = self.pars['initARDscales'], inducing_points = inducing_points)
        
        if not modelStateDict is None:
            print('load model from stateDict')
            self.load_state_dict(modelStateDict)
            
            # in this case, if there is subsequent retraining, we need to ensure no early convergence due to the fact that we start close to a local minimum already
            self.pars['preTrainEpochs'] = 0
            self.pars['minEPOCH'] = int(5*self.pars['validationFrequency'])
            
        self.model.eval()
    
    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)
        
    def trainModel(self):
        
        if self.pars['model'] == 'exactGP':
            self.trainExactGP()
        if self.pars['model'] == 'variationalGP':
            self.trainVariationalGP()
            
    def preTrainModel(self):
        
        if self.pars['model'] == 'exactGP':
            print('No pre-train needed for exact GP')
        if self.pars['model'] == 'variationalGP':
            self.preTrainVariationalGP()
            
    def initVariationalGP(self, noiseLevel, lam, meanValue, sig = None, ardScales = None, inducing_points = None):
    
        if ardScales is None and sig is None:
            scales = 1.
        elif ardScales is None:
            scales = sig
        elif sig is None:
            scales = ardScales
        else:
            scales = sig * ardScales
        
        print('initialize variational GP'+ (' with '+str(len(inducing_points))+' IPs' if not inducing_points is None else '') +' at bandwidth', (myScientificFormat(sig) if not sig is None else ''), 'and lambda', myScientificFormat(lam))
                
        # whether we assume noisy or noise-free labels, we set up a likelihood here to match the full GP implementation!
        if self.outputDim > 1:
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.outputDim, noise_constraint=Positive(transform=torch.exp, inv_transform=torch.log), has_global_noise=self.pars['has_global_noise'], has_task_noise=self.pars['has_task_noise'])
        else:
            if self.pars['homoscedastic']:
                likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=Positive(transform=torch.exp, inv_transform=torch.log))
            else:
                likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=self.localNoiseVariance(inducing_points))
        if self.pars['doublePrecision']:            
            likelihood = likelihood.double()
            
        has_task_noise = hasattr(likelihood, 'task_noises')
        has_global_noise = hasattr(likelihood, 'noise')
        if self.pars['homoscedastic'] and not noiseLevel is None:
            if self.pars['doublePrecision']:
                if has_global_noise and not has_task_noise:
                    likelihood.noise = torch.tensor(noiseLevel).double()
                if not has_global_noise and has_task_noise:
                    likelihood.task_noises = torch.tensor(noiseLevel).double()
                if has_global_noise and has_task_noise:
                    likelihood.noise = torch.tensor(noiseLevel/2.).double()
                    likelihood.task_noises = torch.tensor(noiseLevel/2.).double()
            else:
                if has_global_noise and not has_task_noise:
                    likelihood.noise = torch.tensor(noiseLevel).float()
                if not has_global_noise and has_task_noise:
                    likelihood.task_noises = torch.tensor(noiseLevel).float()
                if has_global_noise and has_task_noise:
                    likelihood.noise = torch.tensor(noiseLevel/2.).float()
                    likelihood.task_noises = torch.tensor(noiseLevel/2.).float()
                    
        if meanValue is None:
            meanValue = torch.mean(self.y[:,0])
                    
        if self.derivativeModel:
            self.model = derivativeApproxGPModel(inducing_points=inducing_points, likelihood = likelihood, initialBandwidth = scales, initialLambda = lam, fixedInducingPoints = self.pars['fixedIPlocations'], inducingCovarType = self.pars['inducingCovarType'], applyARD = self.pars['applyARD'], initConstant = meanValue, useLabels = self.labelModel)
        else:
            self.model = ApproxGPModel(inducing_points=inducing_points, likelihood = likelihood, initialBandwidth = scales, initialLambda = lam, fixedInducingPoints = self.pars['fixedIPlocations'], inducingCovarType = self.pars['inducingCovarType'], applyARD = self.pars['applyARD'], initConstant = meanValue)
        
        if self.pars['doublePrecision']:
            self.model.covar_module = self.model.covar_module.double()
        
        self.model.eval()
        
    def preTrainVariationalGP(self):
        # keep hyperparameters fixed while only pre-learning a good variational distribution
        
        if self.pars['preTrainEpochs'] == 0:
            print('Warning: No pre-training applied due to specified 0 preTrainEpochs')
            return
        
        # assure that preTrainEpochs is < Inf in this case!
        if self.pars['preTrainEpochs'] == np.Inf:
            defaultNumberEpochs = 100
            print(f"Without validation, the specified number of epochs must be finite!")
            print(f'Reducing number of epochs to {defaultNumberEpochs} instead')
            self.pars['preTrainEpochs'] = defaultNumberEpochs
        
        likelihood = self.model.likelihood

        if self.lossFunction == 'mse':
            mseLoss = torch.nn.MSELoss()
        if self.lossFunction == 'mll':         
            mll = WeightedPredictiveLogLikelihood(likelihood, self.model, num_data=len(self.X))

        print('pre-train a sparse global model wrt. loss: ', self.lossFunction,', batch size: ', self.pars['batchSize'], ', for '+str(self.pars['preTrainEpochs'])+' epochs', sep='')
        self.train()
        likelihood.train()
        
        allParams = []
        par_lr_thresholds = []
        weight_decay = self.pars['weightDecay']
        
        # the expert parameters
        expertPars = list(self.model.variational_strategy._variational_distribution.parameters())
        allParams.append({'params': expertPars, 'lr': self.pars['LR_pre'], 'weight_decay': weight_decay})
        par_lr_thresholds.append(self.pars['LR'])
        
        # for pre-training, deactivate learning of all hyperparameters
        self.model.variational_strategy.inducing_points.requires_grad = False
        
        # the expert hyperparameters
        if self.labelModel:
            self.model.mean_module.constant.requires_grad = False
        
        if self.lossFunction == 'mll':
            if self.pars['homoscedastic']:
                if hasattr(likelihood, 'noise'):
                    likelihood.raw_noise.requires_grad=False
                if hasattr(likelihood, 'task_noises'):
                    likelihood.raw_task_noises.requires_grad=False
        
        # it is computationally more efficient, iff we do not plan to update a variable, then to put its gradient off
        self.model.covar_module.raw_outputscale.requires_grad=False
        self.model.covar_module.base_kernel.raw_lengthscale.requires_grad = False
        
        optimizer = torch.optim.Adam(allParams)
        
        if self.pars['LR_pre'] > self.pars['LR']:
            updateLRafterBatch = True
            if not updateLRafterBatch:
                # blend over from lrPre to lr
                factor = (self.pars['LR']/self.pars['LR_pre'])**(1/pars['preTrainEpochs'])
                scheduler = ReduceLROnPlateau(optimizer, factor=factor, patience = 0, min_lr=par_lr_thresholds)
                scheduler.step(1)
            # we could also do a step after each batch instead of after epoch!
            else:
                # blend over from lrPre to lr
                totalBatchEvals = self.pars['preTrainEpochs'] * int(np.ceil(len(self.X) / self.pars['batchSize']))
                factor = (self.pars['LR']/self.pars['LR_pre'])**(1/totalBatchEvals)
                scheduler = ReduceLROnPlateau(optimizer, factor=factor, patience = 0, min_lr=par_lr_thresholds)
                scheduler.step(1)

        if self.derivativeModel and not self.labelModel:
            torch_dataset = Data.TensorDataset(Variable(self.X), Variable(self.y[:,1:]), self.iwTrain)
        else:     
            torch_dataset = Data.TensorDataset(Variable(self.X), Variable(self.y), self.iwTrain)
            
        loader = Data.DataLoader(dataset=torch_dataset, batch_size=self.pars['batchSize'], shuffle=True, num_workers=0,)

        epoch = 0
        while epoch < self.pars['preTrainEpochs']:
        
            for step, (batch_x, batch_y, batch_IW) in enumerate(loader): # for each training step
            
                b_x = Variable(batch_x)
                b_y = Variable(batch_y)
                # renormalize batch weights
                weights = batch_IW * len(batch_IW) / sum(batch_IW).item()

                prediction = self(b_x, predictionsOnly = True, jitter = self.pars['jitter'], cholJitter = self.pars['cholJitter'])     # input x and predict based on x
                
                # implement weighted loss
                if self.lossFunction == 'mse':
                    rootWeights = weights.pow(0.5).reshape(-1,1)
                    preds = prediction.mean
                    if preds.ndim < 2:
                        preds = preds.unsqueeze(-1)
                    loss = mseLoss(preds * rootWeights, b_y * rootWeights)
                if self.lossFunction == 'mll':
                    if self.pars['homoscedastic']:
                        loss = -mll(prediction, b_y.squeeze(), weights = weights)
                    else:
                        localBatchNoise = self.localNoiseVariance(b_x)
                        loss = -mll(prediction, b_y.squeeze(), weights = weights, noise = localBatchNoise)     # must be (1. nn output, 2. target)
                
                objective_loss = loss.item()
                loss_str = myScientificFormat(objective_loss) + ' (objective)'
                allLosses = (objective_loss,)
                
                if not self.pars['fixedSigma'] and self.pars['applyARD'] and self.pars['smallARDbandwidthPenalty'] > 0:
                    smallARDbandwidth_loss = - self.model.covar_module.base_kernel.raw_lengthscale.mean()
                    
                    loss += self.pars['smallARDbandwidthPenalty']*smallARDbandwidth_loss
                    bwLoss = loss.item() - sum(allLosses)
                    allLosses += (bwLoss,)
                    loss_str += ' + ' + myScientificFormat(bwLoss) + ' (small ARD bandwidth penalty)'
                    
                if not self.pars['fixedSigma'] and self.pars['applyARD'] and self.pars['anisotropicBandwidthPenalty'] > 0:
                    anisotropicBandwidth_loss = self.model.covar_module.base_kernel.lengthscale.var() / (self.model.covar_module.base_kernel.lengthscale.mean()**2 + 1e-10)
                
                    loss += self.pars['anisotropicBandwidthPenalty']*anisotropicBandwidth_loss
                    bwLoss = loss.item() - sum(allLosses)
                    allLosses += (bwLoss,)
                    loss_str += ' + ' + myScientificFormat(bwLoss) + ' (anisotropic bandwidth penalty)'

                allLosses = (loss,) + allLosses
                loss_str = 'Loss: ' + myScientificFormat(loss) + ' = ' + loss_str
                self.loss_str = loss_str

                optimizer.zero_grad()   # clear gradients for next train
                loss.backward()         # backpropagation, compute gradients
                optimizer.step()        # apply gradients
                if updateLRafterBatch:
                    # shrink lr statically
                    if self.pars['LR_pre'] > self.pars['LR']:
                        scheduler.step(1)
                
            if not updateLRafterBatch:
                # shrink lr statically
                if self.pars['LR_pre'] > self.pars['LR']:
                    scheduler.step(1)
                
            # print some variables
            if (epoch+1) % self.pars['validationFrequency'] == 0:
                
                printString = 'Iter '+ str(epoch+1)
                if self.labelModel:
                    printMean = myScientificFormat(self.model.mean_module.constant.item())
                    printString += ', mean ' + str(printMean)
                if self.lossFunction == 'mll':
                    if self.pars['homoscedastic']:
                        if hasattr(likelihood, 'noise'):
                            printString += ', global noise ' + myScientificFormat(likelihood.noise.item(), 2)
                        if hasattr(likelihood, 'task_noises'):
                            printString += ', task noise ' + ' '.join([myScientificFormat(tn, 2) for tn in likelihood.task_noises.detach().numpy()])
                    else:
                        printString += ', heteroscedastic'
                
                printLambda = myScientificFormat(self.model.covar_module.outputscale.item())
                  
                printString += ', lambda ' + str(printLambda)
                
                if self.pars['applyARD']:
                    printSigma = [myScientificFormat(sig.item()) for sig in self.model.covar_module.base_kernel.lengthscale[0]] # lengthscale is 1xd
                else:
                    printSigma = myScientificFormat(self.model.covar_module.base_kernel.lengthscale.item())
                printString += ', sigma ' + str(printSigma)

                print(printString)
                print(loss_str)
                
            epoch += 1
            #if self.pars['LR_pre'] > self.pars['LR'] and optimizer.param_groups[0]['lr'] == par_lr_thresholds[0]:
            #    break
        self.eval()
    
    def trainVariationalGP(self):
    
        likelihood = self.model.likelihood

        if self.lossFunction == 'mse':
            mseLoss = torch.nn.MSELoss()
        if self.lossFunction == 'mll':         
            mll = WeightedPredictiveLogLikelihood(likelihood, self.model, num_data=len(self.X))
            
        if self.valLoss == 'mll':
            valMll = WeightedPredictiveLogLikelihood(likelihood, self.model, num_data=len(self.X))

        print('train a sparse global model wrt. loss: ', self.lossFunction,', batch size: ', self.pars['batchSize'], sep='')
        self.train()
        likelihood.train()
        
        allParams = []
        par_lr_thresholds = []
        weight_decay = self.pars['weightDecay']
        
        # the expert parameters
        expertPars = list(self.model.variational_strategy._variational_distribution.parameters())
        allParams.append({'params': expertPars, 'lr': self.pars['LR'], 'weight_decay': weight_decay})
        par_lr_thresholds.append(self.pars['minLR'])
        
        # the expert inducing point positions
        if not self.pars['fixedIPlocations']:
            self.model.variational_strategy.inducing_points.requires_grad = True
            parsExpertInducingPoints = [self.model.variational_strategy.inducing_points]
            allParams.append({'params': parsExpertInducingPoints, 'lr': self.pars['LR']*self.pars['lrFactorIPlocations'], 'weight_decay': weight_decay})
            par_lr_thresholds.append(self.pars['minLR']*self.pars['lrFactorIPlocations'])
        
        # the expert hyperparameters
        if self.labelModel:
            if self.pars['fixedMean']:
                self.model.mean_module.constant.requires_grad = False
            else:
                self.model.mean_module.constant.requires_grad = True
                allParams.append({'params': list(self.model.mean_module.parameters()), 'lr': self.pars['LR']*self.pars['lrFactorGPRmean'], 'weight_decay': weight_decay})
                par_lr_thresholds.append(self.pars['minLR'] * self.pars['lrFactorGPRmean'])
        
        if self.lossFunction == 'mll':
            if self.pars['homoscedastic']:
                if self.pars['fixedNoise']:
                    if hasattr(likelihood, 'noise'):
                        likelihood.raw_noise.requires_grad=False
                    if hasattr(likelihood, 'task_noises'):
                        likelihood.raw_task_noises.requires_grad=False
                else:
                    if hasattr(likelihood, 'noise'):
                        likelihood.raw_noise.requires_grad=True
                    if hasattr(likelihood, 'task_noises'):
                        likelihood.raw_task_noises.requires_grad=True
                    allParams.append({'params': list(likelihood.parameters()), 'lr': self.pars['LR']*self.pars['lrFactorGPRhyperparameters'], 'weight_decay': weight_decay})
                    par_lr_thresholds.append(self.pars['minLR'] * self.pars['lrFactorGPRhyperparameters'])
        
        if self.pars['fixedLambda']:
            # it is computationally more efficient, iff we do not plan to update a variable, then to put its gradient off
            self.model.covar_module.raw_outputscale.requires_grad=False
        else:
            self.model.covar_module.raw_outputscale.requires_grad=True
            allParams.append({'params': self.model.covar_module.raw_outputscale, 'lr': self.pars['LR']*self.pars['lrFactorGPRhyperparameters'], 'weight_decay': weight_decay})
            par_lr_thresholds.append(self.pars['minLR'] * self.pars['lrFactorGPRhyperparameters'])
            
        if self.pars['fixedSigma']:
            # it is computationally more efficient, iff we do not plan to update a variable, then to put its gradient off
            self.model.covar_module.base_kernel.raw_lengthscale.requires_grad = False
        else:
            self.model.covar_module.base_kernel.raw_lengthscale.requires_grad = True
            allParams.append({'params': self.model.covar_module.base_kernel.raw_lengthscale, 'lr': self.pars['LR']*self.pars['lrFactorGPRhyperparameters'], 'weight_decay': weight_decay})
            par_lr_thresholds.append(self.pars['minLR'] * self.pars['lrFactorGPRhyperparameters'])
        
        optimizer = torch.optim.Adam(allParams)
        
        if self.pars['validationFrequency'] % self.pars['subValidationFrequency'] == 0:
            subValidationFrequency = self.pars['subValidationFrequency']
        else:
            subValidationFrequency = 1
        patience = int(self.pars['patienceFactor'] * self.pars['validationFrequency'] / subValidationFrequency)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience = patience, min_lr=par_lr_thresholds)

        if self.derivativeModel and not self.labelModel:
            torch_dataset = Data.TensorDataset(Variable(self.X), Variable(self.y[:,1:]), self.iwTrain)
        else:     
            torch_dataset = Data.TensorDataset(Variable(self.X), Variable(self.y), self.iwTrain)
            
        loader = Data.DataLoader(dataset=torch_dataset, batch_size=self.pars['batchSize'], shuffle=True, num_workers=0,)

        epoch = 0
        while epoch < self.pars['maxEPOCH']:
        
            for step, (batch_x, batch_y, batch_IW) in enumerate(loader): # for each training step
            
                b_x = Variable(batch_x)
                b_y = Variable(batch_y)
                # renormalize batch weights
                weights = batch_IW * len(batch_IW) / sum(batch_IW).item()

                prediction = self(b_x, predictionsOnly = True, jitter = self.pars['jitter'], cholJitter = self.pars['cholJitter'])     # input x and predict based on x
                
                # implement weighted loss
                if self.lossFunction == 'mse':
                    rootWeights = weights.pow(0.5).reshape(-1,1)
                    preds = prediction.mean
                    if preds.ndim < 2:
                        preds = preds.unsqueeze(-1)
                    loss = mseLoss(preds * rootWeights, b_y * rootWeights)
                if self.lossFunction == 'mll':
                    if self.pars['homoscedastic']:
                        loss = -mll(prediction, b_y.squeeze(), weights = weights)
                    else:
                        localBatchNoise = self.localNoiseVariance(b_x)
                        loss = -mll(prediction, b_y.squeeze(), weights = weights, noise = localBatchNoise)     # must be (1. nn output, 2. target)
                
                objective_loss = loss.item()
                loss_str = myScientificFormat(objective_loss) + ' (objective)'
                allLosses = (objective_loss,)
                
                if not self.pars['fixedSigma'] and self.pars['applyARD'] and self.pars['smallARDbandwidthPenalty'] > 0:
                    smallARDbandwidth_loss = - self.model.covar_module.base_kernel.raw_lengthscale.mean()
                    
                    loss += self.pars['smallARDbandwidthPenalty']*smallARDbandwidth_loss
                    bwLoss = loss.item() - sum(allLosses)
                    allLosses += (bwLoss,)
                    loss_str += ' + ' + myScientificFormat(bwLoss) + ' (small ARD bandwidth penalty)'
                    
                if not self.pars['fixedSigma'] and self.pars['applyARD'] and self.pars['anisotropicBandwidthPenalty'] > 0:
                    anisotropicBandwidth_loss = self.model.covar_module.base_kernel.lengthscale.var() / (self.model.covar_module.base_kernel.lengthscale.mean()**2 + 1e-10)
                
                    loss += self.pars['anisotropicBandwidthPenalty']*anisotropicBandwidth_loss
                    bwLoss = loss.item() - sum(allLosses)
                    allLosses += (bwLoss,)
                    loss_str += ' + ' + myScientificFormat(bwLoss) + ' (anisotropic bandwidth penalty)'

                allLosses = (loss,) + allLosses
                loss_str = 'Loss: ' + myScientificFormat(loss) + ' = ' + loss_str
                self.loss_str = loss_str

                optimizer.zero_grad()   # clear gradients for next train
                loss.backward()         # backpropagation, compute gradients
                optimizer.step()        # apply gradients
                
            if ((epoch+1) % subValidationFrequency == 0) or ((epoch+1) % self.pars['validationFrequency'] == 0):
                # do validation
                self.eval()
                likelihood.eval()

                if self.valLoss == 'mll':

                    with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False, solves = False), gpytorch.settings.fast_pred_var(True):
                        valPred = self(self.xVal, jitter = self.pars['jitter'], cholJitter = self.pars['cholJitter'])

                    if self.pars['homoscedastic']:
                        if self.labelModel:
                            valLoss = -valMll(valPred, self.yVal.squeeze(), weights = self.iwVal)
                        else:
                            valLoss = -valMll(valPred, self.yVal[:,1:].squeeze(), weights = self.iwVal)
                    else:
                        localValNoise = self.localNoiseVariance(self.xVal)
                        if self.labelModel:
                            valLoss = -valMll(valPred, self.yVal.squeeze(), weights = self.iwVal, noise = localValNoise)
                        else:
                            valLoss = -valMll(valPred, self.yVal[:,1:].squeeze(), weights = self.iwVal, noise = localValNoise)

                if self.valLoss == 'mse':
                    with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False, solves = False), gpytorch.settings.fast_pred_var(True):
                        valPred = self(self.xVal, predictionsOnly = True, labelsOnly = False, jitter = self.pars['jitter'], cholJitter = self.pars['cholJitter']).mean

                    if valPred.ndim < 2:
                        valPred = valPred.unsqueeze(-1)
                        
                    if self.labelModel:
                        valLoss = torch.mean((valPred - self.yVal)**2 * self.iwVal.reshape(-1,1))**0.5
                        if self.derivativeModel:
                            valRMSElabel = torch.mean((valPred[:,0] - self.yVal[:,0])**2 * self.iwVal)**0.5
                            valRMSEderivative = torch.mean((valPred[:,1:] - self.yVal[:,1:])**2 * self.iwVal.reshape(-1,1))**0.5
                    else:
                        valLoss = torch.mean((valPred - self.yVal[:,1:])**2 * self.iwVal.reshape(-1,1))**0.5
                        
                valLoss = valLoss.item()
                
                self.train()
                likelihood.train()

            if epoch >= self.pars['minEPOCH'] and (epoch+1) % subValidationFrequency == 0:
                # if validation loss stagnates, reduce learning rate of optimizer
                scheduler.step(valLoss)
                
            # print some variables
            if (epoch+1) % self.pars['validationFrequency'] == 0:
                
                printString = 'Iter '+ str(epoch+1)
                if self.labelModel:
                    printMean = myScientificFormat(self.model.mean_module.constant.item())
                    printString += ', mean ' + str(printMean)
                if self.lossFunction == 'mll':
                    if self.pars['homoscedastic']:
                        if hasattr(likelihood, 'noise'):
                            printString += ', global noise ' + myScientificFormat(likelihood.noise.item(), 2)
                        if hasattr(likelihood, 'task_noises'):
                            printString += ', task noise ' + ' '.join([myScientificFormat(tn, 2) for tn in likelihood.task_noises.detach().numpy()])
                    else:
                        printString += ', heteroscedastic'
                
                printLambda = myScientificFormat(self.model.covar_module.outputscale.item())
                printString += ', lambda ' + str(printLambda)
                
                if self.pars['applyARD']:
                    printSigma = [myScientificFormat(sig.item()) for sig in self.model.covar_module.base_kernel.lengthscale[0]] # lengthscale is 1xd
                else:
                    printSigma = myScientificFormat(self.model.covar_module.base_kernel.lengthscale.item())

                printString += ', sigma ' + str(printSigma)

                print(printString)
                print(loss_str)
                    
                if self.valLoss == 'mse':
                    if self.labelModel and self.derivativeModel:
                        val_loss_str = 'current validation RMSE total: ' + str(valLoss) + ', label: ' + str(valRMSElabel) + ', derivatives: ' + str(valRMSEderivative)
                    else:
                        val_loss_str = 'current validation RMSE: ' + str(valLoss)
                if self.valLoss == 'mll':
                    val_loss_str = 'current validation MLL: ' + str(valLoss)
                self.val_loss_str = val_loss_str
                
                print(val_loss_str)
                print('current learning rate: ', myScientificFormat(optimizer.param_groups[0]['lr']))
                
            epoch += 1
            if self.pars['LR'] > self.pars['minLR'] and optimizer.param_groups[0]['lr'] == par_lr_thresholds[0]:
                break
        self.eval()
        
    def initExactGP(self, noiseLevel, lam, meanValue, sig = None, ardScales = None, inducing_points = None):
    
        if ardScales is None and sig is None:
            scales = 1.
        elif ardScales is None:
            scales = sig
        elif sig is None:
            scales = ardScales
        else:
            scales = sig * ardScales
        
        print('initialize exact GP'+ (' with '+str(len(inducing_points))+' IPs' if not inducing_points is None else '') +' at bandwidth', (myScientificFormat(sig) if not sig is None else ''), 'and lambda', myScientificFormat(lam))
        
        # set up the expert
        # whether we assume noisy or noise-free labels, we set up a likelihood here to match the full GP implementation!
        if self.outputDim > 1:
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.outputDim, noise_constraint=Positive(transform=torch.exp, inv_transform=torch.log), has_global_noise=self.pars['has_global_noise'], has_task_noise=self.pars['has_task_noise'])
        else:
            if self.pars['homoscedastic']:
                likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=Positive(transform=torch.exp, inv_transform=torch.log))
            else:
                likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=self.localNoiseVariance(self.X)) # TODO maybe this should be inducing_points in case of given inducing_points
        if self.pars['doublePrecision']:            
            likelihood = likelihood.double()
            
        has_task_noise = hasattr(likelihood, 'task_noises')
        has_global_noise = hasattr(likelihood, 'noise')
        if self.pars['homoscedastic'] and not noiseLevel is None:
            if self.pars['doublePrecision']:
                if has_global_noise and not has_task_noise:
                    likelihood.noise = torch.tensor(noiseLevel).double()
                if not has_global_noise and has_task_noise:
                    likelihood.task_noises = torch.tensor(noiseLevel).double()
                if has_global_noise and has_task_noise:
                    likelihood.noise = torch.tensor(noiseLevel/2.).double()
                    likelihood.task_noises = torch.tensor(noiseLevel/2.).double()
            else:
                if has_global_noise and not has_task_noise:
                    likelihood.noise = torch.tensor(noiseLevel).float()
                if not has_global_noise and has_task_noise:
                    likelihood.task_noises = torch.tensor(noiseLevel).float()
                if has_global_noise and has_task_noise:
                    likelihood.noise = torch.tensor(noiseLevel/2.).float()
                    likelihood.task_noises = torch.tensor(noiseLevel/2.).float()
            
        if self.derivativeModel:
            self.model = derivativeExactGPModel(self.X, self.y, likelihood, initialMean = meanValue, inducing_points = inducing_points, initialBandwidth = scales, initialLambda = lam, applyARD = self.pars['applyARD'], doublePrecision = self.pars['doublePrecision'], jitter = self.pars['jitter'], cholJitter = self.pars['cholJitter'], useLabels = self.labelModel, fixedInducingPoints = self.pars['fixedIPlocations'])
        else:
            self.model = ExactGPModel(self.X, self.y, likelihood, initialMean = meanValue, inducing_points = inducing_points, initialBandwidth = scales, initialLambda = lam, applyARD = self.pars['applyARD'], doublePrecision = self.pars['doublePrecision'], jitter = self.pars['jitter'], cholJitter = self.pars['cholJitter'], fixedInducingPoints = self.pars['fixedIPlocations'])
        
        if self.pars['doublePrecision']:
            self.model.covar_module = self.model.covar_module.double()
        
        self.eval()
        
    def trainExactGP(self):
        
        # check if there is actually anything to train
        noTrainNeeded = (self.pars['fixedIPlocations'] or not hasattr(self.model.covar_module, 'inducing_points')) and (not self.labelModel or self.pars['fixedMean']) and (self.lossFunction == 'mse' or self.pars['fixedNoise']) and (self.pars['fixedLambda']) and (self.pars['fixedSigma'])
        if noTrainNeeded:
            print('Model has no trainable parameters')
            return
            
        likelihood = self.model.likelihood
        
        if self.lossFunction == 'mse':
            #mseLoss = torch.nn.MSELoss()
            looMSELoss = WeightedLeaveOneOutMSE(likelihood, self.model)
                
        if self.lossFunction == 'mll':
            #mll = WeightedExactPredictiveLogLikelihood(likelihood, self.model)
            mll = WeightedLeaveOneOutPseudoLikelihood(likelihood, self.model)
            
        if self.valLoss == 'mll':
            valMll = WeightedExactPredictiveLogLikelihood(likelihood, self.model)

            
        # now do the training
        likelihood.train()
        self.train()
        
        print('train a full global expert')

        allParams = []
        par_lr_thresholds = []
        weight_decay = self.pars['weightDecay']
        
        # the expert inducing point positions
        if hasattr(self.model.covar_module, 'inducing_points') and not self.pars['fixedIPlocations']:
            parsExpertInducingPoints = [self.model.covar_module.inducing_points]
            allParams.append({'params': parsExpertInducingPoints, 'lr': self.pars['LR']*self.pars['lrFactorIPlocations'], 'weight_decay': weight_decay})
            par_lr_thresholds.append(self.pars['minLR']*self.pars['lrFactorIPlocations'])
        
        # the expert hyperparameters
        if self.labelModel:
            if self.pars['fixedMean']:
                self.model.mean_module.constant.requires_grad = False
            else:
                allParams.append({'params': list(self.model.mean_module.parameters()), 'lr': self.pars['LR']*self.pars['lrFactorGPRmean'], 'weight_decay': weight_decay})
                par_lr_thresholds.append(self.pars['minLR'] * self.pars['lrFactorGPRmean'])
        
        if self.lossFunction == 'mll':
            if self.pars['homoscedastic']:
                if self.pars['fixedNoise']:
                    if hasattr(likelihood, 'noise'):
                        likelihood.raw_noise.requires_grad=False
                    if hasattr(likelihood, 'task_noises'):
                        likelihood.raw_task_noises.requires_grad=False
                else:
                    allParams.append({'params': list(likelihood.parameters()), 'lr': self.pars['LR']*self.pars['lrFactorGPRhyperparameters'], 'weight_decay': weight_decay})
                    par_lr_thresholds.append(self.pars['minLR'] * self.pars['lrFactorGPRhyperparameters'])
                
        if self.pars['fixedLambda']:
            # it is computationally more efficient, iff we do not plan to update a variable, then to put its gradient off
            if hasattr(self.model.covar_module, 'raw_outputscale'):
                self.model.covar_module.raw_outputscale.requires_grad=False
            else:
                self.model.covar_module.base_kernel.raw_outputscale.requires_grad=False
        else:
            if hasattr(self.model.covar_module, 'raw_outputscale'):
                allParams.append({'params': self.model.covar_module.raw_outputscale, 'lr': self.pars['LR']*self.pars['lrFactorGPRhyperparameters'], 'weight_decay': weight_decay})
            else:
                allParams.append({'params': self.model.covar_module.base_kernel.raw_outputscale, 'lr': self.pars['LR']*self.pars['lrFactorGPRhyperparameters'], 'weight_decay': weight_decay})
            par_lr_thresholds.append(self.pars['minLR'] * self.pars['lrFactorGPRhyperparameters'])
        
        if self.pars['fixedSigma']:
            # it is computationally more efficient, iff we do not plan to update a variable, then to put its gradient off
            if hasattr(self.model.covar_module.base_kernel, 'raw_lengthscale'):
                self.model.covar_module.base_kernel.raw_lengthscale.requires_grad = False
            else:
                self.model.covar_module.base_kernel.base_kernel.raw_lengthscale.requires_grad = False
        else:
            if hasattr(self.model.covar_module.base_kernel, 'raw_lengthscale'):
                allParams.append({'params': self.model.covar_module.base_kernel.raw_lengthscale, 'lr': self.pars['LR']*self.pars['lrFactorGPRhyperparameters'], 'weight_decay': weight_decay})
            else:
                allParams.append({'params': self.model.covar_module.base_kernel.base_kernel.raw_lengthscale, 'lr': self.pars['LR']*self.pars['lrFactorGPRhyperparameters'], 'weight_decay': weight_decay})
            par_lr_thresholds.append(self.pars['minLR'] * self.pars['lrFactorGPRhyperparameters'])
        
        optimizer = torch.optim.Adam(allParams)
        
        if self.pars['validationFrequency'] % self.pars['subValidationFrequency'] == 0:
            subValidationFrequency = self.pars['subValidationFrequency']
        else:
            subValidationFrequency = 1
        patience = int(self.pars['patienceFactor'] * self.pars['validationFrequency'] / subValidationFrequency)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience = patience, min_lr=par_lr_thresholds)


        weights = self.iwTrain
        weights = weights * len(weights) / sum(weights).item()
        rootWeights = weights.pow(0.5).reshape(-1,1)

        epoch = 0
        while epoch < self.pars['maxEPOCH']:
            # make sure to reestimate alpha of the expert
            self.model._clear_cache()

            # implement weighted loss
            if self.lossFunction == 'mse':
                loss = looMSELoss(self.y.squeeze(), weights = weights)
            if self.lossFunction == 'mll':
                if self.pars['homoscedastic']:
                    loss = -mll(self.y.squeeze(), weights = weights)
                else:
                    localNoise = self.localNoiseVariance(self.X)
                    loss = -mll(self.y.squeeze(), weights = weights, noise = localNoise)

            objective_loss = loss.item()
            loss_str = myScientificFormat(objective_loss) + ' (objective)'
            allLosses = (objective_loss,)

            if not self.pars['fixedSigma'] and self.pars['applyARD'] and self.pars['smallARDbandwidthPenalty'] > 0:
                if hasattr(self.model.covar_module.base_kernel, 'raw_lengthscale'):
                    smallARDbandwidth_loss = - self.model.covar_module.base_kernel.raw_lengthscale.mean()
                else:
                    smallARDbandwidth_loss = - self.model.covar_module.base_kernel.base_kernel.raw_lengthscale.mean()

                loss += self.pars['smallARDbandwidthPenalty']*smallARDbandwidth_loss
                bwLoss = loss.item() - sum(allLosses)
                allLosses += (bwLoss,)
                loss_str += ' + ' + myScientificFormat(bwLoss) + ' (small ARD bandwidth penalty)'

            if not self.pars['fixedSigma'] and self.pars['applyARD'] and self.pars['anisotropicBandwidthPenalty'] > 0:
                anisotropicBandwidth_loss = self.model.covar_module.base_kernel.lengthscale.var() / (self.model.covar_module.base_kernel.lengthscale.mean()**2 + 1e-10)

                loss += self.pars['anisotropicBandwidthPenalty']*anisotropicBandwidth_loss
                bwLoss = loss.item() - sum(allLosses)
                allLosses += (bwLoss,)
                loss_str += ' + ' + myScientificFormat(bwLoss) + ' (anisotropic bandwidth penalty)'

            allLosses = (loss,) + allLosses
            loss_str = 'Loss: ' + myScientificFormat(loss) + ' = ' + loss_str
            self.loss_str = loss_str

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
                
            # do validation
            if ((epoch+1) % subValidationFrequency == 0) or ((epoch+1) % self.pars['validationFrequency'] == 0):
                self.eval()
                likelihood.eval()
                if self.valLoss == 'mll':
                    with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False), gpytorch.settings.fast_pred_var(True):
                        valPred = self(self.xVal, jitter = self.pars['jitter'], cholJitter = self.pars['cholJitter'])
                    if self.pars['homoscedastic']:
                        if self.labelModel:
                            valLoss = -valMll(valPred, self.yVal.squeeze(), weights = self.iwVal)
                        else:
                            valLoss = -valMll(valPred, self.yVal[:,1:].squeeze(), weights = self.iwVal)
                    else:
                        localValNoise = self.localNoiseVariance(self.xVal)
                        if self.labelModel:
                            valLoss = -valMll(valPred, self.yVal.squeeze(), weights = self.iwVal, noise = localValNoise)
                        else:
                            valLoss = -valMll(valPred, self.yVal[:,1:].squeeze(), weights = self.iwVal, noise = localValNoise)

                if self.valLoss == 'mse':
                    with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False), gpytorch.settings.fast_pred_var(True):
                        valPred = self(self.xVal, predictionsOnly = True, labelsOnly = False, jitter = self.pars['jitter'], cholJitter = self.pars['cholJitter']).mean

                    if valPred.ndim < 2:
                        valPred = valPred.unsqueeze(-1)

                    if self.labelModel:
                        valLoss = torch.mean((valPred - self.yVal)**2 * self.iwVal.reshape(-1,1))**0.5
                        if self.derivativeModel:
                            valRMSElabel = torch.mean((valPred[:,0] - self.yVal[:,0])**2 * self.iwVal)**0.5
                            valRMSEderivative = torch.mean((valPred[:,1:] - self.yVal[:,1:])**2 * self.iwVal.reshape(-1,1))**0.5
                    else:
                        valLoss = torch.mean((valPred - self.yVal[:,1:])**2 * self.iwVal.reshape(-1,1))**0.5
                        
                valLoss = valLoss.item()

                likelihood.train()
                self.train()
            if (epoch+1) % subValidationFrequency == 0:
                # if validation loss stagnates, reduce learning rate of optimizer
                scheduler.step(valLoss)

                
            # print some variables
            if (epoch+1) % self.pars['validationFrequency'] == 0:
                
                printString = 'Iter '+ str(epoch+1)
                if self.labelModel:
                    printMean = myScientificFormat(self.model.mean_module.constant.item())
                    printString += ', mean ' + str(printMean)
                    
                if not self.pars['noiseFree']:
                    if self.pars['homoscedastic']:
                        printString += ', global noise ' + myScientificFormat(likelihood.noise.item(), 2)
                    else:
                        printString += ', heteroscedastic'
                
                if hasattr(self.model.covar_module, 'raw_outputscale'):
                    printLambda = myScientificFormat(self.model.covar_module.outputscale.item())
                else:
                    printLambda = myScientificFormat(self.model.covar_module.base_kernel.outputscale.item())
                printString += ', lambda ' + str(printLambda)
                
                if hasattr(self.model.covar_module.base_kernel, 'raw_lengthscale'):
                    if self.pars['applyARD']:
                        printSigma = [myScientificFormat(sig.item()) for sig in self.model.covar_module.base_kernel.lengthscale[0]] # lengthscale is 1xd
                    else:
                        printSigma = myScientificFormat(self.model.covar_module.base_kernel.lengthscale.item())
                else:
                    if self.pars['applyARD']:
                        printSigma = [myScientificFormat(sig.item()) for sig in self.model.covar_module.base_kernel.base_kernel.lengthscale[0]] # lengthscale is 1xd
                    else:
                        printSigma = myScientificFormat(self.model.covar_module.base_kernel.base_kernel.lengthscale.item())
                printString += ', sigma ' + str(printSigma)

                print(printString)
                print(loss_str)
                
                if self.valLoss == 'mse':
                    if self.labelModel and self.derivativeModel:
                        val_loss_str = 'current validation RMSE total: ' + str(valLoss) + ', label: ' + str(valRMSElabel) + ', derivatives: ' + str(valRMSEderivative)
                    else:
                        val_loss_str = 'current validation RMSE: ' + str(valLoss)
                if self.valLoss == 'mll':
                    val_loss_str = 'current validation MLL: ' + str(valLoss)
                self.val_loss_str = val_loss_str
                
                print(val_loss_str)
                print('current learning rate: ', myScientificFormat(optimizer.param_groups[0]['lr']))
            
            epoch += 1
            if self.pars['LR'] > self.pars['minLR'] and optimizer.param_groups[0]['lr'] == par_lr_thresholds[0]:
                break
        self.eval()
    
class ExactGPModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, initialMean = None, inducing_points = None, initialBandwidth = None, initialLambda = None, applyARD = False, doublePrecision = False, jitter = None, cholJitter = None, fixedInducingPoints = True):
        
        exactSparse = (not inducing_points is None)
        
        if not torch.is_tensor(train_x):
            if doublePrecision:
                train_x = torch.from_numpy(train_x).double()
                train_y = torch.from_numpy(train_y).double()
            else:
                train_x = torch.from_numpy(train_x).float()
                train_y = torch.from_numpy(train_y).float()
    
        super(ExactGPModel, self).__init__(train_x, train_y.squeeze(), likelihood)
        self.jitter = jitter
        self.cholJitter = cholJitter
        
        self.mean_module = gpytorch.means.ConstantMean().type(self.train_targets.dtype)
        if initialMean is None:
            initialMean = train_y.mean()
        torch.nn.init.constant_(self.mean_module.constant, val = initialMean)
        
        inputDim = train_x[0].numel()
        if applyARD:
            ard_num_dims = inputDim
        else:
            ard_num_dims = None
        
        # standard constraint is Positive() which applies Softplus. It behaves awkward for larger values
        if exactSparse:
            # exact sparse GP
            self.covar_module = gpytorch.kernels.InducingPointKernel(gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims, lengthscale_constraint=Positive(transform=torch.exp, inv_transform=torch.log)), outputscale_constraint=Positive(transform=torch.exp, inv_transform=torch.log)), inducing_points = inducing_points, likelihood=likelihood, cholJitter = self.cholJitter)
            self.covar_module.inducing_points.requires_grad = not fixedInducingPoints
        else:
            # exact full GP
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims, lengthscale_constraint=Positive(transform=torch.exp, inv_transform=torch.log)), outputscale_constraint=Positive(transform=torch.exp, inv_transform=torch.log))
        
        if exactSparse:
            if not initialLambda is None:
                self.covar_module.base_kernel._set_outputscale(initialLambda)
            if not initialBandwidth is None:
                self.covar_module.base_kernel.base_kernel._set_lengthscale(initialBandwidth)
        else:
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
        
    def looPredictiveDistribution(self, **kwargs):
        if _is_in_cache_ignore_args(self, "mu"):
            return MultivariateNormal(self._memoize_cache['mu'], DiagLinearOperator(self._memoize_cache['sigma2']))
        else:
            #print('going into loo Preds')
            trbckup = self.training
            self.training = True
            # if we put training off, we obtain the posterior distribution instead!
            trainingPriorDistribution = self(*self.train_inputs, jitter = self.jitter, cholJitter = self.cholJitter)
            self.training = trbckup
            # this adds the noise term to the diagonal
            output = self.likelihood(trainingPriorDistribution, self.train_inputs)
            m = output.mean
            m = m.reshape(*self.train_targets.shape)
            L = output.lazy_covariance_matrix.cholesky(upper=False, jitter = self.cholJitter)
            identity = torch.eye(*L.shape[-2:], dtype=m.dtype, device=m.device)
            sigma2 = 1.0 / L._cholesky_solve(identity, upper=False).diagonal(dim1=-1, dim2=-2)  # 1 / diag(inv(K))
            mu = self.train_targets - L._cholesky_solve((self.train_targets - m).unsqueeze(-1), upper=False).squeeze(-1) * sigma2
            
            _add_to_cache_ignore_args(self, "mu", mu)
            _add_to_cache_ignore_args(self, "sigma2", sigma2)
        
            if mu.grad_fn is not None:
                wrapper = functools.partial(clear_cache_hook, self)
                functools.update_wrapper(wrapper, clear_cache_hook)
                mu.grad_fn.register_hook(wrapper)
            if sigma2.grad_fn is not None:
                wrapper = functools.partial(clear_cache_hook, self)
                functools.update_wrapper(wrapper, clear_cache_hook)
                sigma2.grad_fn.register_hook(wrapper)
        
        return MultivariateNormal(mu, DiagLinearOperator(sigma2))
    
class derivativeExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, initialMean = None, inducing_points = None, initialBandwidth = None, initialLambda = None, applyARD = False, useLabels = True, doublePrecision = True, jitter = None, cholJitter = None, fixedInducingPoints = True):
    
        exactSparse = (not inducing_points is None)
        
        if not torch.is_tensor(train_x):
            if doublePrecision:
                train_x = torch.from_numpy(train_x).double()
                train_y = torch.from_numpy(train_y).double()
            else:
                train_x = torch.from_numpy(train_x).float()
                train_y = torch.from_numpy(train_y).float()
    
        if useLabels:
            super(derivativeExactGPModel, self).__init__(train_x, train_y, likelihood)
        else:
            super(derivativeExactGPModel, self).__init__(train_x, train_y[:,1:], likelihood)
        self.jitter = jitter
        self.cholJitter = cholJitter
        
        self.mean_module = gpytorch.means.ConstantMeanGrad(onlyDerivative = not useLabels).type(train_y.dtype)
        if useLabels:
            if initialMean is None:
                initialMean = train_y[:,0].mean()
            torch.nn.init.constant_(self.mean_module.constant, val = initialMean)
        # double casting does not work for the mean module, if there is no parameter, i.e. when labels are not included, but only derivatives
        
        inputDim = train_x[0].numel()
        outputDim = inputDim
        if useLabels:
            outputDim += 1

        if applyARD:
            ard_num_dims = inputDim
        else:
            ard_num_dims = None
            
        # standard constraint is Positive() which applies Softplus. It behaves awkward for larger values
        if exactSparse:
            # exact sparse GP
            self.covar_module = gpytorch.kernels.InducingPointKernel(gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernelGrad(ard_num_dims=ard_num_dims, lengthscale_constraint = Positive(transform = torch.exp,  inv_transform = torch.log), onlyDerivatives = not useLabels), outputscale_constraint=Positive(transform=torch.exp, inv_transform=torch.log)), inducing_points = inducing_points, likelihood=likelihood)
            self.covar_module.inducing_points.requires_grad = not fixedInducingPoints
        else:
            # exact full GP
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernelGrad(ard_num_dims=ard_num_dims, lengthscale_constraint = Positive(transform = torch.exp,  inv_transform = torch.log), onlyDerivatives = not useLabels), outputscale_constraint=Positive(transform=torch.exp, inv_transform=torch.log))
        
        if exactSparse:
            if not initialLambda is None:
                self.covar_module.base_kernel._set_outputscale(initialLambda)
            if not initialBandwidth is None:
                self.covar_module.base_kernel.base_kernel._set_lengthscale(initialBandwidth)
        else:
            if not initialLambda is None:
                self.covar_module._set_outputscale(initialLambda)
            if not initialBandwidth is None:
                self.covar_module.base_kernel._set_lengthscale(initialBandwidth)
            
        self.outputDimension = outputDim

    def forward(self, x, y = None, **kwargs):
        if not y is None:
            mean_x = self.mean_module(y)#.type(x.dtype)
            covar_x = self.covar_module(x, y)
        else:
            mean_x = self.mean_module(x)#.type(x.dtype)
            covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)
        
    def looPredictiveDistribution(self, calculateSigma = True):

        if _is_in_cache_ignore_args(self, "mu"):
            if calculateSigma:
                return MultitaskMultivariateNormal(self._memoize_cache['mu'], DiagLinearOperator(self._memoize_cache['sigma2']))
            else:
                return MultitaskMultivariateNormal(self._memoize_cache['mu'], DiagLinearOperator(torch.ones_like(self._memoize_cache['mu']).flatten(-2)))
        else:
            trbckup = self.training
            self.training = True
            # if we put training off, we obtain the posterior distribution instead!
            trainingPriorDistribution = self(*self.train_inputs, jitter = self.jitter, cholJitter = self.cholJitter)
            self.training = trbckup
            # this adds the noise term to the diagonal
            output = self.likelihood(trainingPriorDistribution, self.train_inputs)
            m = output.mean
            m = m.reshape(*self.train_targets.shape)
            L = output.lazy_covariance_matrix.cholesky(upper=False, jitter = self.cholJitter)
            identity = torch.eye(*L.shape[-2:], dtype=m.dtype, device=m.device)
            sigma2 = 1.0 / L._cholesky_solve(identity, upper=False).diagonal(dim1=-1, dim2=-2)  # 1 / diag(inv(K))
            mu = torch.flatten(self.train_targets,-2) - L._cholesky_solve(torch.flatten(self.train_targets - m,-2).unsqueeze(-1), upper=False).squeeze(-1) * sigma2
            mu = mu.reshape(*self.train_targets.shape)
            _add_to_cache_ignore_args(self, "mu", mu)
            if calculateSigma:
                _add_to_cache_ignore_args(self, "sigma2", sigma2)
        
            if mu.grad_fn is not None:
                wrapper = functools.partial(clear_cache_hook, self)
                functools.update_wrapper(wrapper, clear_cache_hook)
                mu.grad_fn.register_hook(wrapper)
            if calculateSigma and sigma2.grad_fn is not None:
                wrapper = functools.partial(clear_cache_hook, self)
                functools.update_wrapper(wrapper, clear_cache_hook)
                sigma2.grad_fn.register_hook(wrapper)
        
        if calculateSigma:
            return MultitaskMultivariateNormal(mu, DiagLinearOperator(sigma2))
        else:
            return MultitaskMultivariateNormal(mu, DiagLinearOperator(torch.ones_like(mu).flatten(-2)))
        
    def locoPredictiveDistribution(self, calculateSigma = True): # this is the leave-one-configuration-out version of loo
    
        if _is_in_cache_ignore_args(self, "mu"):
            if calculateSigma:
                return MultitaskMultivariateNormal(self._memoize_cache['mu'], BlockDiagLinearOperator(self._memoize_cache['sigma2']))
            else:
                return MultitaskMultivariateNormal(self._memoize_cache['mu'], DiagLinearOperator(torch.ones_like(self._memoize_cache['mu']).flatten(-2)))
        else:
            trbckup = self.training
            self.training = True
            # if we put training off, we obtain the posterior distribution instead!
            trainingPriorDistribution = self(*self.train_inputs, jitter = self.jitter, cholJitter = self.cholJitter)
            self.training = trbckup
            # this adds the noise term to the diagonal
            output = self.likelihood(trainingPriorDistribution, self.train_inputs)
            m = output.mean
            m = m.reshape(*self.train_targets.shape)
            L = output.lazy_covariance_matrix.cholesky(upper=False, jitter = self.cholJitter)
            identity = torch.eye(*L.shape[-2:], dtype=m.dtype, device=m.device)
            covInv = L._cholesky_solve(identity, upper=False)
            muFull = L._cholesky_solve(torch.flatten(self.train_targets - m,-2).unsqueeze(-1), upper=False).squeeze(-1)
            numConfigs = len(self.train_targets)
            mu = torch.zeros_like(self.train_targets)
            mu += self.train_targets
            if calculateSigma:
                smallIdentity = torch.eye(self.outputDimension, self.outputDimension, dtype=m.dtype, device=m.device)
                sigma2 = torch.zeros([numConfigs,self.outputDimension,self.outputDimension])
            for i in range(numConfigs):
                configInx = i*self.outputDimension + np.arange(self.outputDimension)
                Lsub = psd_safe_cholesky(covInv[configInx][:, configInx].type(gpytorch.settings._linalg_dtype_cholesky.value()), jitter = self.cholJitter, max_tries=100)
                Lsub = TriangularLinearOperator(Lsub)
                mu[i] -= Lsub._cholesky_solve(muFull[configInx].unsqueeze(-1), upper=False).squeeze(-1)
                if calculateSigma:
                    sigma2[i] = Lsub._cholesky_solve(smallIdentity, upper=False)
                    
            _add_to_cache_ignore_args(self, "mu", mu)
            if calculateSigma:
                _add_to_cache_ignore_args(self, "sigma2", sigma2)
        
            if mu.grad_fn is not None:
                wrapper = functools.partial(clear_cache_hook, self)
                functools.update_wrapper(wrapper, clear_cache_hook)
                mu.grad_fn.register_hook(wrapper)
            if calculateSigma and sigma2.grad_fn is not None:
                wrapper = functools.partial(clear_cache_hook, self)
                functools.update_wrapper(wrapper, clear_cache_hook)
                sigma2.grad_fn.register_hook(wrapper)
        
        if calculateSigma:
            return MultitaskMultivariateNormal(mu, BlockDiagLinearOperator(sigma2))
        else:
            return MultitaskMultivariateNormal(mu, DiagLinearOperator(torch.ones_like(mu).flatten(-2)))
        
class ApproxGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, likelihood = None, initialBandwidth = None, initialLambda = None, fixedInducingPoints = True, inducingCovarType = 'scalar', applyARD = False, initConstant = 0.):
        
        if inducingCovarType == 'full':
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        if inducingCovarType == 'diag':
            variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(inducing_points.size(0))
        if inducingCovarType == 'scalar':
            variational_distribution = gpytorch.variational.DeltaVariationalDistribution(inducing_points.size(0))
        
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations= not fixedInducingPoints).type(inducing_points.dtype)
        super(ApproxGPModel, self).__init__(variational_strategy)
        
        self.likelihood = likelihood
        
        self.mean_module = gpytorch.means.ConstantMean().type(inducing_points.dtype)
        torch.nn.init.constant_(self.mean_module.constant, val = initConstant)
        
        inputDim = inducing_points[0].numel()
        if applyARD:
            ard_num_dims = inputDim
        else:
            ard_num_dims = None
        
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims, lengthscale_constraint=Positive(transform=torch.exp, inv_transform=torch.log)), outputscale_constraint=Positive(transform=torch.exp, inv_transform=torch.log))
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
        
class derivativeApproxGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, likelihood = None, initialBandwidth = None, initialLambda = None, fixedInducingPoints = True, inducingCovarType = 'scalar', applyARD = False, initConstant = 0., useLabels = True):
        
        inputDim = inducing_points[0].numel()
        outputDim = inputDim
        if useLabels:
            outputDim += 1
        if applyARD:
            ard_num_dims = inputDim
        else:
            ard_num_dims = None
        
        if inducingCovarType == 'full':
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(-2), batch_shape=torch.Size([outputDim]))
        if inducingCovarType == 'diag':
            variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(inducing_points.size(-2), batch_shape=torch.Size([outputDim]))
        if inducingCovarType == 'scalar':
            variational_distribution = gpytorch.variational.DeltaVariationalDistribution(inducing_points.size(-2), batch_shape=torch.Size([outputDim]))
        
        
        from variational_strategy_derivative import VariationalStrategy_derivative
        variational_strategy = VariationalStrategy_derivative(self, inducing_points, variational_distribution, learn_inducing_locations= not fixedInducingPoints).type(inducing_points.dtype)
            
        super().__init__(variational_strategy)
        
        self.likelihood = likelihood
        
        self.mean_module = gpytorch.means.ConstantMeanGrad(onlyDerivative=not useLabels).type(inducing_points.dtype)
        if useLabels:
            torch.nn.init.constant_(self.mean_module.constant, val = initConstant)
        # double casting does not work for the mean module, if there is no parameter, i.e. when labels are not included, but only derivatives
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernelGrad(ard_num_dims=ard_num_dims, lengthscale_constraint = Positive(transform = torch.exp,  inv_transform = torch.log), onlyDerivatives = not useLabels), outputscale_constraint=Positive(transform=torch.exp, inv_transform=torch.log))
        
        if not initialLambda is None:
            self.covar_module._set_outputscale(initialLambda)
        if not initialBandwidth is None:
            self.covar_module.base_kernel._set_lengthscale(initialBandwidth)
        self.outputDimension = outputDim
            
    def forward(self, x, y = None, **kwargs):
        if not y is None:
            mean_x = self.mean_module(y)
            covar_x = self.covar_module(x, y)
        else:
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)
    
class WeightedPredictiveLogLikelihood(_ApproximateMarginalLogLikelihood):

    def _log_likelihood_term(self, approximate_dist_f, target, **kwargs):
        if 'weights' in kwargs:
            w = kwargs['weights']
            return (self.likelihood.log_marginal(target, approximate_dist_f, **kwargs) * w).sum(-1)
        else:
            return self.likelihood.log_marginal(target, approximate_dist_f, **kwargs).sum(-1)

    def forward(self, approximate_dist_f, target, **kwargs):

        return super().forward(approximate_dist_f, target, **kwargs)
        
class WeightedExactPredictiveLogLikelihood(ExactMarginalLogLikelihood):

    def __init__(self, likelihood, model):
        super().__init__(likelihood=likelihood, model=model)
        self.likelihood = likelihood
        self.model = model

    def forward(self, function_dist: MultivariateNormal, target: Tensor, *params, **kwargs) -> Tensor:
        if not isinstance(function_dist, MultivariateNormal):
            raise RuntimeError("ExactMarginalLogLikelihood can only operate on Gaussian random variables")

        if 'weights' in kwargs:
            w = kwargs['weights']
            res = (self.likelihood.log_marginal(target, function_dist, **kwargs) * w).sum(-1)
        else:
            res = self.likelihood.log_marginal(target, function_dist, **kwargs).sum(-1)

        # Get the log prob of the marginal distribution
        #output = self.likelihood(function_dist, *params)
        #res = output.log_marginal(target)
        res = self._add_other_terms(res, params)
        
        # Scale by the amount of data we have
        num_batch = function_dist.event_shape[0]
        return res.div_(num_batch)
        
class WeightedLeaveOneOutPseudoLikelihood(ExactMarginalLogLikelihood):

    def __init__(self, likelihood, model):
        super().__init__(likelihood=likelihood, model=model)
        self.likelihood = likelihood
        self.model = model

    def forward(self, target: Tensor, *params, **kwargs) -> Tensor: #mu = None, sigma2 = None
        
        if 'mu' in kwargs and 'sigma2' in kwargs:
            mu = kwargs['mu']
            sigma2 = kwargs['sigma2']
        else:
            mu = None
            sigma2 = None
            
        if 'subInx' in kwargs:
            subInx = kwargs['subInx']
        else:
            subInx = None
            
        if mu is None:
            looPredDist = self.model.looPredictiveDistribution()
            mu = looPredDist.mean
            sigma2 = looPredDist.lazy_covariance_matrix.diag()
            if not subInx is None:
                mu = mu[subInx]
                sigma2 = sigma2[subInx]

        term1 = -0.5 * sigma2.log()
        term2 = -0.5 * (target - mu).pow(2.0) / sigma2
        term3 = -0.5 * math.log(2 * math.pi)
        log_lik_terms = term1 + term2 + term3
        if 'weights' in kwargs:
            w = kwargs['weights']
            res = (log_lik_terms*w).sum(dim=-1)
        else:
            res = log_lik_terms.sum(dim=-1)

        res = self._add_other_terms(res, params)
        
        # Scale by the amount of data we have
        num_batch = len(target)
        return res.div_(num_batch)
        
class WeightedLeaveOneOutMSE(ExactMarginalLogLikelihood):
    def __init__(self, likelihood, model):
        super().__init__(likelihood=likelihood, model=model)
        self.likelihood = likelihood
        self.model = model

    def forward(self, target: Tensor, *params, **kwargs) -> Tensor:
        
        if 'mu' in kwargs:
            mu = kwargs['mu']
        else:
            mu = None
            
        if 'subInx' in kwargs:
            subInx = kwargs['subInx']
        else:
            subInx = None
            
        if mu is None:
            looPredDist = self.model.looPredictiveDistribution()
            mu = looPredDist.mean
            if not subInx is None:
                mu = mu[subInx]
            
        squared_error_terms = (target - mu).pow(2.0)
        if 'weights' in kwargs:
            w = kwargs['weights']
            res = (squared_error_terms*w).sum(dim=-1)
        else:
            res = squared_error_terms.sum(dim=-1)

        res = self._add_other_terms(res, params)
        
        # Scale by the amount of data we have
        num_batch = len(target)
        return res.div_(num_batch)
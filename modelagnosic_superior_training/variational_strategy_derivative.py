#!/usr/bin/env python3

import warnings

import torch

from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.lazy import DiagLazyTensor, MatmulLazyTensor, RootLazyTensor, SumLazyTensor, TriangularLazyTensor, delazify
from gpytorch.settings import _linalg_dtype_cholesky, trace_mode
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.utils.errors import CachingError
from gpytorch.utils.memoize import cached, clear_cache_hook, pop_from_cache_ignore_args, _is_in_cache_ignore_args
from gpytorch.utils.warnings import OldVersionWarning
from gpytorch.variational._variational_strategy import _VariationalStrategy

def _ensure_updated_strategy_flag_set(
    state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
):
    device = state_dict[list(state_dict.keys())[0]].device
    if prefix + "updated_strategy" not in state_dict:
        state_dict[prefix + "updated_strategy"] = torch.tensor(False, device=device)
        warnings.warn(
            "You have loaded a variational GP model (using `VariationalStrategy`) from a previous version of "
            "GPyTorch. We have updated the parameters of your model to work with the new version of "
            "`VariationalStrategy` that uses whitened parameters.\nYour model will work as expected, but we "
            "recommend that you re-save your model.",
            OldVersionWarning,
        )


class VariationalStrategy_derivative(_VariationalStrategy):
    r"""
    The standard variational strategy, as defined by `Hensman et al. (2015)`_.
    This strategy takes a set of :math:`m \ll n` inducing points :math:`\mathbf Z`
    and applies an approximate distribution :math:`q( \mathbf u)` over their function values.
    (Here, we use the common notation :math:`\mathbf u = f(\mathbf Z)`.
    The approximate function distribution for any abitrary input :math:`\mathbf X` is given by:

    .. math::

        q( f(\mathbf X) ) = \int p( f(\mathbf X) \mid \mathbf u) q(\mathbf u) \: d\mathbf u

    This variational strategy uses "whitening" to accelerate the optimization of the variational
    parameters. See `Matthews (2017)`_ for more info.

    :param ~gpytorch.models.ApproximateGP model: Model this strategy is applied to.
        Typically passed in when the VariationalStrategy is created in the
        __init__ method of the user defined model.
    :param torch.Tensor inducing_points: Tensor containing a set of inducing
        points to use for variational inference.
    :param ~gpytorch.variational.VariationalDistribution variational_distribution: A
        VariationalDistribution object that represents the form of the variational distribution :math:`q(\mathbf u)`
    :param learn_inducing_locations: (Default True): Whether or not
        the inducing point locations :math:`\mathbf Z` should be learned (i.e. are they
        parameters of the model).
    :type learn_inducing_locations: `bool`, optional

    .. _Hensman et al. (2015):
        http://proceedings.mlr.press/v38/hensman15.pdf
    .. _Matthews (2017):
        https://www.repository.cam.ac.uk/handle/1810/278022
    """

    def __init__(self, model, inducing_points, variational_distribution, learn_inducing_locations=True):
        super().__init__(model, inducing_points, variational_distribution, learn_inducing_locations)
        self.register_buffer("updated_strategy", torch.tensor(True))
        self._register_load_state_dict_pre_hook(_ensure_updated_strategy_flag_set)

    @cached(name="cholesky_factor", ignore_args=True)
    def _cholesky_factor(self, induc_induc_covar, jitter = None):
        # I increased max_tries from 3 to 100
        L = psd_safe_cholesky(delazify(induc_induc_covar).type(_linalg_dtype_cholesky.value()), jitter = jitter, max_tries=100)
        return TriangularLazyTensor(L)

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        zeros = torch.zeros(
            self._variational_distribution.shape(),
            dtype=self._variational_distribution.dtype,
            device=self._variational_distribution.device,
        )
        ones = torch.ones_like(zeros.flatten(-2))
        res = MultitaskMultivariateNormal(zeros, DiagLazyTensor(ones))
        return res
        

    def forward(self, x, inducing_points, inducing_values, variational_inducing_covar=None, **kwargs):
    
        # try the following for reduced computational complexity, if we only require predictions
        predictionsOnly = False
        if 'predictionsOnly' in kwargs:
            predictionsOnly = kwargs['predictionsOnly']
        # furthermore implement, if we are only interested in label predictions without derivative predictions
        labelsOnly = False
        if 'labelsOnly' in kwargs:
            labelsOnly = kwargs['labelsOnly']
        jitter = None
        if 'jitter' in kwargs:
            jitter = kwargs['jitter']
        cholJitter = None
        if 'cholJitter' in kwargs:
            cholJitter = kwargs['cholJitter']
        
            
        trainSize = inducing_points.shape[-2]
        testSize, inputDim = x.shape
        outputDim = self.model.covar_module.base_kernel.num_outputs_per_input(x,None)
        # if we are in batch mode, adapt batch size accordingly
        memoryThreshold = int(2**25)
        batchThresholdSize = int(max(1, memoryThreshold // (trainSize * outputDim**2)))
            
        # idea: in predictionsOnly-mode, if len(x) exceeds a threshold (e.g. 1000), slice up x and call forward again on each slice. Then merge results
        if not self.model.training and predictionsOnly and len(x) > batchThresholdSize:
            predictive_mean = torch.empty(*x.shape[:-1], outputDim)
            inx = 0
            while inx < x.shape[-2]:
                inxOld = inx
                inx += batchThresholdSize
                predictive_mean[..., inxOld:inx, :] = self.forward(x[..., inxOld:inx, :], inducing_points, inducing_values, variational_inducing_covar = variational_inducing_covar, **kwargs).mean
            fakeCovar = DiagLazyTensor(torch.ones_like(predictive_mean))
            return MultivariateNormal(predictive_mean, fakeCovar)
            
        PEsegmentation = self.model.covar_module.base_kernel.PEsegmentation
                    
        # Compute full prior distribution
        if predictionsOnly:
            # distinguish, if we already have L or not
            if not _is_in_cache_ignore_args(self, "cholesky_factor"):
                full_inputs = torch.cat([inducing_points, x], dim=-2)
            else:
                # if we have L in predictions-only mode, we dont need induc_induc_covar part
                full_inputs = x
            full_output = self.model.forward(inducing_points, full_inputs, **kwargs)
        else:
            full_inputs = torch.cat([inducing_points, x], dim=-2)
            full_output = self.model.forward(full_inputs, **kwargs)
            
        # Covariance terms
        num_induc = inducing_points.size(-2)
        num_data = x.size(-2)
        num_total = full_inputs.size(-2)
        num_induc_in_mean = num_total - num_data
        
        # From here, we have two very different covars, depending on, whether a PEsegmentation is specified or not
        if PEsegmentation is None:
            # the usual case
            full_covar = full_output.lazy_covariance_matrix
            
            if not _is_in_cache_ignore_args(self, "cholesky_factor"):
                if jitter is None:
                    induc_induc_covar = full_covar[..., :(num_induc*outputDim), :(num_induc*outputDim)].add_jitter()
                elif jitter > 0.:
                    induc_induc_covar = full_covar[..., :(num_induc*outputDim), :(num_induc*outputDim)].add_jitter(jitter_val=jitter)
                else:
                    induc_induc_covar = full_covar[..., :(num_induc*outputDim), :(num_induc*outputDim)]
                # I added this for enforcing symmetry
                induc_induc_covar = induc_induc_covar.evaluate()
                induc_induc_covar = (induc_induc_covar + induc_induc_covar.transpose(-2,-1))/2
            else:
                induc_induc_covar = None
            
            L = self._cholesky_factor(induc_induc_covar, jitter = cholJitter)

            if L.shape != torch.Size([num_induc*outputDim,num_induc*outputDim]):
                # Aggressive caching can cause nasty shape incompatibilies when evaluating with different batch shapes
                # TODO: Use a hook fo this
                try:
                    pop_from_cache_ignore_args(self, "cholesky_factor")
                except CachingError:
                    pass
                induc_induc_covar = self.model.forward(inducing_points, inducing_points, **kwargs).lazy_covariance_matrix
                if jitter is None:
                    induc_induc_covar = induc_induc_covar.add_jitter()
                elif jitter > 0.:
                    induc_induc_covar = induc_induc_covar.add_jitter(jitter_val=jitter)
                L = self._cholesky_factor(induc_induc_covar, jitter = cholJitter)
                
            if labelsOnly:
                test_mean = full_output.mean[..., num_induc_in_mean:, [0]]
                labelIndices = num_induc_in_mean*outputDim + torch.arange(num_data)*outputDim
                all_data_covar = full_covar[..., labelIndices]
                induc_data_covar = all_data_covar[..., :(num_induc*outputDim), :].evaluate()
                if not predictionsOnly:
                    data_data_covar = all_data_covar[..., labelIndices, :]
                outputDim = 1
            else:
                test_mean = full_output.mean[..., num_induc_in_mean:, :]
                induc_data_covar = full_covar[..., :(num_induc*outputDim), (num_induc_in_mean*outputDim):].evaluate()
                if not predictionsOnly:
                    data_data_covar = full_covar[..., (num_induc*outputDim):, (num_induc_in_mean*outputDim):]
            
            # Compute the mean of q(f)
            # k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X
            # TODO While the below way of evaluation is faster, there seems to be no savings (probably backpropagation becomes much more involved?)
            # if predictionsOnly:
                # predictive_mean = (induc_data_covar.transpose(-1, -2).type(_linalg_dtype_cholesky.value()) @ L.transpose(-1, -2).inv_matmul(torch.flatten(inducing_values,-2).unsqueeze(-1).type(_linalg_dtype_cholesky.value())).squeeze(-1).reshape(-1, outputDim)).to(full_inputs.dtype) + test_mean
            # else:
            
            # Compute interpolation terms
            # K_ZZ^{-1/2} K_ZX
            # K_ZZ^{-1/2} \mu_Z
            interp_term = L.inv_matmul(induc_data_covar.type(_linalg_dtype_cholesky.value())).to(full_inputs.dtype)
            predictive_mean = (interp_term.transpose(-1, -2) @ torch.flatten(inducing_values,-2).unsqueeze(-1)).squeeze(-1).reshape(-1, outputDim) + test_mean
            
            if predictionsOnly:
                #return predictive_mean
                fakeCovar = DiagLazyTensor(torch.ones_like(predictive_mean))
                return MultivariateNormal(predictive_mean, fakeCovar)

            # Compute the covariance of q(f)
            # K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
            middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1)
            if variational_inducing_covar is not None:
                middle_term = SumLazyTensor(variational_inducing_covar, middle_term)

            if trace_mode.on():
                predictive_covar = (
                    data_data_covar.add_jitter(1e-4).evaluate()
                    + interp_term.transpose(-1, -2) @ middle_term.evaluate() @ interp_term
                )
            else:
                predictive_covar = SumLazyTensor(
                    data_data_covar.add_jitter(1e-4),
                    MatmulLazyTensor(interp_term.transpose(-1, -2), middle_term @ interp_term),
                )

            # Return the distribution
            return MultitaskMultivariateNormal(predictive_mean, predictive_covar)
        else:
            # case of a given PE segmentation
            if self.model.covar_module.base_kernel.onlyForces:
                inputDim = outputDim
                atomInxes = torch.tensor([0,1,2])
                forceBlockOffset = 0
            else:
                inputDim = outputDim - 1
                atomInxes = torch.tensor([1,2,3])
                forceBlockOffset = 3
            
            # labels of a configuration are ordered as follows: [energy, xyz-forces wrt atom 1, ..., xyz-forces wrt atom 9]
            forceInducInxes = []
            groupwiseForceInducInxes = []
            for subSeg in PEsegmentation:
                # get the correlated inxes within a configuration
                subInx = ((torch.tensor(subSeg)*3).reshape(-1,1) + atomInxes).flatten()
                forceInducInxes += (((torch.arange(num_induc)*outputDim).reshape(-1,1) + subInx).flatten()).tolist()
                groupwiseForceInducInxes.append((((torch.arange(num_induc)*outputDim).reshape(-1,1) + subInx).flatten()).tolist())
            totalInducOrder = forceInducInxes
            if not self.model.covar_module.base_kernel.onlyForces:
                energyInducInxes = (torch.arange(num_induc)*outputDim).tolist()
                totalInducOrder += energyInducInxes
            revertInducOrder = sorted(range(len(totalInducOrder)), key=lambda k: totalInducOrder[k])
            
            forceDataInxes = []
            groupwiseForceDataInxes = []
            for subSeg in PEsegmentation:
                # get the correlated inxes within a configuration
                subInx = ((torch.tensor(subSeg)*3).reshape(-1,1) + atomInxes).flatten()
                forceDataInxes += (((torch.arange(num_data)*outputDim).reshape(-1,1) + subInx).flatten()).tolist()
                groupwiseForceDataInxes.append((((torch.arange(num_data)*outputDim).reshape(-1,1) + subInx).flatten()).tolist())
            totalDataOrder = forceDataInxes
            if not self.model.covar_module.base_kernel.onlyForces:
                energyDataInxes = (torch.arange(num_data)*outputDim).tolist()
                totalDataOrder += energyDataInxes
            revertDataOrder = sorted(range(len(totalDataOrder)), key=lambda k: totalDataOrder[k])
            
            full_covar = full_output.lazy_covariance_matrix.evaluate()
            #if self.L is None:
        
            # get induc_induc_covar part from full_covar
            induc_induc_covar = []
            if not self.model.covar_module.base_kernel.onlyForces:
                induc_induc_covar.append(full_covar[0][:num_induc,:num_induc])
                
                induc_induc_covar.append(full_covar[1][:num_induc,:(num_induc*inputDim)])
                
                induc_induc_covar.append(full_covar[2][:(num_induc*inputDim),:num_induc])
            
            for l,subSeg in enumerate(PEsegmentation):
                subInputDim = 3*len(subSeg)
                induc_induc_covar.append(full_covar[forceBlockOffset+l][:(num_induc*subInputDim),:(num_induc*subInputDim)])
                
            # add jitter to all block-diagonal tensors, and enforce symmetry
            for l,covPart in enumerate(induc_induc_covar):
                if self.model.covar_module.base_kernel.onlyForces or (l != 1 and l != 2):
                    covPart += torch.diag(torch.tensor(1e-3, dtype=covPart.dtype, device=covPart.device).repeat(covPart.shape[0]))
                    covPart = (covPart + covPart.transpose(-2,-1)) / 2
            
            L = []
            # add force-group cholesky factors
            for l,subInx in enumerate(PEsegmentation):
                Lf = TriangularLazyTensor(psd_safe_cholesky(delazify(induc_induc_covar[forceBlockOffset+l]).type(_linalg_dtype_cholesky.value()), max_tries=100))
                L.append(Lf)
                
            if not self.model.covar_module.base_kernel.onlyForces:
                # construct schur complement for energies
                T = torch.zeros_like(induc_induc_covar[2])
                currentInx = 0
                for l,Lf in enumerate(L):
                    lastInx = currentInx
                    blockSize = Lf.shape[0]
                    currentInx += blockSize
                    T[lastInx:currentInx] = Lf.inv_matmul(induc_induc_covar[2][lastInx:currentInx].type(_linalg_dtype_cholesky.value()))
                S = induc_induc_covar[0] - T.transpose(-1, -2) @ T
                Ls = TriangularLazyTensor(psd_safe_cholesky(delazify(S).type(_linalg_dtype_cholesky.value()), max_tries=100))
                    
                L = [Ls, torch.tensor(0), T.transpose(-1, -2)] + L
            
            # try to cache stuff in evaluation mode
            #    if not self.model.training:
            #        # save this L, it can be reused in evaluation mode
            #        self.L = L
            #else:
            #    L = self.L
            
            if labelsOnly:
                test_mean = full_output.mean[..., num_induc_in_mean:, [0]]
                labelIndices = num_induc_in_mean + torch.arange(num_data)
                all_data_covar = [full_covar[0][..., labelIndices], full_covar[2][..., labelIndices]]
                
                induc_data_covar = []
                induc_data_covar.append(all_data_covar[0][..., :num_induc,:])
                induc_data_covar.append(all_data_covar[1][..., :(num_induc*inputDim),:])
                 
                if not predictionsOnly:
                    data_data_covar = all_data_covar[0][..., labelIndices, :]
                outputDim = 1
            else:
                test_mean = full_output.mean[..., num_induc_in_mean:, :]
                # induc_data_covar = full_covar[..., :(num_induc*outputDim), (num_induc_in_mean*outputDim):].evaluate()
                
                # get induc_data_covar part from full_covar
                induc_data_covar = []
                if not self.model.covar_module.base_kernel.onlyForces:
                    induc_data_covar.append(full_covar[0][:num_induc,num_induc_in_mean:])
                    
                    induc_data_covar.append(full_covar[1][:num_induc,(num_induc_in_mean*inputDim):])
                    
                    induc_data_covar.append(full_covar[2][:(num_induc*inputDim),num_induc_in_mean:])
                
                for l,subSeg in enumerate(PEsegmentation):
                    subInputDim = 3*len(subSeg)
                    induc_data_covar.append(full_covar[forceBlockOffset+l][:(num_induc*subInputDim),(num_induc_in_mean*subInputDim):])
                
                if not predictionsOnly:
                    data_data_covar = []
                    if not self.model.covar_module.base_kernel.onlyForces:
                        data_data_covar.append(full_covar[0][num_induc:,num_induc_in_mean:])
                        
                        data_data_covar.append(full_covar[1][num_induc:,(num_induc_in_mean*inputDim):])
                        
                        data_data_covar.append(full_covar[2][(num_induc*inputDim):,num_induc_in_mean:])
                    
                    for l,subSeg in enumerate(PEsegmentation):
                        subInputDim = 3*len(subSeg)
                        data_data_covar.append(full_covar[forceBlockOffset+l][(num_induc*subInputDim):,(num_induc_in_mean*subInputDim):])
                    
            # Compute the mean of q(f)
            # k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X

            # Compute interpolation terms
            # K_ZZ^{-1/2} K_ZX
            # K_ZZ^{-1/2} \mu_Z
            
            # interp_term = L.inv_matmul(induc_data_covar.type(_linalg_dtype_cholesky.value())).to(full_inputs.dtype)
            interp_term = []
            # solve the force blocks individually
            for l,subInx in enumerate(PEsegmentation):
                interp_term.append(L[forceBlockOffset+l].inv_matmul(induc_data_covar[forceBlockOffset+l].type(_linalg_dtype_cholesky.value())).to(full_inputs.dtype))
                
            if not self.model.covar_module.base_kernel.onlyForces:
                interp_term_FE = []
                currentInx = 0
                for l,subSeg in enumerate(PEsegmentation):
                    lastInx = currentInx
                    subInx = ((torch.tensor(subSeg)*3).reshape(-1,1) + torch.tensor([0,1,2])).flatten()
                    subInputDim = len(subInx)
                    currentInx += subInputDim*num_induc
                    interp_term_FE.append(L[forceBlockOffset+l].inv_matmul(induc_data_covar[2][lastInx:currentInx].type(_linalg_dtype_cholesky.value())).to(full_inputs.dtype))
                interp_term_FE = torch.vstack(interp_term_FE)
                
                # calculate T^tr interp_term_F
                TtopInterp_term_F = []
                currentInx = 0
                for l,subSeg in enumerate(PEsegmentation):
                    lastInx = currentInx
                    subInx = ((torch.tensor(subSeg)*3).reshape(-1,1) + torch.tensor([0,1,2])).flatten()
                    subInputDim = len(subInx)
                    currentInx += subInputDim*num_induc
                    TtopInterp_term_F.append(L[2][..., lastInx:currentInx] @ interp_term[l])
                TtopInterp_term_F.append(L[2] @ interp_term_FE)    
                TtopInterp_term_F = torch.hstack(TtopInterp_term_F)
                
                interp_term_EF_EE = L[0].inv_matmul((torch.hstack([induc_data_covar[1], induc_data_covar[0]]) - TtopInterp_term_F).type(_linalg_dtype_cholesky.value())).to(full_inputs.dtype)
                
                interp_term = [interp_term_EF_EE[..., (inputDim*num_data):], interp_term_EF_EE[..., :(inputDim*num_data)], interp_term_FE] + interp_term
            
            if self.model.covar_module.base_kernel.onlyForces:
                flattenedInducingValues = torch.flatten(inducing_values,-2).unsqueeze(-1)
                predictive_mean = []
                for l,subInx in enumerate(groupwiseForceInducInxes):
                    predictive_mean.append(interp_term[forceBlockOffset+l].transpose(-1, -2) @ flattenedInducingValues[subInx])
                predictive_mean = torch.vstack(predictive_mean)
                predictive_mean = predictive_mean[revertDataOrder]
            
                predictive_mean = predictive_mean.squeeze(-1).reshape(-1, outputDim) + test_mean
                if predictionsOnly:
                    #return predictive_mean
                    fakeCovar = DiagLazyTensor(torch.ones_like(predictive_mean))
                    return MultivariateNormal(predictive_mean, fakeCovar)
                else:
                    raise(NotImplementedError)
                
            
            # predictive_mean = (interp_term.transpose(-1, -2) @ torch.flatten(inducing_values,-2).unsqueeze(-1)).squeeze(-1).reshape(-1, outputDim) + test_mean
            # need to take care, as interp_term_FF is only given in sparse/diag form
            flattenedInducingValues = torch.flatten(inducing_values,-2).unsqueeze(-1)
            # interp_term_FF is sparse size (27n) x (27m), interp_term_FE is dense size: (27n) x m, interp_term_EF_EE is dense size n x (28m)
            
            predictive_mean = interp_term_EF_EE.transpose(-1, -2) @ flattenedInducingValues[energyInducInxes]
            currentInx = 0
            for l,subInx in enumerate(groupwiseForceInducInxes):
                lastInx = currentInx
                currentInx += len(groupwiseForceDataInxes[l])
                predictive_mean[lastInx:currentInx] += interp_term[forceBlockOffset+l].transpose(-1, -2) @ flattenedInducingValues[subInx]
            predictive_mean[currentInx:] = interp_term[2].transpose(-1, -2) @ flattenedInducingValues[forceInducInxes]

            predictive_mean = predictive_mean[revertDataOrder]
            
            predictive_mean = predictive_mean.squeeze(-1).reshape(-1, outputDim) + test_mean
            
        if predictionsOnly:
            #return predictive_mean
            fakeCovar = DiagLazyTensor(torch.ones_like(predictive_mean))
            return MultivariateNormal(predictive_mean, fakeCovar)
            
        if not PEsegmentation is None:
            raise(NotImplementedError)
        # problem: data_data_covar is given in sparse form, suggesting to construct predictive_covar also sparse. But then we would need to extend the MultivariateNormal classes to deal with this case.
            
        # Compute the covariance of q(f)
        # K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
        middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1)
        if variational_inducing_covar is not None:
            middle_term = SumLazyTensor(variational_inducing_covar, middle_term)

        if trace_mode.on():
            predictive_covar = (
                data_data_covar.add_jitter(1e-4).evaluate()
                + interp_term.transpose(-1, -2) @ middle_term.evaluate() @ interp_term
            )
        else:
            predictive_covar = SumLazyTensor(
                data_data_covar.add_jitter(1e-4),
                MatmulLazyTensor(interp_term.transpose(-1, -2), middle_term @ interp_term),
            )

        # Return the distribution
        return MultitaskMultivariateNormal(predictive_mean, predictive_covar)

    def __call__(self, x, prior=False, **kwargs):
        if not self.updated_strategy.item() and not prior:
            with torch.no_grad():
                # Get unwhitened p(u)
                prior_function_dist = self(self.inducing_points, prior=True)
                prior_mean = prior_function_dist.loc
                L = self._cholesky_factor(prior_function_dist.lazy_covariance_matrix.add_jitter())

                # Temporarily turn off noise that's added to the mean
                orig_mean_init_std = self._variational_distribution.mean_init_std
                self._variational_distribution.mean_init_std = 0.0

                # Change the variational parameters to be whitened
                variational_dist = self.variational_distribution
                mean_diff = (variational_dist.loc - prior_mean).unsqueeze(-1).type(_linalg_dtype_cholesky.value())
                whitened_mean = L.inv_matmul(mean_diff).squeeze(-1).to(variational_dist.loc.dtype)
                covar_root = variational_dist.lazy_covariance_matrix.root_decomposition().root.evaluate().type(_linalg_dtype_cholesky.value())
                whitened_covar = RootLazyTensor(L.inv_matmul(covar_root).to(variational_dist.loc.dtype))
                whitened_variational_distribution = variational_dist.__class__(whitened_mean, whitened_covar)
                self._variational_distribution.initialize_variational_distribution(whitened_variational_distribution)

                # Reset the random noise parameter of the model
                self._variational_distribution.mean_init_std = orig_mean_init_std

                # Reset the cache
                clear_cache_hook(self)

                # Mark that we have updated the variational strategy
                self.updated_strategy.fill_(True)

        return super().__call__(x, prior=prior, **kwargs)

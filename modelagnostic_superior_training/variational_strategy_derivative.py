#!/usr/bin/env python3
'''
This module sets up a variational Gaussian process that models derivative information.

It extends the GPyTorch library.

Details on this library can be found in "GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration" by (Gardner et. al, 2018)
https://proceedings.neurips.cc/paper/2018/file/27e8e17134dd7083b050476733207ea1-Paper.pdf
'''

import warnings
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from linear_operator import to_dense
from linear_operator.operators import (
    CholLinearOperator,
    DiagLinearOperator,
    LinearOperator,
    MatmulLinearOperator,
    RootLinearOperator,
    SumLinearOperator,
    TriangularLinearOperator,
)
from linear_operator.utils.cholesky import psd_safe_cholesky
from linear_operator.utils.errors import NotPSDError
from torch import Tensor

from gpytorch.variational._variational_strategy import _VariationalStrategy
from gpytorch.variational.cholesky_variational_distribution import CholeskyVariationalDistribution

from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.models import ApproximateGP
from gpytorch.settings import _linalg_dtype_cholesky, trace_mode
from gpytorch.utils.errors import CachingError
from gpytorch.utils.memoize import cached, clear_cache_hook, pop_from_cache_ignore_args, _is_in_cache_ignore_args # TODO added by DANNY
from gpytorch.utils.warnings import OldVersionWarning
from gpytorch.variational import _VariationalDistribution


def _ensure_updated_strategy_flag_set(
    state_dict: Dict[str, Tensor],
    prefix: str,
    local_metadata: Dict[str, Any],
    strict: bool,
    missing_keys: Iterable[str],
    unexpected_keys: Iterable[str],
    error_msgs: Iterable[str],
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

    :param model: Model this strategy is applied to.
        Typically passed in when the VariationalStrategy is created in the
        __init__ method of the user defined model.
    :param inducing_points: Tensor containing a set of inducing
        points to use for variational inference.
    :param variational_distribution: A
        VariationalDistribution object that represents the form of the variational distribution :math:`q(\mathbf u)`
    :param learn_inducing_locations: (Default True): Whether or not
        the inducing point locations :math:`\mathbf Z` should be learned (i.e. are they
        parameters of the model).
    :param jitter_val: Amount of diagonal jitter to add for Cholesky factorization numerical stability

    .. _Hensman et al. (2015):
        http://proceedings.mlr.press/v38/hensman15.pdf
    .. _Matthews (2017):
        https://www.repository.cam.ac.uk/handle/1810/278022
    """
    
    def __init__(
        self,
        model: ApproximateGP,
        inducing_points: Tensor,
        variational_distribution: _VariationalDistribution,
        learn_inducing_locations: bool = True,
        jitter_val: Optional[float] = None,
    ):
        super().__init__(
            model, inducing_points, variational_distribution, learn_inducing_locations, jitter_val=jitter_val
        )
        self.register_buffer("updated_strategy", torch.tensor(True))
        self._register_load_state_dict_pre_hook(_ensure_updated_strategy_flag_set)
        self.has_fantasy_strategy = True
        
    @cached(name="cholesky_factor", ignore_args=True)
    def _cholesky_factor(self, induc_induc_covar: LinearOperator, jitter = None) -> TriangularLinearOperator: # TODO changed by DANNY
        L = psd_safe_cholesky(to_dense(induc_induc_covar).type(_linalg_dtype_cholesky.value()), jitter = jitter, max_tries=100) # TODO changed by DANNY
        return TriangularLinearOperator(L)

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self) -> MultitaskMultivariateNormal:
        zeros = torch.zeros(
            self._variational_distribution.shape(),
            dtype=self._variational_distribution.dtype,
            device=self._variational_distribution.device,
        )
        ones = torch.ones_like(zeros.flatten(-2))
        res = MultitaskMultivariateNormal(zeros, DiagLinearOperator(ones))
        return res
    
    @property
    @cached(name="pseudo_points_memo")
    def pseudo_points(self) -> Tuple[Tensor, Tensor]:
        # TODO: have var_mean, var_cov come from a method of _variational_distribution
        # while having Kmm_root be a root decomposition to enable CIQVariationalDistribution support.

        # retrieve the variational mean, m and covariance matrix, S.
        if not isinstance(self._variational_distribution, CholeskyVariationalDistribution):
            raise NotImplementedError(
                "Only CholeskyVariationalDistribution has pseudo-point support currently, ",
                "but your _variational_distribution is a ",
                self._variational_distribution.__name__,
            )

        var_cov_root = TriangularLinearOperator(self._variational_distribution.chol_variational_covar)
        var_cov = CholLinearOperator(var_cov_root)
        var_mean = self.variational_distribution.mean
        if var_mean.shape[-1] != 1:
            var_mean = var_mean.unsqueeze(-1)

        # compute R = I - S
        cov_diff = var_cov.add_jitter(-1.0)
        cov_diff = -1.0 * cov_diff

        # K^{1/2}
        Kmm = self.model.covar_module(self.inducing_points)
        Kmm_root = Kmm.cholesky()

        # D_a = (S^{-1} - K^{-1})^{-1} = S + S R^{-1} S
        # note that in the whitened case R = I - S, unwhitened R = K - S
        # we compute (R R^{T})^{-1} R^T S for stability reasons as R is probably not PSD.
        eval_var_cov = var_cov.to_dense()
        eval_rhs = cov_diff.transpose(-1, -2).matmul(eval_var_cov)
        inner_term = cov_diff.matmul(cov_diff.transpose(-1, -2))
        # TODO: flag the jitter here
        inner_solve = inner_term.add_jitter(self.jitter_val).solve(eval_rhs, eval_var_cov.transpose(-1, -2))
        inducing_covar = var_cov + inner_solve

        inducing_covar = Kmm_root.matmul(inducing_covar).matmul(Kmm_root.transpose(-1, -2))

        # mean term: D_a S^{-1} m
        # unwhitened: (S - S R^{-1} S) S^{-1} m = (I - S R^{-1}) m
        rhs = cov_diff.transpose(-1, -2).matmul(var_mean)
        # TODO: this jitter too
        inner_rhs_mean_solve = inner_term.add_jitter(self.jitter_val).solve(rhs)
        pseudo_target_mean = Kmm_root.matmul(inner_rhs_mean_solve)

        # ensure inducing covar is psd
        # TODO: make this be an explicit root decomposition
        try:
            pseudo_target_covar = CholLinearOperator(inducing_covar.add_jitter(self.jitter_val).cholesky()).to_dense()
        except NotPSDError:
            from linear_operator.operators import DiagLinearOperator

            evals, evecs = torch.linalg.eigh(inducing_covar)
            pseudo_target_covar = (
                evecs.matmul(DiagLinearOperator(evals + self.jitter_val)).matmul(evecs.transpose(-1, -2)).to_dense()
            )

        return pseudo_target_covar, pseudo_target_mean    

    def forward(
        self,
        x: Tensor,
        inducing_points: Tensor,
        inducing_values: Tensor,
        variational_inducing_covar: Optional[LinearOperator] = None,
        **kwargs,
    ) -> MultitaskMultivariateNormal:
    
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
        outputDim = self.model.covar_module.base_kernel.num_outputs_per_input(x, None)
        # if we are in batch mode, adapt batch size accordingly
        memoryThreshold = int(2**25)
        batchThresholdSize = int(max(1, memoryThreshold // (trainSize * outputDim**2)))
            
        # idea: in predictionsOnly-mode, if the size of x exceeds a threshold (e.g. 1000), slice up x and call forward again on each slice. Then merge results
        if not self.model.training and predictionsOnly and x.shape[-2] > batchThresholdSize:
            predictive_mean = torch.empty(x.shape[:-1], outputDim)
            inx = 0
            while inx < x.shape[-2]:
                inxOld = inx
                inx += batchThresholdSize
                predictive_mean[..., inxOld:inx, :] = self.forward(x[..., inxOld:inx, :], inducing_points, inducing_values, variational_inducing_covar = variational_inducing_covar, **kwargs).mean
            fakeCovar = DiagLinearOperator(torch.ones_like(predictive_mean))
            return MultitaskMultivariateNormal(predictive_mean, fakeCovar)
            
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
        full_covar = full_output.lazy_covariance_matrix
            
        # Covariance terms
        num_induc = inducing_points.size(-2)
        num_data = x.size(-2)
        num_total = full_inputs.size(-2)
        num_induc_in_mean = num_total - num_data
        
        if not _is_in_cache_ignore_args(self, "cholesky_factor"):
            if jitter is None:
                induc_induc_covar = full_covar[..., :(num_induc*outputDim), :(num_induc*outputDim)].add_jitter()
            elif jitter > 0.:
                induc_induc_covar = full_covar[..., :(num_induc*outputDim), :(num_induc*outputDim)].add_jitter(jitter_val=jitter)
            else:
                induc_induc_covar = full_covar[..., :(num_induc*outputDim), :(num_induc*outputDim)]
                
            # I added this for enforcing symmetry
            induc_induc_covar = induc_induc_covar.to_dense()
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
                
        # Compute interpolation terms
        # K_ZZ^{-1/2} K_ZX
        # K_ZZ^{-1/2} \mu_Z
        #interp_term = L.inv_matmul(induc_data_covar.type(_linalg_dtype_cholesky.value())).to(full_inputs.dtype)
        interp_term = L.solve(induc_data_covar.type(_linalg_dtype_cholesky.value())).to(full_inputs.dtype)
        
        # Compute the mean of q(f)
        # k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X
        predictive_mean = (interp_term.transpose(-1, -2) @ torch.flatten(inducing_values,-2).unsqueeze(-1)).squeeze(-1).reshape(-1, outputDim) + test_mean
        
        if predictionsOnly:
            #return predictive_mean
            fakeCovar = DiagLinearOperator(torch.ones_like(predictive_mean))
            return MultitaskMultivariateNormal(predictive_mean, fakeCovar)

        # Compute the covariance of q(f)
        # K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
        middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1)
        if variational_inducing_covar is not None:
            middle_term = SumLinearOperator(variational_inducing_covar, middle_term)

        if trace_mode.on():
            predictive_covar = (
                data_data_covar.add_jitter(self.jitter_val).to_dense()
                + interp_term.transpose(-1, -2) @ middle_term.to_dense() @ interp_term
            )
        else:
            predictive_covar = SumLinearOperator(
                data_data_covar.add_jitter(self.jitter_val),
                MatmulLinearOperator(interp_term.transpose(-1, -2), middle_term @ interp_term),
            )

        # Return the distribution
        return MultitaskMultivariateNormal(predictive_mean, predictive_covar)

    def __call__(self, x: Tensor, prior: bool = False, **kwargs) -> MultitaskMultivariateNormal:
        
        cholJitter = None
        if 'cholJitter' in kwargs:
            cholJitter = kwargs['cholJitter']
        
        if not self.updated_strategy.item() and not prior:
            with torch.no_grad():
                # Get unwhitened p(u)
                prior_function_dist = self(self.inducing_points, prior=True)
                prior_mean = prior_function_dist.loc
                L = self._cholesky_factor(prior_function_dist.lazy_covariance_matrix.add_jitter(self.jitter_val), jitter = cholJitter)

                # Temporarily turn off noise that's added to the mean
                orig_mean_init_std = self._variational_distribution.mean_init_std
                self._variational_distribution.mean_init_std = 0.0

                # Change the variational parameters to be whitened
                variational_dist = self.variational_distribution
                if isinstance(variational_dist, MultitaskMultivariateNormal):
                    mean_diff = (variational_dist.loc - prior_mean).unsqueeze(-1).type(_linalg_dtype_cholesky.value())
                    whitened_mean = L.solve(mean_diff).squeeze(-1).to(variational_dist.loc.dtype)
                    covar_root = variational_dist.lazy_covariance_matrix.root_decomposition().root.to_dense()
                    covar_root = covar_root.type(_linalg_dtype_cholesky.value())
                    whitened_covar = RootLinearOperator(L.solve(covar_root).to(variational_dist.loc.dtype))
                    whitened_variational_distribution = variational_dist.__class__(whitened_mean, whitened_covar)
                    self._variational_distribution.initialize_variational_distribution(
                        whitened_variational_distribution
                    )

                # Reset the random noise parameter of the model
                self._variational_distribution.mean_init_std = orig_mean_init_std

                # Reset the cache
                clear_cache_hook(self)

                # Mark that we have updated the variational strategy
                self.updated_strategy.fill_(True)

        return super().__call__(x, prior=prior, **kwargs)

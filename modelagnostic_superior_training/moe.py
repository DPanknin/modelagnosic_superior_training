"""
This module contains the definition of the sparse mixture of Gaussian processes model.

This code is an adaption/extension of the "Sparsely Gated Mixture of Experts Layer" implementation by David Rau:
https://github.com/davidmrau/mixture-of-experts

Details on this layer can be found in "Outrageously Large Neural Networks" by (Shazeer et. al, 2017)
https://arxiv.org/abs/1701.06538
"""

# MIT License
#
# Copyright (c) 2023 Danny Panknin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
import gpytorch
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import DiagLinearOperator
from linear_operator import to_linear_operator

class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = sorted_experts[index_sorted_experts[:, 1],0]
        # calculate num samples that each expert gets
        self._part_sizes = list((gates > 0).sum(0).numpy())
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)


    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)#.exp()

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True).type(self._gates.dtype)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.type(self._gates.dtype))
        # add eps to all zero values in order to avoid nans when going back to log space
        #combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined#.log()


    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)




class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer, which applies softmax and a sparsification on the provided gate. Then, predictions are
    Args:
    input_size: integer - size of the input
    experts: module list of experts
    gate: the gate module
    likelihood: the shared likelihood module of the experts
    noisy_gating: a boolean - Shall noise be imposed on the gate?
    initialNoiseStdDev: The initial noise level imposed on the gate output - Increase exploration of the individual expert supports while training to not end up prematurely in a local optimum
    softK: an integer for soft sparsification - Experts that are used for prediction but whose gate-outputs do not rank under the top-(softK) values are accounted for in a pentalty term            
    sparsityK: an integer for hard sparsification - Only these experts are used for prediction, whose gate-outputs rank under the top-(sparsityK) values
    xShift_stddev: The noise level imposed on the gate-inputs - The segmentation of the input space with respect to the experts can come with spurious extrapolation behavior at the unnatural boundaries of the individual expert supports. Imposing noise on the gate-inputs extends the support of each expert while training to overcome these boundary effects.
    """

    def __init__(self, input_size, experts, gate, likelihood=None, noisy_gating=True, initialNoiseStdDev = 0.1, softK = None, sparsityK = None, xShift_stddev = 0):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = len(experts)
        self.input_size = input_size
        # softK penalizes all probability mass that is assigned to experts beyond the top-softK experts 
        if softK is None:
            self.softK = self.num_experts
        else:
            self.softK = softK
        # sparsityK cuts of the non-top-sparsityK experts; I.e. softK < sparsityK (for softK >= sparsityK the soft-penalty is 0.)
        if sparsityK is None:
            self.sparsityK = self.num_experts
        else:
            self.sparsityK = sparsityK
        # instantiate experts
        self.experts = experts
        if hasattr(experts[0], 'outputDimension'):
            self.outputDimension = experts[0].outputDimension
        else:
            self.outputDimension = 1
        self.gate = gate
        self.w_noise = nn.Parameter(torch.zeros(input_size, self.num_experts), requires_grad=noisy_gating)
        if noisy_gating:
            self.b_noise = nn.Parameter(torch.zeros(1, self.num_experts) + np.log(initialNoiseStdDev), requires_grad=noisy_gating)
        else:
            self.b_noise = nn.Parameter(torch.zeros(1, self.num_experts) - np.Inf, requires_grad=noisy_gating)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.normal = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        self.likelihood = likelihood
        self.xShift_stddev = xShift_stddev

        assert(self.sparsityK <= self.num_experts)
        assert(self.softK <= self.num_experts)
        
    def penalizeSmallBandwidths(self, load):
        """Compute a penalty-term on the use of small bandwidth experts for regularization while training - Larger bandwidth experts with (locally) comparable predictive power to small bandwidth experts should be favored
        
        """
        return(load @ torch.linspace(1,0,self.num_experts).type(self.gate.variational_strategy.base_variational_strategy.inducing_points.dtype) / load.sum()*2.)


    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch) * m + self.sparsityK
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat,0 , threshold_positions_if_out), 1)
        # is each value currently in the top k.
        prob_if_in = self.normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = self.normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, noise_epsilon=0., iw = None):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            noise_epsilon: a float
            iw: importance weights of x
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts] - quantify how much each individual expert is used in total
        """
        clean_logits = self.gate(x, predictionsOnly = True).mean
        if self.gate.training and self.noisy_gating:
            raw_noise_stddev = x @ self.w_noise + self.b_noise
            noise_stddev = torch.exp(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + ( torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        #top_logits, top_indices = logits.topk(min(self.sparsityK + 1, self.num_experts), dim=1)
        top_logits, top_indices = logits.topk(self.num_experts, dim=1)
        top_k_logits = top_logits[:, :self.sparsityK]
        top_k_indices = top_indices[:, :self.sparsityK]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
            
        if self.experts.training and self.xShift_stddev > 0:
            expertBandwidths = torch.zeros(len(self.experts))
            for i,expert in enumerate(self.experts):
                if hasattr(expert.model.covar_module.base_kernel, 'raw_lengthscale'):
                    expertBandwidths[i] = expert.model.covar_module.base_kernel.lengthscale
                else:
                    expertBandwidths[i] = expert.model.covar_module.base_kernel.base_kernel.lengthscale

            sigmaPreds = torch.exp((gates*torch.log(expertBandwidths)).sum(1)).reshape(-1,1)

            # now randomize x-positions according to sigma estimates
            clean_logits = self.gate(x + ( torch.randn_like(x) * sigmaPreds * self.xShift_stddev), predictionsOnly = True).mean
            if self.gate.training and self.noisy_gating:
                raw_noise_stddev = x @ self.w_noise + self.b_noise
                noise_stddev = torch.exp(raw_noise_stddev) + noise_epsilon
                noisy_logits = clean_logits + ( torch.randn_like(clean_logits) * noise_stddev)
                logits = noisy_logits
            else:
                logits = clean_logits
            
            # calculate topk + 1 that will be needed for the noisy gates
            #top_logits, top_indices = logits.topk(min(self.sparsityK + 1, self.num_experts), dim=1)
            top_logits, top_indices = logits.topk(self.num_experts, dim=1)
            top_k_logits = top_logits[:, :self.sparsityK]
            top_k_indices = top_indices[:, :self.sparsityK]
            top_k_gates = self.softmax(top_k_logits)
            
            zeros = torch.zeros_like(logits, requires_grad=True)
            gates = zeros.scatter(1, top_k_indices, top_k_gates)
            
            if False:
                # set all gate values > 0 to 1 / sum(gate values > 0)
                gates = (gates > 0).type(self.gate.variational_strategy.base_variational_strategy.inducing_points.dtype)
                gates = gates / gates.sum(1).reshape(-1,1)
        
        if self.gate.training:

            if iw is None:
                load = gates.sum(0)
            else:
                load = iw @ gates
        else:
            load = None
            
        return gates, load

    def forward(self, x, iw = None, returnLabelPredictions = True, returnSigmaPredictions = False, returnUncertainties = False, combine = True, precalcExpertY = None, precalcExpertCovar = None, **kwargs):
        """Args:
        x: tensor shape [batch_size, input_size]
        small_bandwidth_loss_coeff: a scalar - multiplier on usage of small bandwidth candidates, should be normalized wrt noise level
        iw: importance weights of x
        
        Returns:
        y: a tensor with shape [batch_size, output_size].
        smallBandwidth_loss: a scalar - This should be added into the overall training loss of the model.  The backpropagation of this loss encourages use of larger bandwidth experts
        """
        
        predictionsOnly = False
        if 'predictionsOnly' in kwargs:
            predictionsOnly = kwargs['predictionsOnly']
        labelsOnly = False
        if 'labelsOnly' in kwargs:
            labelsOnly = kwargs['labelsOnly']
        
        if self.gate.training:
            gates, load = self.noisy_top_k_gating(x, iw=iw)
        else:
            with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False):
                gates, load = self.noisy_top_k_gating(x, iw=iw)
        
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        batch_inx = torch.split(dispatcher._batch_index, dispatcher._part_sizes, dim=0)
        
        ret = ()
        if returnLabelPredictions:
            # get the output dimension
            if labelsOnly:
                outputDim = 1
            else:
                outputDim = self.outputDimension
        
            expert_outputs = []
            expert_output_distributions = []
            
            for i,expert in enumerate(self.experts):
                if len(batch_inx[i]) > 0:
                    if precalcExpertY is None:
                        predDist = expert(expert_inputs[i] if len(expert_inputs[i].size()) == 2 else expert_inputs[i].unsqueeze(1), **kwargs)
                        preds = predDist.mean
                        if preds.dim() == 1:
                            preds = preds.unsqueeze(-1)
                    else:
                        preds = precalcExpertY[i][batch_inx[i]]
                        if predictionsOnly:
                            covar = DiagLinearOperator(torch.ones_like(preds))
                        else:
                            covar = precalcExpertCovar[i][batch_inx[i]][:, batch_inx[i]]
                        predDist = MultivariateNormal(preds, to_linear_operator(covar))
                        if preds.dim() == 1:
                            preds = preds.unsqueeze(-1)
                else:
                    preds = torch.zeros([0,outputDim])
                    predDist = []
                expert_outputs.append(preds)
                expert_output_distributions.append(predDist)
            y = dispatcher.combine(expert_outputs)
            if y.dim() == 1:
                y = y.unsqueeze(-1)
            ret += (y,)
        else:
            ret += (None,)
            expert_output_distributions = None
            
        if not combine:
            ret += (expert_output_distributions, gates, batch_inx)
            
        if self.gate.training:
            
            ##### penalize usage of smaller bandwiths
            smallBandwidth_loss = self.penalizeSmallBandwidths(load)
        
            ret += (smallBandwidth_loss,)

        if returnSigmaPredictions:
            sigmaPreds = torch.zeros(len(x))
            for i,expert in enumerate(self.experts):
                if len(batch_inx[i]) == 0:
                    # no data assigned to this expert
                    continue
                    
                if hasattr(expert.model.covar_module.base_kernel, 'raw_lengthscale'):
                    sigmaPreds[batch_inx[i]] += (torch.log(expert.model.covar_module.base_kernel.lengthscale) * gates[i]).ravel()
                else:
                    sigmaPreds[batch_inx[i]] += (torch.log(expert.model.covar_module.base_kernel.base_kernel.lengthscale) * gates[i]).ravel()
                
            sigmaPreds = torch.exp(sigmaPreds)
        
            ret += (sigmaPreds,)
            
        if returnUncertainties:
            if not returnLabelPredictions or predictionsOnly:
                # in this case we need to calculate expert_output_distributions
                expert_output_distributions = []
                
                for i,expert in enumerate(self.experts):
                    if len(batch_inx[i]) > 0:
                        if precalcExpertY is None:
                            predDist = expert(expert_inputs[i] if len(expert_inputs[i].size()) == 2 else expert_inputs[i].unsqueeze(1), **kwargs)
                        else:
                            covar = precalcExpertCovar[i][batch_inx[i]][:, batch_inx[i]]
                            predDist = MultivariateNormal(preds, covar)
                    else:
                        predDist = []
                    expert_output_distributions.append(predDist)
                
            predictiveUncertainties = torch.zeros(len(x))
            for i,expert in enumerate(self.experts):
                if len(batch_inx[i]) == 0:
                    # no data assigned to this expert
                    continue
                predictiveUncertainties[batch_inx[i]] += expert_output_distributions[i].lazy_covariance_matrix.diag().detach() * gates[i].ravel()
            ret += (predictiveUncertainties,)
            
        return ret

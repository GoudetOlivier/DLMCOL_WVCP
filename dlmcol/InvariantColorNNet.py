import numpy as np
import torch as th
from torch import nn
from torch.nn.parameter import Parameter


def InvariantColorModule(module_class, num_color, remix):
    """Takes a module class as input and return a module class that mixes batch elements
    :args module_class: class of a th.nn.Module
    :return: batched module class
    """

    class _InvariantColorModuleClass(th.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self._ind_module = module_class(
                *args,
                **kwargs,
            )
            if remix:
                self._mean_module = module_class(*args, **kwargs)

        def forward(self, *inputs):
            if remix:
                bs = num_color
                red_inputs = [th.mean(x, dim=1, keepdim=True) for x in inputs]
                return bs * self._ind_module(*inputs) / (bs + 1) + self._mean_module(
                    *red_inputs
                ) / (bs + 1)
            return self._ind_module(*inputs)

        def reset_parameters(self):
            self._ind_module.reset_parameters()
            if remix:
                self._mean_module.reset_parameters()

    return _InvariantColorModuleClass


class InvariantBatchNorm1d(th.nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super().__init__()
        # self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(th.Tensor(1))
            self.bias = Parameter(th.Tensor(1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer("running_mean", th.zeros(1))
            self.register_buffer("running_var", th.ones(1))
            self.register_buffer("num_batches_tracked", th.tensor(0, dtype=th.long))
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            th.nn.init.uniform_(self.weight)
            th.nn.init.zeros_(self.bias)

    def forward(self, input):

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        # calculate running estimates
        # print("input.size()")
        # print(input.size())
        if self.training:
            mean = th.mean(input, [0, 1])
            # use biased var in train
            var = th.var(input, [0, 1], unbiased=False)
            # mean = th.mean(input, [0,1,2])
            # # use biased var in train
            # var = th.var(input,[0,1,2], unbiased=False)
            n = input.numel()
            with th.no_grad():
                self.running_mean = (
                    exponential_average_factor * mean
                    + (1 - exponential_average_factor) * self.running_mean
                )
                # update running_var with unbiased var
                self.running_var = (
                    exponential_average_factor * var * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_var
                )
        else:
            mean = self.running_mean
            var = self.running_var
        input = (input - mean) / (th.sqrt(var + self.eps))
        if self.affine:
            input = input * self.weight + self.bias
        return input


class InvariantColorNNet(nn.Module):
    def __init__(self, size, k, dropout, remix, layers_size):
        super().__init__()
        layers = []
        self.size = size
        self.k = k
        self.dropout = dropout
        self.remix = remix
        InvariantColorLinear = InvariantColorModule(th.nn.Linear, self.k, self.remix)
        for i in range(len(layers_size) - 1):
            layers.append(InvariantColorLinear(layers_size[i], layers_size[i + 1]))
            if i != len(layers_size) - 2:
                layers.append(InvariantBatchNorm1d(layers_size[i + 1]))
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, s):
        s = th.transpose(s, 1, 2)
        s = s.float()
        s = self.layers(s)
        return th.sum(s, 1)

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

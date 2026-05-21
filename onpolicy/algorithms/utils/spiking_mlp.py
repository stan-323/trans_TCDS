import numpy as np
import torch
import torch.nn as nn

from .util import init


def surrogate_spike(x, scale=5.0):
    hard_spike = (x >= 0).to(x.dtype)
    soft_spike = torch.sigmoid(scale * x)
    return hard_spike + soft_spike - soft_spike.detach()


class SpikingMLPBase(nn.Module):
    def __init__(self, args, obs_shape, adaptive_threshold=False):
        super(SpikingMLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size
        self.time_steps = int(getattr(args, "snn_time_steps", 16))
        self.decay = float(getattr(args, "snn_decay", 0.5))
        self.base_threshold = float(getattr(args, "snn_threshold", 1.0))
        self.threshold_beta = float(getattr(args, "snn_threshold_beta", 0.1))
        self.adaptive_threshold = adaptive_threshold
        self.log_spike_stats = getattr(args, "log_spike_stats", False)
        self._last_spike_stats = {}
        self.spike_stats = self._last_spike_stats

        if self.time_steps < 1:
            raise ValueError("snn_time_steps must be >= 1")
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("snn_decay must be in [0, 1)")
        if self.base_threshold <= 0.0:
            raise ValueError("snn_threshold must be > 0")
        if self.threshold_beta < 0.0:
            raise ValueError("snn_threshold_beta must be >= 0")

        obs_dim = int(np.prod(obs_shape))

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        gain = nn.init.calculate_gain(["tanh", "relu"][self._use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        layer_dims = [obs_dim] + [self.hidden_size] * (self._layer_N + 1)
        self.linears = nn.ModuleList(
            [
                init_(nn.Linear(layer_dims[i], layer_dims[i + 1]))
                for i in range(len(layer_dims) - 1)
            ]
        )
        self.current_norms = nn.ModuleList(
            [nn.LayerNorm(self.hidden_size) for _ in self.linears]
        )

    def get_spike_stats(self):
        return dict(self._last_spike_stats)

    def _threshold_update(self, membrane):
        if not self.adaptive_threshold:
            return membrane.new_full(membrane.shape, self.base_threshold)

        max_threshold = max(self.base_threshold * 10.0, self.base_threshold + 1.0)
        max_activity = (max_threshold - self.base_threshold) / max(
            self.threshold_beta, 1.0e-6
        )
        activity = membrane.abs().clamp(max=max_activity)
        threshold = self.base_threshold + self.threshold_beta * activity
        return threshold.clamp(min=1.0e-6, max=max_threshold)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        if self._use_feature_normalization:
            x = self.feature_norm(x)

        batch_size = x.size(0)
        membranes = [
            x.new_zeros(batch_size, self.hidden_size) for _ in range(len(self.linears))
        ]
        thresholds = [
            x.new_full((batch_size, self.hidden_size), self.base_threshold)
            for _ in range(len(self.linears))
        ]
        output_sum = x.new_zeros(batch_size, self.hidden_size)
        spike_count = x.new_tensor(0.0)
        element_count = 0
        membrane_abs_sum = x.new_tensor(0.0)
        threshold_sum = x.new_tensor(0.0)

        for _ in range(self.time_steps):
            layer_input = x
            for layer_id, (linear, current_norm) in enumerate(
                zip(self.linears, self.current_norms)
            ):
                current = current_norm(linear(layer_input))
                membrane = self.decay * membranes[layer_id] + current
                threshold = thresholds[layer_id]
                spikes = surrogate_spike(membrane - threshold)
                membranes[layer_id] = membrane * (1.0 - spikes.detach())
                thresholds[layer_id] = self._threshold_update(membranes[layer_id])

                spike_count = spike_count + spikes.detach().sum()
                element_count += spikes.numel()
                membrane_abs_sum = membrane_abs_sum + membrane.detach().abs().sum()
                threshold_sum = threshold_sum + threshold.detach().sum()

                layer_input = spikes
                if layer_id == len(self.linears) - 1:
                    output_sum = output_sum + spikes

        features = output_sum / float(self.time_steps)
        denom = max(element_count, 1)
        self._last_spike_stats = {
            "spike_rate": float((spike_count / denom).item()),
            "membrane_abs_mean": float((membrane_abs_sum / denom).item()),
            "threshold_mean": float((threshold_sum / denom).item()),
            "time_steps": self.time_steps,
        }
        self.spike_stats = self._last_spike_stats
        return features

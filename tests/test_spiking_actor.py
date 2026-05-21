from argparse import Namespace
import sys
import types

import pytest
import torch

absl = types.ModuleType("absl")
flags = types.ModuleType("absl.flags")


class _Flags:
    def __call__(self, argv):
        return argv


flags.FLAGS = _Flags()
absl.flags = flags
sys.modules.setdefault("absl", absl)
sys.modules.setdefault("absl.flags", flags)

from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.config import get_config


class Box:
    def __init__(self, shape):
        self.shape = shape


class Discrete:
    def __init__(self, n):
        self.n = n


def make_args(actor_arch):
    return Namespace(
        actor_arch=actor_arch,
        algorithm_name="mappo",
        gain=0.01,
        hidden_size=32,
        layer_N=1,
        log_spike_stats=False,
        recurrent_N=1,
        snn_decay=0.5,
        snn_threshold=1.0,
        snn_threshold_beta=0.1,
        snn_time_steps=4,
        use_ReLU=True,
        use_feature_normalization=True,
        use_naive_recurrent_policy=False,
        use_orthogonal=True,
        use_policy_active_masks=True,
        use_recurrent_policy=False,
        use_stacked_frames=False,
        stacked_frames=1,
    )


@pytest.mark.parametrize("actor_arch", ["ann", "snn_lif", "snn_at"])
def test_actor_forward_and_evaluate_actions_shapes_are_stable(actor_arch):
    torch.manual_seed(0)
    actor = R_Actor(make_args(actor_arch), Box((6,)), Discrete(5))
    obs = torch.randn(7, 6)
    rnn_states = torch.zeros(7, 1, 32)
    masks = torch.ones(7, 1)

    if actor_arch == "ann":
        assert isinstance(actor.base, MLPBase)
    else:
        assert hasattr(actor.base, "get_spike_stats")

    actions, action_log_probs, new_rnn_states = actor(
        obs, rnn_states, masks, deterministic=True
    )

    assert actions.shape == (7, 1)
    assert action_log_probs.shape == (7, 1)
    assert new_rnn_states.shape == rnn_states.shape
    assert torch.isfinite(action_log_probs).all()

    eval_log_probs, entropy = actor.evaluate_actions(obs, rnn_states, actions, masks)

    assert eval_log_probs.shape == (7, 1)
    assert entropy.ndim == 0
    assert torch.isfinite(eval_log_probs).all()
    assert torch.isfinite(entropy)

    if actor_arch != "ann":
        spike_stats = actor.base.get_spike_stats()
        assert torch.isfinite(torch.tensor(spike_stats["spike_rate"]))
        assert 0.0 <= spike_stats["spike_rate"] <= 1.0
        assert spike_stats["time_steps"] == 4
        if actor_arch == "snn_at":
            assert torch.isfinite(torch.tensor(spike_stats["threshold_mean"]))


@pytest.mark.parametrize("actor_arch", ["snn_lif", "snn_at"])
def test_snn_actor_parameters_receive_nonzero_gradients(actor_arch):
    torch.manual_seed(1)
    actor = R_Actor(make_args(actor_arch), Box((6,)), Discrete(5))
    obs = torch.randn(7, 6)
    rnn_states = torch.zeros(7, 1, 32)
    masks = torch.ones(7, 1)

    actions, _, _ = actor(obs, rnn_states, masks, deterministic=True)
    action_log_probs, entropy = actor.evaluate_actions(obs, rnn_states, actions, masks)
    loss = -(action_log_probs.mean() + 0.01 * entropy)
    loss.backward()

    snn_grads = [
        param.grad.detach().abs().sum()
        for name, param in actor.named_parameters()
        if name.startswith("base.") and param.requires_grad and param.grad is not None
    ]

    assert snn_grads
    assert torch.stack(snn_grads).sum() > 0
    assert all(torch.isfinite(grad).all() for grad in snn_grads)


def test_spiking_actor_cli_args_are_registered():
    parser = get_config()

    defaults = parser.parse_args([])
    assert defaults.actor_arch == "ann"
    assert defaults.snn_time_steps == 16
    assert defaults.snn_decay == 0.5
    assert defaults.snn_threshold == 1.0
    assert defaults.snn_threshold_beta == 0.1
    assert defaults.log_spike_stats is False

    parsed = parser.parse_args(
        [
            "--actor_arch",
            "snn_at",
            "--snn_time_steps",
            "8",
            "--snn_decay",
            "0.25",
            "--snn_threshold",
            "0.75",
            "--snn_threshold_beta",
            "0.05",
            "--log_spike_stats",
        ]
    )

    assert parsed.actor_arch == "snn_at"
    assert parsed.snn_time_steps == 8
    assert parsed.snn_decay == 0.25
    assert parsed.snn_threshold == 0.75
    assert parsed.snn_threshold_beta == 0.05
    assert parsed.log_spike_stats is True

# RTX 4090 v3 Paper Figures

These figures use only experiments whose source path contains `linux_4090_v3`.
Each condition has 5 seeds for `simple_spread` with N=3, 5, and 7 agents/landmarks.

## Key mean rewards

| Agents | ANN | SNN-LIF | SNN-AT |
|---:|---:|---:|---:|
| 3 | -189.06 | -189.07 | -186.33 |
| 5 | -404.12 | -369.81 | -399.04 |
| 7 | -687.04 | -669.06 | -632.04 |

Positive reward differences mean the spiking actor is less negative and therefore better.
The SynOps figure is a proxy visualization, not a direct hardware energy measurement.

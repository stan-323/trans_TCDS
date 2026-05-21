# Environment Setup Report

Project: `spiking_mappo_official`
Environment: `C:\Users\Lenovo\anaconda3\envs\lge_cmr`
Date: 2026-05-17

## CUDA / Torch

- Python: 3.10.20
- Torch: 2.8.0+cu128
- Torch CUDA runtime: 12.8
- CUDA available: true
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- Torch was not reinstalled or changed.

## Requested Imports

All requested imports passed after setup:

| Package | Version |
| --- | --- |
| seaborn | 0.13.2 |
| setproctitle | 1.3.7 |
| tensorboardX | 2.6.5 |
| gym | 0.17.2 |
| numpy | 2.2.6 |
| torch | 2.8.0+cu128 |
| spikingjelly | 0.0.0.0.14 |
| pandas | 2.3.3 |
| matplotlib | 3.10.9 |

Additional MPE script imports checked:

| Package | Version |
| --- | --- |
| wandb | 0.27.0 |
| imageio | 2.37.3 |

## Installs Performed

Command:

```powershell
& 'C:\Users\Lenovo\anaconda3\envs\lge_cmr\python.exe' -m pip install --upgrade-strategy only-if-needed setproctitle tensorboardX "gym==0.17.2" spikingjelly wandb
```

Direct packages installed:

- setproctitle 1.3.7
- tensorboardX 2.6.5
- gym 0.17.2
- spikingjelly 0.0.0.0.14
- wandb 0.27.0

Transitive packages installed:

- annotated-types 0.7.0
- certifi 2026.4.22
- charset_normalizer 3.4.7
- click 8.3.3
- cloudpickle 1.3.0
- future 1.0.0
- gitdb 4.0.12
- gitpython 3.1.50
- idna 3.15
- platformdirs 4.9.6
- pydantic 2.13.4
- pydantic-core 2.46.4
- pyglet 1.5.0
- requests 2.34.2
- sentry-sdk 2.60.0
- smmap 5.0.3
- typing-inspection 0.4.2
- urllib3 2.7.0

## Local Package

`onpolicy` was already installed editable from this repo:

```text
Name: onpolicy
Version: 0.1.0
Editable project location: E:\Codex_Project\<workspace>\spiking_mappo_official
```

Because the editable install was already present, `pip install -e .` was not rerun.

## Verification Commands

CUDA/import check:

```powershell
& 'C:\Users\Lenovo\anaconda3\envs\lge_cmr\python.exe' - <import-and-cuda-check-script>
```

Requested smoke import:

```powershell
& 'C:\Users\Lenovo\anaconda3\envs\lge_cmr\python.exe' -c "import onpolicy.config; import onpolicy.envs.mpe.MPE_env; import onpolicy.algorithms.r_mappo.algorithm.r_actor_critic; print('smoke imports ok')"
```

Result:

```text
smoke imports ok
```

Additional MPE entry import:

```powershell
& 'C:\Users\Lenovo\anaconda3\envs\lge_cmr\python.exe' -c "import onpolicy.scripts.train.train_mpe; print('train_mpe import ok')"
```

Result:

```text
train_mpe import ok
```

Dependency consistency:

```powershell
& 'C:\Users\Lenovo\anaconda3\envs\lge_cmr\python.exe' -m pip check
```

Result:

```text
No broken requirements found.
```

## Blockers

None found for the requested CUDA/import smoke checks.

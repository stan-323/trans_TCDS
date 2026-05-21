# Spiking-MAPPO Result Tools

Run from the repository root with the `lge_cmr` Python:

```powershell
& 'C:\Users\Lenovo\anaconda3\envs\lge_cmr\python.exe' tools\summarize_results.py --repo-root . --output results_summary.csv
& 'C:\Users\Lenovo\anaconda3\envs\lge_cmr\python.exe' tools\plot_results.py --summary results_summary.csv --output-dir plots
```

`summarize_results.py` scans official `onpolicy/scripts/results` folders and `scripts_windows/logs`.
It exits successfully with an empty header-only CSV when no runs have completed.

`plot_results.py` creates `reward_curves.png` and `synops_proxy.png` when enough completed rows are available.
If no usable rows exist yet, it exits successfully with a clear message.

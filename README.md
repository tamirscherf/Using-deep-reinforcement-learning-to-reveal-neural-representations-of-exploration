# Using deep reinforcement learning to reveal neural representations of exploration

This directory contains the analysis code for the manuscript. The fMRI data,
behavioral data, trained-model checkpoints, and other inputs are not included
and are needed to run it.

- `paths.py` — Data and cache paths used across the analysis scripts.
- `data_pipeline.py` — Load saved analysis outputs or compute them if missing.
- `model_pipeline.py` — Deep-RL actor-critic model, task environment, training loop, and routines for extracting LSTM activations and likelihoods from participant behavior.
- `behavior_utils.py` — Exploration/exploitation labeling, value binning, slopes, entropy, and shared statistics.
- `plotting_utils.py` — Figure styling, palettes, and reusable plot helpers (RDMs, PCA, glass brain).
- `task_behavior.py` — Participant behavior, model likelihood, and exploration by value.
- `lstm_representations.py` — LSTM unit selection, actor weights, latent-space PCA, model uncertainty.
- `rsa_brain.py` — Functional ROIs, model RDMs, and neural-vs-model RSA.
- `exploration_neural_dynamics.py` — Sub-RDM decomposition, vmPFC diagonal slope, exploration–neural coupling, robustness.
- `supplementary_plots.py` — Supplementary analyses and robustness checks.
- `fmri_pipeline.py` — Functional masks, first-level GLMs, per-ROI Mahalanobis RDMs, and searchlight RSA.
- `rsa_utils.py` — Shared RSA helpers (shuffle null, correlations, sub-RDM stats).
- `requirements.txt` — Python libraries used in this project.

"""Data and cache paths used across the analysis scripts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd

# Directories
BEHAVIOR_DIR = Path("data/behavior")
RSA_RESULTS_DIR = Path("data/rsa_results")
MASKS_DIR = Path("data/masks")
FMRIPREP_DIR = Path("data/fmriprep")
BETA_MAPS_DIR = Path("data/beta_maps")
LOGS_DIR = Path("logs")
SIN_FUNC_DIR = Path("sin_func")
FIGS_DIR = Path("figures")
EXTENDED_FIGS_DIR = FIGS_DIR / "extended_data"

N_VALUE_BINS = 5
NEURAL_RDM_DISTANCE = "mahalanobis"

# Raw inputs
RAW_BEHAVIOR_GAIN = BEHAVIOR_DIR / "all_subj_df_gain.pkl"
RAW_BEHAVIOR_LOSS = BEHAVIOR_DIR / "all_subj_df_loss.pkl"
RAW_PERFORMANCE_GAIN = BEHAVIOR_DIR / "overall_performance_gain.pkl"
RAW_PERFORMANCE_LOSS = BEHAVIOR_DIR / "overall_performance_loss.pkl"
BRAIN_MASK_IMG = MASKS_DIR / "brain_mask.nii.gz"
TRAINED_MODEL_NAME = "trained_model"
TRAINED_MODEL_PATH = LOGS_DIR / TRAINED_MODEL_NAME

SIN_DATA_NPY = SIN_FUNC_DIR / "sin_data.npy"
MAX_REWARD_NPY = SIN_FUNC_DIR / "max_reward_all_cmb.npy"
MIN_PUNISHMENT_NPY = SIN_FUNC_DIR / "min_punishment_all_cmb.npy"

# Behavior outputs
ALL_TRIALS_DF = BEHAVIOR_DIR / "all_trials_df.pkl"
EPISODE_SUMMARIES = BEHAVIOR_DIR / "episode_summaries.pkl"
MODEL_OUTPUTS = BEHAVIOR_DIR / "model_outputs.pkl"
LIKELIHOOD_DF = BEHAVIOR_DIR / "likelihood_df.pkl"
DF_BINS = BEHAVIOR_DIR / "df_bins.pkl"

# LSTM outputs
LSTM_ACTIVATIONS = BEHAVIOR_DIR / "lstm_activations.pkl"
UNIT_SELECTION = BEHAVIOR_DIR / "unit_selection.pkl"
COMMON_SHAPE_DF = BEHAVIOR_DIR / "common_shape_df.pkl"
LSTM_CATEGORIES = RSA_RESULTS_DIR / "lstm_categories.pkl"

# fMRI and searchlight outputs
FUNCTIONAL_MASKERS = MASKS_DIR / "functional_maskers.pkl"
SECOND_LEVEL_FB_MODULATION = MASKS_DIR / "second_level_fb_modulation.pkl"
BETA_MAPS_PKL = BETA_MAPS_DIR / "beta_maps.pkl"
BETA_RESIDUALS_PKL = BETA_MAPS_DIR / "beta_residuals.pkl"
BETA_DOF_PKL = BETA_MAPS_DIR / "beta_dof.pkl"


def neural_rdms_path(n_bins: int = N_VALUE_BINS,
                     distance: str = NEURAL_RDM_DISTANCE) -> Path:
    return RSA_RESULTS_DIR / f"neural_rdms_{distance}_bins{n_bins}.pkl"


SEARCHLIGHT_MODEL_MAPS = RSA_RESULTS_DIR / "searchlight_model_maps.pkl"
SEARCHLIGHT_SECOND_LEVEL = RSA_RESULTS_DIR / "searchlight_second_level.pkl"
SEARCHLIGHT_DIAG_SLOPE_MAPS = RSA_RESULTS_DIR / "searchlight_diag_slope_maps.pkl"
EXPLORATION_CORRELATION_IMG = RSA_RESULTS_DIR / "exploration_correlation_img.pkl"

# RSA outputs
MULTI_MODEL_DF = RSA_RESULTS_DIR / "multi_model_df.pkl"
RSA_PAIRED_TTEST_DF = RSA_RESULTS_DIR / "rsa_paired_ttest_df.pkl"

# Exploration dynamics outputs
SUB_RDM_DF = RSA_RESULTS_DIR / "sub_rdm_df.pkl"
ROBUSTNESS_DF = RSA_RESULTS_DIR / "robustness_df.pkl"

# Optional supplementary inputs
HALF_RANDOM_CORR_DF = BEHAVIOR_DIR / "half_random_corr_df.pkl"
MULTISEED_CORR_DF = RSA_RESULTS_DIR / "multiseed_corr_df.pkl"
MASK_METADATA_CSV = MASKS_DIR / "mask_metadata.csv"
RSA_BEHAVIOR_DF = RSA_RESULTS_DIR / "rsa_behavior_df.pkl"
LSTM_RSA_DF = RSA_RESULTS_DIR / "lstm_rsa_df.pkl"
PARTIAL_CORR_RESULTS_DF = RSA_RESULTS_DIR / "partial_corr_results_df.pkl"
MEAN_RDMS_LSTM_MODELS_ALL = RSA_RESULTS_DIR / "mean_rdms_lstm_models_all.pkl"


def multi_model_df_units_th_path(threshold: float) -> Path:
    return RSA_RESULTS_DIR / f"multi_model_df_units_th{threshold}.csv"


T = TypeVar("T")


def optional_load(path: Path, loader: Callable[[Path], T] | None = None) -> T | None:
    """Load *path* if it exists; otherwise return None."""
    if not path.exists():
        return None
    if loader is None:
        return pd.read_pickle(path)  # type: ignore[return-value]
    return loader(path)


def load_pickle(path: Path) -> Any:
    return pd.read_pickle(path)


def save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(obj, path)


def load_npy(path: Path) -> np.ndarray:
    return np.load(path)


def save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def load_or_compute(
    path: Path,
    compute_fn: Callable[[], T],
    *,
    load_fn: Callable[[Path], T] | None = None,
    save_fn: Callable[[Path, T], None] | None = None,
    force: bool = False,
) -> T:
    """Load a saved result or compute, save, and return it."""
    if load_fn is None:
        load_fn = load_pickle
    if save_fn is None:
        save_fn = save_pickle
    if not force and path.exists():
        return load_fn(path)
    result = compute_fn()
    save_fn(path, result)
    return result

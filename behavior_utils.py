"""Shared behavioral-analysis utilities.

Contains the definitions of exploration/exploitation, value binning, the slope
measures, entropy/uncertainty, general statistics helpers, and the routine that
assembles the trial-level DataFrame combining participant behavior with
the model's per-trial outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr, zscore
from statsmodels.stats.multitest import multipletests

# --------------------------------------------------------------------------- #
# Experiment constants
# --------------------------------------------------------------------------- #
N_SUBJ = 31
N_ARMS = 3
CONDITIONS = ["Gain", "Loss"]

# Default exploration window and number of value bins.
WINDOW_SIZE = 3
N_VALUE_BINS = 5

# Value-bin edges (points) for Gain and Loss conditions.
GAIN_BIN_RANGE = (30, 80)
LOSS_BIN_RANGE = (-70, -20)


# ========================================================================== #
# Exploration / exploitation labeling
# ========================================================================== #
def switch_with_last_fix(x: pd.Series) -> pd.Series:
    """Boolean 'switch' (action differs from next trial); last trial = False."""
    result = x != x.shift(-1)
    result.iloc[-1] = False
    return result


def resume_choice(actions: np.ndarray, window: int = WINDOW_SIZE, window_stay: int = 2) -> np.ndarray:
    """Mark 'resume' trials: switching back to a recently-used arm and staying.

    A switch that returns to an arm used within the last ``window`` trials and is
    then repeated for ``window_stay`` trials is a *resume* rather than a genuine
    exploratory switch.
    """
    resume = np.zeros(len(actions), dtype=bool)
    for i in range(1, len(actions) - window_stay + 1):
        if actions[i] != actions[i - 1]:  # a switch
            prev_window = actions[max(0, i - window):i]
            if actions[i] in prev_window and np.all(actions[i:i + window_stay] == actions[i]):
                resume[i - 1] = True
    return resume


def add_choice_type_columns(df: pd.DataFrame, window: int = WINDOW_SIZE) -> pd.DataFrame:
    """Add switch / resume / exploration / exploitation columns for both the
    participant's actions and the model's most-probable actions."""
    grp = df.groupby(["subject", "condition"])

    df["switch"] = grp["action"].transform(switch_with_last_fix)
    df["resume"] = grp["action"].transform(lambda s: resume_choice(s.values, window=window))
    df["exploration"] = df["switch"] & ~df["resume"]
    df["exploitation"] = ~df["switch"]

    # Model choice = arm with highest actor probability (1-indexed to match action).
    df["max_prob_act"] = df[[f"act_{i}_prob" for i in range(N_ARMS)]].values.argmax(axis=1) + 1
    df["model_switch"] = df.groupby(["subject", "condition"], group_keys=False).apply(
        lambda g: g.action.ne(g.max_prob_act.shift(-1).fillna(g.max_prob_act.iloc[0]))
    )
    df.loc[df.groupby(["subject", "condition"]).tail(1).index, "model_switch"] = False
    df["model_resume"] = df.groupby(["subject", "condition"])["max_prob_act"].transform(
        lambda s: resume_choice(s.values, window=window)
    )
    df["model_exploration"] = df["model_switch"] & ~df["model_resume"]
    df["model_exploitation"] = ~df["model_switch"]

    # Trial-wise agreement measures between model and participant.
    df["exploration_fit"] = df["model_exploration"] == df["exploration"]
    df["action_fit"] = df.groupby(["subject", "condition"])["action"].transform(
        lambda s: s.shift(-1) == df.loc[s.index, "max_prob_act"]
    )
    return df


# ========================================================================== #
# Value binning
# ========================================================================== #
def _fixed_edges(feedback: pd.Series, lo: float, hi: float, n_bins: int) -> list:
    """Equal-width bin edges over [lo, hi], with outer edges clamped to data."""
    inner = np.linspace(lo, hi, n_bins + 1)[1:-1]
    return [feedback.min() - 1] + list(inner) + [feedback.max() + 1]


def add_value_bins(df: pd.DataFrame, n_bins: int = N_VALUE_BINS) -> pd.DataFrame:
    """Assign each trial a fixed-width value bin (``fb_group``) per condition."""
    labels = list(range(1, n_bins + 1))
    for cond, (lo, hi) in (("Gain", GAIN_BIN_RANGE), ("Loss", LOSS_BIN_RANGE)):
        mask = df.condition == cond
        edges = _fixed_edges(df.loc[mask, "feedback"], lo, hi, n_bins)
        df.loc[mask, "fb_group"] = pd.cut(
            df.loc[mask, "feedback"], bins=edges, labels=labels, include_lowest=True
        )
    df["fb_group"] = df["fb_group"].astype(int)
    return df


def add_value_quantiles(df: pd.DataFrame, n_bins: int = N_VALUE_BINS) -> pd.DataFrame:
    """Assign each trial an equal-count value quantile per action x condition x subject.

    This matches the quantile scheme used for the RSA categories (3 actions x
    5 value bins).
    """
    df["fb_quantile"] = df.groupby(["action", "condition", "subject"])["feedback"].transform(
        lambda x: pd.qcut(x, q=n_bins, labels=False, retbins=False).to_numpy() + 1
    )
    return df


# ========================================================================== #
# Scalar behavioral measures
# ========================================================================== #
def slope_from_xy(y: np.ndarray) -> float:
    """Least-squares slope of ``y`` against 1..len(y)."""
    x = np.arange(1, len(y) + 1)
    xm, ym = x.mean(), y.mean()
    return np.dot(x - xm, y - ym) / np.dot(x - xm, x - xm)


def calc_diff_from_vec(vec, vec_len: int = 5) -> float:
    """Difference between the last and first value-bin (high minus low value)."""
    if len(vec) != vec_len:
        raise ValueError(f"Vector length {len(vec)} != expected {vec_len}.")
    return vec[4:].mean() - vec[:1].mean()


def calculate_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Row-wise Shannon entropy of an (n, k) probability array."""
    probabilities = np.clip(probabilities, 1e-10, 1.0)
    return -np.sum(probabilities * np.log(probabilities), axis=1)


def action_entropy(values: np.ndarray) -> float:
    """Entropy of the empirical action distribution in a sliding window."""
    _, counts = np.unique(values, return_counts=True)
    p = counts / len(values)
    return -np.sum(p * np.log(p))


def exploration_slope_by_episode(df_binned: pd.DataFrame, value_col: str,
                                 explore_col: str = "exploration") -> dict:
    """Per-episode slope of exploration rate across value bins/quantiles.

    ``df_binned`` is the trial DataFrame averaged within
    (subject, condition, value bin).
    """
    slopes = {cond: [] for cond in CONDITIONS}
    for cond in CONDITIONS:
        for subj in range(N_SUBJ):
            sub = df_binned[(df_binned.subject == subj) & (df_binned.condition == cond)]
            slopes[cond].append(slope_from_xy(sub.sort_values(value_col)[explore_col].values))
    return slopes


# ========================================================================== #
# Building the trial-level DataFrame
# ========================================================================== #
def build_all_trials_df(behavior_gain, behavior_loss, model_outputs,
                        window: int = WINDOW_SIZE, n_bins: int = N_VALUE_BINS) -> pd.DataFrame:
    """Combine participant behavior and model outputs into one trial-level frame.

    Parameters
    ----------
    behavior_gain, behavior_loss : list of DataFrames (per subject) with 1-indexed
        ``choice`` and ``FB`` columns.
    model_outputs : dict from :func:`model_pipeline.feed_all_subjects`, keyed by
        'Gain'/'Loss', each with 'probs' and 'critic' lists per subject.
    """
    frames = []
    for cond, behavior in (("Gain", behavior_gain), ("Loss", behavior_loss)):
        for subj in range(N_SUBJ):
            d = behavior[subj][["choice", "FB"]].rename(
                columns={"FB": "feedback", "choice": "action"}
            ).copy()
            d["trial"] = d.index.values
            d["condition"] = cond
            d["subject"] = subj
            d["critic"] = model_outputs[cond]["critic"][subj]
            for i in range(N_ARMS):
                d[f"act_{i}_prob"] = model_outputs[cond]["probs"][subj][:, i]
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    df["act_ent"] = calculate_entropy(df[[f"act_{i}_prob" for i in range(N_ARMS)]].to_numpy())
    df = add_choice_type_columns(df, window=window)
    df = add_value_bins(df, n_bins=n_bins)
    df = add_value_quantiles(df, n_bins=n_bins)
    return df


def trim_to_common_length(df: pd.DataFrame, common: int = 175) -> pd.DataFrame:
    """Keep the central ``common`` trials of every (subject, condition) episode.

    Equal trimming of the start and end removes onset/offset biases and equalizes
    episode length across participants.
    """
    out = []
    for _, d in df.groupby(["subject", "condition"]):
        rem = len(d) - common
        left = rem // 2
        out.append(d.iloc[left:len(d) - (rem - left)])
    return pd.concat(out)


# ========================================================================== #
# Statistics helpers
# ========================================================================== #
def pearson_zfiltered(x, y, z_thresh: float = 3, return_mask: bool = False):
    """Pearson correlation after removing points beyond ``z_thresh`` z-scores."""
    mask = (np.abs(zscore(x, nan_policy="omit")) < z_thresh) & \
           (np.abs(zscore(y, nan_policy="omit")) < z_thresh)
    r, p = pearsonr(np.asarray(x)[mask], np.asarray(y)[mask])
    return ((r, p), mask) if return_mask else (r, p)


def spearman_zfiltered(x, y, z_thresh: float = 3, return_mask: bool = False):
    """Spearman correlation after removing points beyond ``z_thresh`` z-scores."""
    mask = (np.abs(zscore(x, nan_policy="omit")) < z_thresh) & \
           (np.abs(zscore(y, nan_policy="omit")) < z_thresh)
    rho, p = spearmanr(np.asarray(x)[mask], np.asarray(y)[mask])
    return ((rho, p), mask) if return_mask else (rho, p)


def quadratic_fit_stats(df, x_col, y_col):
    """Quadratic OLS fit; returns (adjusted R^2, quadratic beta, quadratic p)."""
    x = df[x_col].values
    y = df[y_col].values
    X = sm.add_constant(np.column_stack([x, x ** 2]))
    model = sm.OLS(y, X).fit()
    return model.rsquared_adj, model.params[2], model.pvalues[2]


def permutation_correlation_pvalue(x, y, n_perm: int = 10000):
    """Two-sided permutation p-value for a Pearson correlation."""
    x, y = np.asarray(x), np.asarray(y)
    observed, _ = pearsonr(x, y)
    perm = np.array([pearsonr(x, np.random.permutation(y))[0] for _ in range(n_perm)])
    p = np.mean(np.abs(perm) >= np.abs(observed))
    return observed, p


def corr_by_roi(df, rois, x_col, y_col, z_thresh: float = 3, alpha: float = 0.05):
    """Per-ROI Pearson/Spearman correlation of ``x_col`` vs ``y_col`` with FDR.

    Outliers beyond ``z_thresh`` z-scores (on either variable) are removed within
    each ROI. Returns a dict of arrays keyed by ROI order.
    """
    out = {"roi": [], "pearson_r": [], "pearson_p": [], "spearman_rho": [], "spearman_p": []}
    for roi in rois:
        d = df[df.roi == roi]
        (r, p), mask = pearson_zfiltered(d[x_col], d[y_col], z_thresh, return_mask=True)
        rho, sp = spearmanr(d[x_col][mask], d[y_col][mask])
        out["roi"].append(roi)
        out["pearson_r"].append(r)
        out["pearson_p"].append(p)
        out["spearman_rho"].append(rho)
        out["spearman_p"].append(sp)
    out["pearson_p_fdr"] = multipletests(out["pearson_p"], alpha=alpha, method="fdr_bh")[1]
    out["spearman_p_fdr"] = multipletests(out["spearman_p"], alpha=alpha, method="fdr_bh")[1]
    return out


def partial_correlations(df, outcome, predictors):
    """Partial correlation of each predictor with ``outcome``, controlling for the others."""
    rows = []
    for var in predictors:
        covars = [v for v in predictors if v != var]
        formula = " + ".join(covars)
        y_resid = smf.ols(f"{outcome} ~ {formula}", data=df).fit().resid
        x_resid = smf.ols(f"{var} ~ {formula}", data=df).fit().resid
        r, p = pearsonr(x_resid, y_resid)
        rows.append({"predictor": var, "partial_r": r, "p_value": p})
    return pd.DataFrame(rows)

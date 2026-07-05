"""LSTM unit selection and latent-space structure.

Unit consistency diagnostics, actor weights, PCA of selected representations,
and model uncertainty across exploration and value.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from scipy.stats import ttest_rel
from sklearn.decomposition import PCA

import behavior_utils as bu
import model_pipeline as mp
import paths
import plotting_utils as pu

N_LSTM = mp.CFG["net"]["hidden_size"]
COMMON_TRIALS = 175
UNIT_CORR_THRESHOLD = 0.5  # consistency threshold for retaining a unit


# ========================================================================== #
# Prepare activations at a common episode length
# ========================================================================== #
def to_common_shape(activations, common=COMMON_TRIALS):
    """Trim each subject's activations to the central ``common`` trials.

    ``activations`` is a list (per subject) of (n_trials x n_units) arrays.
    """
    out = np.zeros((len(activations), common, N_LSTM))
    for subj, act in enumerate(activations):
        rem = act.shape[0] - common
        left = rem // 2
        out[subj] = act[left:act.shape[0] - (rem - left), :]
    return out


# ========================================================================== #
# Unit selection
# ========================================================================== #
def unit_consistency(gain_act, loss_act):
    """Mean within- and across-condition consistency of every LSTM unit.

    * within-condition: mean pairwise correlation across participants of a unit's
      trial-by-trial activation, separately for Gain and Loss;
    * across-condition: mean within-subject correlation between Gain and Loss.

    Returns the per-unit mean of the two within-condition values (used for
    selection) plus the full 3-row matrix (across, gain, loss) for the heatmap.
    """
    within_gain = np.array([
        np.corrcoef(np.round(gain_act[:, :, u], 2))[np.triu_indices(gain_act.shape[0], 1)].mean()
        for u in range(N_LSTM)])
    within_loss = np.array([
        np.corrcoef(np.round(loss_act[:, :, u], 2))[np.triu_indices(loss_act.shape[0], 1)].mean()
        for u in range(N_LSTM)])
    across = np.array([
        np.mean([np.corrcoef(gain_act[s, :, u], loss_act[s, :, u])[0, 1]
                 for s in range(gain_act.shape[0])])
        for u in range(N_LSTM)])

    mean_within = np.nanmean(
        np.stack([np.nan_to_num(within_gain, nan=1), np.nan_to_num(within_loss, nan=1)]), axis=0)
    consistency_mat = np.stack([np.nan_to_num(across, nan=1),
                                np.nan_to_num(within_gain, nan=1),
                                np.nan_to_num(within_loss, nan=1)])
    return mean_within, consistency_mat


def select_units(mean_within, threshold=UNIT_CORR_THRESHOLD):
    """Retain units whose mean within-condition consistency is *below* threshold.

    Highly consistent units encode task structure common to all participants
    (e.g. trial index); the value/decision code lives in the less-consistent
    units.
    """
    return np.argwhere(mean_within < threshold)[:, 0]


def plot_unit_consistency(mean_within, consistency_mat, selected_units, threshold=UNIT_CORR_THRESHOLD):
    """Sorted mean consistency and per-unit heatmap."""
    order_desc = np.argsort(mean_within)[::-1]
    sorted_idx = np.argsort(mean_within)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 6),
                                     gridspec_kw={"width_ratios": [1, 0.45]})
    ax_l.scatter(np.arange(N_LSTM), mean_within[sorted_idx], c="gray",
                 s=25, edgecolors="black", linewidth=0.3)
    ax_l.hlines(threshold, 0, N_LSTM, color="gray", linestyle="--")
    ax_l.set(xlabel="LSTM units (sorted)", ylabel="Mean correlation",
             title="Within-condition consistency")

    sns.heatmap(consistency_mat[:, order_desc].T,
                cmap=sns.color_palette("blend:#7AB,#EDA", as_cmap=True),
                vmin=-1, vmax=1, cbar=True, ax=ax_r)
    ax_r.set(title="Consistency heatmap")
    ax_r.set_xticks([0.5, 1.5, 2.5])
    ax_r.set_xticklabels(["Across", "Gain", "Loss"], rotation=45, ha="right", weight="bold")
    ax_r.set_yticks([])
    ax_r.axhline(N_LSTM - len(selected_units), color="black", linewidth=3, linestyle="--")
    plt.tight_layout()
    return fig


# ========================================================================== #
# Actor weights
# ========================================================================== #
def plot_actor_weights(trained_model, mean_within):
    """Actor read-out weights per LSTM unit."""
    w_actor = trained_model.actor.state_dict()["actor.weight"].numpy()
    w_abs_sum = np.sum(np.abs(w_actor), axis=0)
    sorted_idx = np.argsort(mean_within)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(range(N_LSTM), w_abs_sum[sorted_idx], color="gray")
    ax.set(xlabel="LSTM unit", ylabel="Sum |actor weight|", title="Actor weights per unit")
    ax.set_xticks([])
    ax.grid(True)
    plt.tight_layout()
    return fig


# ========================================================================== #
# Example unit activations
# ========================================================================== #
def plot_example_unit_traces(gain_common, units=(34, 2)):
    """Example decision-relevant and irrelevant unit activations."""
    fig, axes = plt.subplots(len(units), 1, figsize=(9, 6), sharex=True)
    titles = ["Decision-relevant unit", "Decision-irrelevant unit"]
    for ax, unit, title in zip(axes, units, titles):
        for subj in range(gain_common.shape[0]):
            ax.plot(gain_common[subj][:, unit], color="gray", alpha=0.15, linewidth=1)
        ax.set(ylabel="Activation", title=title)
        ax.grid(True)
    axes[-1].set_xlabel("Trial")
    plt.tight_layout()
    return fig


# ========================================================================== #
# PCA of selected units
# ========================================================================== #
def plot_selected_units_pca(gain_common, selected_units, gain_trials):
    """PCA of selected units colored by value and decision."""
    pca = PCA(n_components=10)
    pca_xy = pca.fit_transform(np.concatenate(gain_common[:, :, selected_units], axis=0))

    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    pu.plot_pca_over_value(pca_xy, gain_trials.action.values,
                           gain_trials.feedback.values, ax=axes[0])
    pu.plot_pca_over_decision(pca_xy, gain_trials.action.values,
                              gain_trials.model_exploration.values, ax=axes[1])
    fig.suptitle("PCA of LSTM representations", fontsize=pu.FONT_SIZE_TITLE)
    plt.tight_layout()
    return fig


# ========================================================================== #
# Model uncertainty
# ========================================================================== #
def plot_action_uncertainty(common_df, n_bins=bu.N_VALUE_BINS):
    """Actor entropy for exploration vs exploitation and across value."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.barplot(data=common_df, x="model_exploration", y="act_ent",
                hue="model_exploration", palette=[pu.EXPLOIT_COLOR, pu.EXPLORE_COLOR],
                errorbar=("ci", 95), ax=axes[0], legend=False)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Exploitation", "Exploration"], fontsize=pu.FONT_SIZE_AXIS)
    axes[0].set(xlabel="", ylabel="Uncertainty (entropy)", title="Model uncertainty")

    paired = common_df.pivot_table(index=["subject", "condition"],
                                   columns="model_exploration", values="act_ent",
                                   aggfunc="mean").dropna()
    t, p = ttest_rel(paired[True], paired[False])
    axes[0].text(0.5, 0.95, f"t={t:.2f}, p={p:.1e}", transform=axes[0].transAxes, ha="center")

    sns.pointplot(data=common_df, x="fb_group", y="act_ent", errorbar="sd",
                  color="gray", ax=axes[1])
    axes[1].set(xlabel="", ylabel="Model uncertainty (entropy)",
                title="Uncertainty across value")
    pu.set_dual_xaxis_labels(axes[1],
                             pu.value_bin_labels(bu.GAIN_BIN_RANGE, n_bins),
                             pu.value_bin_labels(bu.LOSS_BIN_RANGE, n_bins))
    plt.tight_layout()
    return fig


# ========================================================================== #
# LSTM activations
# ========================================================================== #
def compute_lstm_activations():
    """Trim activations to a common episode length and select units."""
    import task_behavior as tb

    if paths.MODEL_OUTPUTS.exists():
        model_outputs = paths.load_pickle(paths.MODEL_OUTPUTS)
    else:
        model_outputs = tb.compute_behavior_tables()["model_outputs"]
    behavior_gain, behavior_loss = tb.load_behavior()

    gain_common = to_common_shape(model_outputs["Gain"]["lstm"])
    loss_common = to_common_shape(model_outputs["Loss"]["lstm"])
    mean_within, consistency_mat = unit_consistency(gain_common, loss_common)
    selected_units = select_units(mean_within)

    all_trials_df = bu.build_all_trials_df(behavior_gain, behavior_loss, model_outputs)
    common_df = bu.trim_to_common_length(all_trials_df, common=COMMON_TRIALS)

    activations = {
        "gain_common": gain_common,
        "loss_common": loss_common,
        "model_outputs": model_outputs,
    }
    selection = {
        "mean_within": mean_within,
        "consistency_mat": consistency_mat,
        "selected_units": selected_units,
    }
    paths.save_pickle(paths.LSTM_ACTIVATIONS, activations)
    paths.save_pickle(paths.UNIT_SELECTION, selection)
    paths.save_pickle(paths.COMMON_SHAPE_DF, common_df)

    lstm_categories = _build_lstm_categories(
        gain_common, loss_common, selected_units, common_df)
    paths.save_pickle(paths.LSTM_CATEGORIES, lstm_categories)

    return {**activations, **selection, "common_shape_df": common_df,
            "lstm_categories": lstm_categories}


def _build_lstm_categories(gain_common, loss_common, selected_units, common_df):
    """Mean per-category LSTM representation for RSA (3 arms x 5 value bins)."""
    rows = []
    for cond, common in (("Gain", gain_common), ("Loss", loss_common)):
        trials = common_df[common_df.condition == cond]
        for subj in range(bu.N_SUBJ):
            subj_trials = trials[trials.subject == subj].sort_values("trial")
            for (arm, fb_grp), grp in subj_trials.groupby(["action", "fb_group"]):
                positions = grp["trial"].to_numpy().astype(int) - 1
                vals = common[subj][positions][:, selected_units].mean(axis=0)
                rows.append({
                    "subject": subj,
                    "condition": cond,
                    "act_fb_grp": f"A{int(arm)}_fb_{int(fb_grp)}",
                    "LSTM_vals": vals,
                })
    return pd.DataFrame(rows)


# ========================================================================== #
# Runner
# ========================================================================== #
def main(force: bool = False):
    import data_pipeline as dp

    data = dp.get_lstm_activations(force=force)
    gain_common = data["gain_common"]
    loss_common = data["loss_common"]
    mean_within = data["mean_within"]
    consistency_mat = data["consistency_mat"]
    selected_units = data["selected_units"]
    common_df = data["common_shape_df"]

    trained_model, _ = mp.load_trained_model()
    gain_trials = common_df[common_df.condition == "Gain"]

    plot_example_unit_traces(gain_common)
    plot_unit_consistency(mean_within, consistency_mat, selected_units)
    plot_actor_weights(trained_model, mean_within)
    plot_selected_units_pca(gain_common, selected_units, gain_trials)
    plot_action_uncertainty(common_df)
    plt.show()


if __name__ == "__main__":
    main()

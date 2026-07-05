"""Exploration dynamics and vmPFC representational geometry.

Sub-RDM decomposition, diagonal slope of across-action dissimilarity, its coupling
to exploration slope, and robustness across analysis parameters.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, t as t_dist
from statsmodels.stats.multitest import multipletests

import behavior_utils as bu
import paths
import plotting_utils as pu
import rsa_utils as ru
from rsa_brain import N_VALUE_BINS, build_lstm_rdm, build_naive_models
from rsa_utils import ALL_ROIS, FB_ROIS


# ========================================================================== #
# ========================================================================== #
def rdms_to_long(neural_rdms, rois=ALL_ROIS):
    """Flatten the per-ROI/subject RDM dict to a long DataFrame of pairwise cells."""
    frames = []
    for cond in ["rew", "pun"]:
        for roi in rois:
            for subj, rdm in neural_rdms[cond][roi].items():
                frames.append(rdm.to_df())
    df = pd.concat(frames)
    df["cat1_cat2"] = df["act_fb_grp_1"] + "_" + df["act_fb_grp_2"]
    df["fbgrp1_fbgrp2"] = df["fb_group_1"].astype(str) + "_" + df["fb_group_2"].astype(str)
    return df


def split_within_across(rdm_long):
    """Split pairwise cells into within-action and across-action sub-RDMs.

    A cell is *within-action* when both categories share the same arm, and
    *across-action* otherwise. Returns the mean of each per
    (roi, condition, value-pair, subject).
    """
    same_action = (rdm_long["cat1_cat2"].str.extract(r"^(A\d+)_fb_\d+.*(A\d+)_fb_\d+")[0]
                   .eq(rdm_long["cat1_cat2"].str.extract(r"^(A\d+)_fb_\d+.*(A\d+)_fb_\d+")[1]))
    group_cols = ["roi", "condition", "fbgrp1_fbgrp2", "subj"]
    within = rdm_long[same_action].groupby(group_cols).mean().reset_index()
    across = rdm_long[~same_action].groupby(group_cols).mean().reset_index()
    return within, across


def plot_subrdm_correlations(sub_rdm_df):
    """Within- and across-action sub-RDM correlation to models."""
    palette = {"lstm_subj": "C0", "act_fb": "C1"}
    figs = []
    for sub_type in ("within", "across"):
        d = sub_rdm_df[(sub_rdm_df.roi.isin(FB_ROIS)) &
                       (sub_rdm_df.sub_rdm_type == sub_type)]
        g = sns.catplot(data=d, y="SUB_RDM_corr2model", x="roi", hue="model",
                        col="condition", kind="bar", palette=palette,
                        hue_order=["lstm_subj", "act_fb"], height=5, aspect=1.2)
        g.fig.suptitle(f"{sub_type.capitalize()}-action sub-RDM fit", weight="bold", y=1.02)
        stats = ru.sub_rdm_paired_ttests(sub_rdm_df, sub_type=sub_type)
        for ax in g.axes.flat:
            cond = ax.get_title().split(" = ")[-1]
            for _, row in stats.iterrows():
                ax.text(0.02, 0.95 - 0.08 * list(stats.roi).index(row.roi),
                        f"{row.roi}: p={row.p:.3g}", transform=ax.transAxes, fontsize=8)
        figs.append(g)
    return figs


def plot_across_action_rdm(across_mean, roi="vmPFC"):
    """Mean across-action RDM with diagonal highlighted."""
    mat = (across_mean[(across_mean.roi == roi)]
           .groupby(["fb_group_1", "fb_group_2"])["dissimilarity"].mean().unstack())
    mat.index = mat.index.map(lambda x: f"Value-{int(x)}")
    mat.columns = mat.columns.map(lambda x: f"Value-{int(x)}")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(mat, cmap="bone_r", square=True, linewidths=0.5, linecolor="black", ax=ax)
    for i in range(mat.shape[0]):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="red", linewidth=0.9))
    ax.set(title=f"{roi} across-action mean RDM", xlabel="", ylabel="")
    plt.tight_layout()
    return fig


# ========================================================================== #
# ========================================================================== #
def diagonal_slope_by_episode(across_mean, rois=ALL_ROIS):
    """Per-episode slope of the across-action RDM diagonal over value bins.

    The diagonal (same value, different arm) measures how distinctly the brain
    represents equally-valued options across arms; its slope across value bins
    is the neural separation gradient.
    """
    rows = []
    diag = across_mean[across_mean.fb_group_1 == across_mean.fb_group_2]
    for roi in rois:
        for cond in ["rew", "pun"]:
            for subj in diag[(diag.roi == roi) & (diag.condition == cond)].subj.unique():
                d = diag[(diag.roi == roi) & (diag.condition == cond) & (diag.subj == subj)]
                y = d.sort_values("fb_group_1")["dissimilarity"].values
                rows.append({"roi": roi, "condition": cond, "subject": subj,
                             "diag_slope": bu.slope_from_xy(y)})
    df = pd.DataFrame(rows)
    df["condition"] = df["condition"].map({"rew": "Gain", "pun": "Loss"})
    return df


def plot_exploration_diagonal_slope(rsa_behavior_df, roi="vmPFC"):
    """Exploration slope vs neural diagonal slope."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, cond in zip(axes, ["Gain", "Loss"]):
        d = rsa_behavior_df[(rsa_behavior_df.roi == roi) &
                            (rsa_behavior_df.condition == cond) &
                            (rsa_behavior_df.exploration_slope < 0)]
        sns.regplot(data=d, x="exploration_slope", y="diag_slope", ax=ax)
        r, p = pearsonr(d["exploration_slope"], d["diag_slope"])
        ax.set_title(f"{cond}: r={r:.2f}, p={p:.1e}", fontsize=pu.FONT_SIZE_AXIS)
        ax.set(xlabel="Exploration slope", ylabel="Diagonal slope" if cond == "Gain" else "")
    fig.suptitle(f"{roi}: exploration vs neural-separation slope", fontsize=pu.FONT_SIZE_TITLE)
    plt.tight_layout()
    return fig


# ========================================================================== #
# ========================================================================== #
def significance_threshold_r(n=60, alpha=0.05):
    """Critical (uncorrected) |r| for two-tailed significance at sample size n."""
    t_crit = t_dist.ppf(1 - alpha / 2, df=n - 2)
    return (t_crit ** 2 / (t_crit ** 2 + n - 2)) ** 0.5


def robustness_across_params(all_trials_df, diag_slope_by_bins, windows, bin_counts):
    """Recompute the exploration-slope / diagonal-slope correlation while varying
    the exploration window and the number of value bins.

    ``diag_slope_by_bins`` maps a bin count to a diagonal-slope DataFrame (as from
    :func:`diagonal_slope_by_episode`) computed on RDMs with that many bins.
    """
    rows = []
    for window in windows:
        df_w = all_trials_df.copy()
        df_w = bu.add_choice_type_columns(df_w, window=window)
        for n_bins in bin_counts:
            df_w = bu.add_value_quantiles(df_w, n_bins=n_bins)
            binned = df_w.groupby(["subject", "condition", "fb_quantile"]).mean().reset_index()
            slopes = bu.exploration_slope_by_episode(binned, "fb_quantile", "exploration")
            diag = diag_slope_by_bins[n_bins].copy()
            diag["exploration_slope"] = diag.apply(
                lambda r: slopes[r.condition][r.subject], axis=1)
            p_list = []
            for roi in ALL_ROIS:
                d = diag[diag.roi == roi]
                (r, p), _ = bu.pearson_zfiltered(d["exploration_slope"], d["diag_slope"],
                                                 return_mask=True)
                rows.append({"window": window, "num_fb_grps": n_bins, "roi": roi,
                             "r_pearson": r, "p_pearson": p})
                p_list.append(p)
            fdr = multipletests(p_list, method="fdr_bh")[1]
            for i, roi in enumerate(ALL_ROIS):
                rows[-(len(ALL_ROIS) - i)]["p_pearson_fdr"] = fdr[i]
    return pd.DataFrame(rows)


def plot_correlation_robustness(robustness_df):
    """Correlation stability across windows and value-bin counts."""
    r_thresh = significance_threshold_r()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, xvar, xlabel in zip(axes, ["window", "num_fb_grps"],
                                ["Exploration window", "Number of value bins"]):
        sns.lineplot(data=robustness_df[robustness_df.roi.isin(FB_ROIS)], x=xvar, y="r_pearson",
                     hue="roi", hue_order=FB_ROIS,
                     palette={r: pu.ROI_COLORS[r] for r in FB_ROIS},
                     errorbar="sd", marker="o", ax=ax)
        ax.axhline(-r_thresh, color="gray", linestyle="dashed")
        ax.set(xlabel=xlabel, ylabel="Pearson correlation")
    fig.suptitle("Robustness of the exploration-slope / neural-slope correlation",
                 fontsize=pu.FONT_SIZE_TITLE)
    plt.tight_layout()
    return fig


# ========================================================================== #
# ========================================================================== #
def plot_exploration_separation_schematic():
    """Schematic linking exploration rate and neural separation."""
    def sigmoid(x, k=8, x0=0.5):
        return 1 / (1 + np.exp(-k * (x - x0)))

    x = np.linspace(0, 1, 100)
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(x, 1 - sigmoid(x, k=6), color="gray", label="Exploration rate")
    ax1.set(ylim=(0, 1), yticks=[0, 0.5, 1], xticks=[0, 0.5, 1],
            xlabel="Value")
    ax1.set_yticklabels(["Low", "Mid", "High"])
    ax1.set_xticklabels(["Low", "Mid", "High"])

    ax2 = ax1.twinx()
    ax2.plot(x, sigmoid(x, k=9), color="gray", linestyle="--", label="vmPFC neural separation")
    ax2.set(ylim=(0, 1), yticks=[0, 0.5, 1])
    ax2.set_yticklabels(["Low", "Mid", "High"])

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper center")
    plt.tight_layout()
    return fig


# ========================================================================== #
# Sub-RDM table
# ========================================================================== #
def compute_sub_rdm_df():
    """Correlate neural within/across sub-RDMs with deep-RL and Action-Value models."""
    import data_pipeline as dp

    neural_rdms = dp.get_neural_rdms()
    if not paths.LSTM_CATEGORIES.exists():
        raise FileNotFoundError(f"Required file not found: {paths.LSTM_CATEGORIES}")
    naive_models, _ = build_naive_models()
    lstm_rdm_dict = build_lstm_rdm(paths.load_pickle(paths.LSTM_CATEGORIES))
    lstm_rew_pun = {"rew": lstm_rdm_dict.get("Gain", {}),
                    "pun": lstm_rdm_dict.get("Loss", {})}
    sub_rdm_df = ru.build_sub_rdm_correlations(
        neural_rdms, lstm_rew_pun, naive_models["act_fb"])
    paths.save_pickle(paths.SUB_RDM_DF, sub_rdm_df)
    return sub_rdm_df


# ========================================================================== #
# Runner
# ========================================================================== #
def main(force: bool = False):
    import data_pipeline as dp

    neural_rdms = dp.get_neural_rdms(force=force)
    rdm_long = rdms_to_long(neural_rdms)
    within_mean, across_mean = split_within_across(rdm_long)
    diag_slope_df = diagonal_slope_by_episode(across_mean)

    try:
        sub_rdm_df = dp.get_sub_rdm_df(force=force)
    except FileNotFoundError:
        sub_rdm_df = None

    episode_df = dp.get_episode_summaries(force=force)
    rsa_behavior_df = diag_slope_df.merge(
        episode_df[["subject", "condition", "exploration_slope"]],
        on=["subject", "condition"], how="left")

    if sub_rdm_df is not None:
        plot_subrdm_correlations(sub_rdm_df)
    plot_across_action_rdm(across_mean, roi="vmPFC")
    plot_exploration_diagonal_slope(rsa_behavior_df, roi="vmPFC")

    robustness_path = paths.ROBUSTNESS_DF
    if robustness_path.exists():
        plot_correlation_robustness(pd.read_pickle(robustness_path))

    plot_exploration_separation_schematic()
    plt.show()


if __name__ == "__main__":
    main()

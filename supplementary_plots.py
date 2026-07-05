"""Supplementary analyses and robustness checks.

Gain/Loss comparisons, exploration-window sensitivity, model validation, unit-selection
thresholds, additional PCA/RSA views, searchlight overlays, multiseed RSA, and
behavior–neural coupling controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, ttest_ind, ttest_rel

import behavior_utils as bu
import data_pipeline as dp
import paths
import plotting_utils as pu
import rsa_utils as ru
from rsa_brain import MODEL_LABELS, MODEL_ORDER, plot_rsa_correlations

FB_ROIS = ru.FB_ROIS
ALL_ROIS = ru.ALL_ROIS
N_VALUE_BINS = ru.N_VALUE_BINS
UNIT_SELECTION_THRESHOLDS = (0.2, 0.5, 0.8)


@dataclass
class SupplementaryData:
    """Loaded supplementary-analysis inputs."""

    episode_df: pd.DataFrame | None = None
    all_trials_df: pd.DataFrame | None = None
    half_random_corr_df: pd.DataFrame | None = None
    gain_loss_corr_df: pd.DataFrame | None = None
    per_trial_df: pd.DataFrame | None = None
    performance_df: pd.DataFrame | None = None
    critic_df: pd.DataFrame | None = None
    multi_model_df: pd.DataFrame | None = None
    multi_model_df_by_thresh: dict[float, pd.DataFrame] = field(default_factory=dict)
    ttest_df: pd.DataFrame | None = None
    pca_gain_xy: np.ndarray | None = None
    pca_loss_xy: np.ndarray | None = None
    explained_var_gain: np.ndarray | None = None
    explained_var_loss: np.ndarray | None = None
    common_shape_df: pd.DataFrame | None = None
    mask_metadata_df: pd.DataFrame | None = None
    second_level_fb_modulation: object = None
    searchlight_model_map: object = None
    correlation_img_th: object = None
    second_level_contrasts: pd.DataFrame | None = None
    multiseed_corr_df: pd.DataFrame | None = None
    mean_rdms_lstm_models_all: list | None = None
    rsa_behavior_df: pd.DataFrame | None = None
    lstm_rsa_df: pd.DataFrame | None = None
    partial_corr_results_df: pd.DataFrame | None = None
    correlation_img: object = None
    functional_maskers: dict | None = None
    gain_common: np.ndarray | None = None
    loss_common: np.ndarray | None = None
    mean_within: np.ndarray | None = None
    consistency_mat: np.ndarray | None = None
    selected_units: np.ndarray | None = None
    loss_trials: pd.DataFrame | None = None


def load_supplementary_data(force: bool = False) -> SupplementaryData:
    """Load supplementary inputs for plotting."""
    behavior = dp.get_behavior_tables(force=force)
    lstm = dp.get_lstm_activations(force=force)

    sl = None
    try:
        sl = dp.get_searchlight_bundle(force=force)
    except (FileNotFoundError, ValueError):
        pass

    data = SupplementaryData(
        episode_df=behavior["episode_df"],
        all_trials_df=behavior["all_trials_df"],
        half_random_corr_df=paths.optional_load(paths.HALF_RANDOM_CORR_DF),
        multi_model_df=paths.optional_load(paths.MULTI_MODEL_DF),
        ttest_df=paths.optional_load(paths.RSA_PAIRED_TTEST_DF),
        common_shape_df=lstm.get("common_shape_df"),
        mask_metadata_df=paths.optional_load(
            paths.MASK_METADATA_CSV, lambda p: pd.read_csv(p, index_col=0)),
        second_level_fb_modulation=paths.optional_load(paths.SECOND_LEVEL_FB_MODULATION),
        searchlight_model_map=sl["searchlight_model_map"] if sl else None,
        second_level_contrasts=sl["second_level_contrasts"] if sl else None,
        multiseed_corr_df=paths.optional_load(paths.MULTISEED_CORR_DF),
        mean_rdms_lstm_models_all=paths.optional_load(paths.MEAN_RDMS_LSTM_MODELS_ALL),
        rsa_behavior_df=paths.optional_load(paths.RSA_BEHAVIOR_DF),
        lstm_rsa_df=paths.optional_load(paths.LSTM_RSA_DF),
        partial_corr_results_df=paths.optional_load(paths.PARTIAL_CORR_RESULTS_DF),
        correlation_img=sl["correlation_img"] if sl else None,
        functional_maskers=paths.optional_load(paths.FUNCTIONAL_MASKERS),
        gain_common=lstm.get("gain_common"),
        loss_common=lstm.get("loss_common"),
        mean_within=lstm.get("mean_within"),
        consistency_mat=lstm.get("consistency_mat"),
        selected_units=lstm.get("selected_units"),
    )

    for th in UNIT_SELECTION_THRESHOLDS:
        df_th = paths.optional_load(
            paths.multi_model_df_units_th_path(th),
            lambda p: pd.read_csv(p, index_col=0),
        )
        if df_th is not None:
            data.multi_model_df_by_thresh[th] = df_th

    if data.multi_model_df is None:
        try:
            data.multi_model_df = dp.get_multi_model_df(force=force)
            data.ttest_df = dp.get_rsa_ttest_df(force=force)
        except (FileNotFoundError, ValueError):
            pass

    if data.per_trial_df is None and data.all_trials_df is not None:
        data.per_trial_df = _prepare_per_trial_df(data.all_trials_df)

    if data.critic_df is None and data.all_trials_df is not None:
        if "critic" in data.all_trials_df.columns and "fb_group" in data.all_trials_df.columns:
            critic_src = data.all_trials_df.copy()
            if "fb_group_new" not in critic_src.columns:
                critic_src["fb_group_new"] = critic_src["fb_group"]
            data.critic_df = (critic_src.groupby(["condition", "fb_group_new"])
                              .agg(critic=("critic", "mean")).reset_index())

    if data.common_shape_df is not None:
        data.loss_trials = data.common_shape_df[data.common_shape_df.condition == "Loss"]

    if data.gain_loss_corr_df is None and data.half_random_corr_df is not None:
        data.gain_loss_corr_df = compute_gain_loss_correlations(data.half_random_corr_df)

    return data


def _require_df(df, columns: list[str] | None = None) -> bool:
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return False
    if columns and any(c not in df.columns for c in columns):
        return False
    return True


def _prepare_per_trial_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-arm model/subject columns for choice plots."""
    out = df.copy()
    if "model_action" not in out.columns and "action" in out.columns:
        out["model_action"] = out["action"]
    if "subj_action" not in out.columns:
        if "choice" in out.columns:
            out["subj_action"] = out["choice"]
        elif "action" in out.columns:
            out["subj_action"] = out["action"]
    if "model_action" not in out.columns and "max_prob_act" in out.columns:
        out["model_action"] = out["max_prob_act"]
    for i in (1, 2, 3):
        if f"arm_{i}_model" not in out.columns and "model_action" in out.columns:
            out[f"arm_{i}_model"] = (out["model_action"] == i).astype(int)
        if f"arm_{i}_subj" not in out.columns and "subj_action" in out.columns:
            out[f"arm_{i}_subj"] = (out["subj_action"] == i).astype(int)
    if "trial" not in out.columns:
        out["trial"] = out.groupby(["subject", "episode"]).cumcount() + 1
    return out


def _enrich_episode_df(episode_df: pd.DataFrame, all_trials_df: pd.DataFrame | None) -> pd.DataFrame:
    """Add exploration_slope_fb_grp to episode summaries when absent."""
    ep = episode_df.copy()
    if "exploration_slope_fb_grp" in ep.columns or all_trials_df is None:
        return ep
    df = all_trials_df.copy()
    if "fb_group" not in df.columns:
        df = bu.add_value_bins(df)
    binned = df.groupby(["subject", "condition", "fb_group"]).mean().reset_index()
    slopes = bu.exploration_slope_by_episode(binned, "fb_group", "exploration")
    ep["exploration_slope_fb_grp"] = ep.apply(
        lambda r: slopes[r.condition][r.subject], axis=1)
    return ep


def compute_gain_loss_correlations(half_random_corr_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate half-random performance correlations."""
    rows = []
    for iter_i in half_random_corr_df["iter"].unique():
        h1_g = half_random_corr_df[
            (half_random_corr_df.iter == iter_i) & (half_random_corr_df.condition == "Gain")
            & (half_random_corr_df.half == "1")].performance.values
        h2_g = half_random_corr_df[
            (half_random_corr_df.iter == iter_i) & (half_random_corr_df.condition == "Gain")
            & (half_random_corr_df.half == "2")].performance.values
        h1_l = half_random_corr_df[
            (half_random_corr_df.iter == iter_i) & (half_random_corr_df.condition == "Loss")
            & (half_random_corr_df.half == "1")].performance.values
        h2_l = half_random_corr_df[
            (half_random_corr_df.iter == iter_i) & (half_random_corr_df.condition == "Loss")
            & (half_random_corr_df.half == "2")].performance.values
        if len(h1_g) < 2:
            continue
        for r, cond in zip(
            [pearsonr(h1_g, h1_l)[0], pearsonr(h2_g, h2_l)[0],
             pearsonr(h1_g, h2_g)[0], pearsonr(h1_l, h2_l)[0]],
            ["Gain-Loss", "Gain-Loss", "Gain-Gain", "Loss-Loss"],
        ):
            rows.append({"condition": cond, "r": r})
    return pd.DataFrame(rows)


# ========================================================================== #
# ========================================================================== #
def plot_gain_loss_comparisons(episode_df, all_trials_df=None):
    """Gain vs Loss comparison bars (likelihood, performance, exploration, slope)."""
    if not _require_df(episode_df, ["subject", "condition", "likelihood", "performance", "exploration"]):
        return None
    ep = _enrich_episode_df(episode_df, all_trials_df)
    if "exploration_slope_fb_grp" not in ep.columns:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    specs = [
        ("likelihood", "Likelihood", "Overall likelihood", None, 0.33),
        ("performance", "Overall performance", "Overall performance", (0, 1), None),
        ("exploration", "Exploration", "Exploration rate", (0, 1), None),
        ("exploration_slope_fb_grp", "Exploration slope", "Exploration slope", None, None),
    ]
    for ax, (col, ylab, title, ylim, chance) in zip(axes.flat, specs):
        pu.bar_strip_pair(ep, col, ax, ylab, title, ylim=ylim, chance=chance)
    axes[0, 0].legend(loc="lower right", fontsize=pu.FONT_SIZE_AXIS, frameon=True)
    plt.tight_layout()
    return fig


def plot_split_half_independence(gain_loss_corr_df=None, half_random_corr_df=None, *, num_iter=10000):
    """Split-half performance independence histogram."""
    if gain_loss_corr_df is None:
        if not _require_df(half_random_corr_df):
            return None
        gain_loss_corr_df = compute_gain_loss_correlations(half_random_corr_df)
    if gain_loss_corr_df is None or gain_loss_corr_df.empty:
        return None

    within = gain_loss_corr_df[gain_loss_corr_df.condition.isin(["Gain-Gain", "Loss-Loss"])].r.mean()
    between = gain_loss_corr_df[gain_loss_corr_df.condition == "Gain-Loss"].r.mean()
    print(f"Mean within-condition correlation (Gain-Gain and Loss-Loss): {within:.4f}")
    print(f"Mean between-condition correlation (Gain-Loss): {between:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {"Gain-Loss": "gray", "Gain-Gain": pu.GAIN_COLOR, "Loss-Loss": pu.LOSS_COLOR}
    sns.histplot(data=gain_loss_corr_df, x="r", bins=10, hue="condition", stat="percent",
                 palette=palette, common_norm=False, ax=ax)
    ax.set_title("Distribution of correlations between performance halves",
                 fontsize=pu.FONT_SIZE_TITLE)
    ax.set_xlabel("Correlation (r)", fontsize=pu.FONT_SIZE_AXIS)
    ax.set_ylabel("Percent", fontsize=pu.FONT_SIZE_AXIS)

    pairs = [("Gain-Gain", "Loss-Loss"), ("Gain-Gain", "Gain-Loss"), ("Loss-Loss", "Gain-Loss")]
    for a, b in pairs:
        x = gain_loss_corr_df.loc[gain_loss_corr_df.condition == a, "r"].dropna()
        y = gain_loss_corr_df.loc[gain_loss_corr_df.condition == b, "r"].dropna()
        t, p = ttest_ind(x, y, equal_var=False)
        print(f"{a} vs {b}: t={t:.2f}, p={p:.3g}, n1={len(x)}, n2={len(y)}")
    plt.tight_layout()
    return fig


# ========================================================================== #
# ========================================================================== #
def build_window_sensitivity_data(all_trials_df, episode_df, windows=range(1, 21),
                                n_bins=N_VALUE_BINS):
    """Per-window episode summaries and performance correlations."""
    df = all_trials_df.copy()
    if "fb_group" not in df.columns:
        df = bu.add_value_bins(df, n_bins=n_bins)

    perf = episode_df.set_index(["subject", "condition"])["performance"]
    rows = []
    for w in windows:
        df_w = bu.add_choice_type_columns(df.copy(), window=w)
        mean_df = (df_w.groupby(["subject", "condition"])
                   .agg(feedback=("feedback", "mean"),
                        switch=("switch", "mean"),
                        exploration=("exploration", "mean"),
                        exploitation=("exploitation", "mean"))
                   .reset_index())
        mean_df["performance"] = mean_df.apply(
            lambda r: perf.loc[(r.subject, r.condition)], axis=1)
        if "likelihood" in episode_df.columns:
            like = episode_df.set_index(["subject", "condition"])["likelihood"]
            mean_df["likelihood"] = mean_df.apply(
                lambda r: like.loc[(r.subject, r.condition)], axis=1)

        binned = df_w.groupby(["subject", "condition", "fb_group"]).mean().reset_index()
        slope_by_subj = bu.exploration_slope_by_episode(binned, "fb_group", "exploration")
        mean_df["exploration_slope_fb_grp"] = mean_df.apply(
            lambda r: slope_by_subj[r.condition][r.subject], axis=1)
        rows.append(mean_df.assign(window=w))
    multi_windows_mean_df = pd.concat(rows, ignore_index=True)

    corr_rows = []
    for w in windows:
        df_w = multi_windows_mean_df[multi_windows_mean_df.window == w]
        r2_rate, _, p_rate = bu.quadratic_fit_stats(df_w, "exploration", "performance")
        r_slope, p_slope = pearsonr(df_w["exploration_slope_fb_grp"], df_w["performance"])
        corr_rows.append({
            "window": w,
            "r_exploration_performance": r2_rate,
            "p_exploration_performance": p_rate,
            "r_exploration_slope_performance": r_slope,
            "p_exploration_slope_performance": p_slope,
        })
    return multi_windows_mean_df, pd.DataFrame(corr_rows)


def plot_exploration_by_value_window(all_trials_df, windows=range(1, 8), n_bins=N_VALUE_BINS):
    """exploration-by-value curve for a range of resume-window sizes."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for window in windows:
        df = bu.add_choice_type_columns(all_trials_df.copy(), window=window)
        if "fb_group" not in df.columns:
            df = bu.add_value_bins(df, n_bins=n_bins)
        binned = df.groupby(["subject", "condition", "fb_group"]).mean().reset_index()
        curve = binned.groupby("fb_group")["exploration"].mean()
        ax.plot(curve.index, curve.values, marker="o", label=f"W={window}")
    ax.set(xlabel="Value bin", ylabel="Exploration rate",
           title="Exploration definition across windows")
    ax.legend(title="Window")
    plt.tight_layout()
    return fig


def plot_window_sensitivity_summary(multi_windows_mean_df, corr_df_performance):
    """2x2 panel figure — rate/slope vs window and correlations to performance."""
    tick_step = max(1, int(multi_windows_mean_df.window.max() // 10))
    xticks = range(1, int(multi_windows_mean_df.window.max()) + 1, tick_step)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    sns.lineplot(data=multi_windows_mean_df, x="window", y="exploration",
                 marker="o", ax=axes[0, 0])
    axes[0, 0].set_xticks(xticks)
    axes[0, 0].set_title("Exploration rate by window size", fontsize=pu.FONT_SIZE_TITLE)
    axes[0, 0].set(xlabel="Window size", ylabel="Exploration rate")

    sns.lineplot(data=multi_windows_mean_df, x="window", y="exploration_slope_fb_grp",
                 marker="o", ax=axes[0, 1])
    axes[0, 1].set_xticks(xticks)
    axes[0, 1].set_title("Exploration slope by window size", fontsize=pu.FONT_SIZE_TITLE)
    axes[0, 1].set(xlabel="Window size", ylabel="Exploration slope")

    sns.lineplot(data=corr_df_performance, x="window", y="r_exploration_performance",
                 marker="o", ax=axes[1, 0])
    axes[1, 0].set_xticks(xticks)
    axes[1, 0].set_title("Exploration rate correlation to performance",
                         fontsize=pu.FONT_SIZE_TITLE)
    axes[1, 0].set(xlabel="Window size",
                   ylabel="Exploration rate correlation to performance")

    sns.lineplot(data=corr_df_performance, x="window",
                 y="r_exploration_slope_performance", marker="o", ax=axes[1, 1])
    axes[1, 1].set_xticks(xticks)
    axes[1, 1].set_title("Exploration slope correlation to performance",
                         fontsize=pu.FONT_SIZE_TITLE)
    axes[1, 1].set(xlabel="Window size",
                   ylabel="Exploration slope correlation to performance")

    plt.tight_layout()
    return fig


def plot_exploration_window_robustness(all_trials_df, episode_df, windows=range(1, 21),
                            value_curve_windows=range(1, 8), n_bins=N_VALUE_BINS):
    """full exploration-window robustness (value curves + 2x2 summary)."""
    if not _require_df(all_trials_df) or not _require_df(episode_df):
        return None, None, None, None
    multi_df, corr_df = build_window_sensitivity_data(
        all_trials_df, episode_df, windows=windows, n_bins=n_bins)
    fig_value = plot_exploration_by_value_window(
        all_trials_df, windows=value_curve_windows, n_bins=n_bins)
    fig_panels = plot_window_sensitivity_summary(multi_df, corr_df)
    return fig_value, fig_panels, multi_df, corr_df


# ========================================================================== #
# ========================================================================== #
def plot_arm_choice_histograms(per_trial_df):
    """model vs subject action distribution per arm."""
    per_trial_df = _prepare_per_trial_df(per_trial_df)
    cols = [f"arm_{i}_model" for i in (1, 2, 3)] + [f"arm_{i}_subj" for i in (1, 2, 3)]
    if not _require_df(per_trial_df, cols + ["trial"]):
        return None
    mean_df = per_trial_df.groupby("trial").mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for i, (arm, ax) in enumerate(zip(["Left", "Middle", "Right"], axes)):
        sns.histplot(data=mean_df, x=f"arm_{i + 1}_model", bins=30, color=pu.MODEL_COLOR,
                     alpha=0.5, label="Model", ax=ax, binrange=(0, 1))
        sns.histplot(data=mean_df, x=f"arm_{i + 1}_subj", bins=30, color=pu.SUBJ_COLOR,
                     alpha=0.5, label="Subjects", ax=ax, binrange=(0, 1))
        ax.vlines(1 / 3, ymin=0, ymax=ax.get_ylim()[1], colors="red", linestyles="dashed")
        ax.set_title(arm, fontsize=pu.FONT_SIZE_TITLE)
        ax.set_xlabel("")
        ax.legend(fontsize=pu.FONT_SIZE_AXIS)
    fig.supxlabel("Proportion of choices", fontsize=pu.FONT_SIZE_AXIS)
    axes[0].set_ylabel("Count", fontsize=pu.FONT_SIZE_AXIS)
    fig.suptitle("Model vs Subject Action Distribution per Arm",
                 fontsize=pu.FONT_SIZE_TITLE, y=1.02)
    plt.tight_layout()
    return fig


def plot_choice_rate_over_trials(per_trial_df, *, panel="both", n_iter=60):
    """choice rate over trials with 95% CI band."""
    import scipy.stats

    per_trial_df = _prepare_per_trial_df(per_trial_df)
    if not _require_df(per_trial_df):
        return None
    mean_df = per_trial_df.groupby("trial").mean(numeric_only=True).reset_index()
    chance = 1 / 3
    t_val = scipy.stats.t.ppf(0.975, df=n_iter - 1)
    ci = t_val * np.sqrt(chance * (1 - chance) / n_iter)
    arm_cols_model = [f"arm_{i}_model" for i in (1, 2, 3)]
    arm_cols_subj = [f"arm_{i}_subj" for i in (1, 2, 3)]

    if panel in ("both", "model"):
        if panel == "both":
            fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
            ax_m, ax_s = axes
        else:
            fig, ax_m = plt.subplots(figsize=(10, 5))
            ax_s = None
        for j, color in enumerate(pu.ARM_COLORS):
            sns.scatterplot(data=mean_df, x="trial", y=arm_cols_model[j],
                            ax=ax_m, color=color, label=f"Arm {j + 1}")
        ax_m.fill_between(mean_df["trial"], chance - ci, chance + ci, color="gray", alpha=0.2)
        ax_m.hlines(chance, 0, len(mean_df), colors="gray", linestyles="dashed")
        ax_m.get_legend().remove()
        ax_m.set_title("Model choice distribution", fontsize=pu.FONT_SIZE_TITLE)
        ax_m.set_ylabel("Proportion of choices", fontsize=pu.FONT_SIZE_AXIS)

    if panel in ("both", "subject"):
        if panel == "subject":
            fig, ax_s = plt.subplots(figsize=(10, 5))
        for j, color in enumerate(pu.ARM_COLORS):
            sns.scatterplot(data=mean_df, x="trial", y=arm_cols_subj[j],
                            ax=ax_s, color=color)
        ax_s.fill_between(mean_df["trial"], chance - ci, chance + ci, color="gray", alpha=0.2)
        ax_s.hlines(chance, 0, len(mean_df), colors="gray", linestyles="dashed")
        ax_s.set_title("Subjects choice distribution", fontsize=pu.FONT_SIZE_TITLE)
        ax_s.set_xlabel("Trial", fontsize=pu.FONT_SIZE_AXIS)
        ax_s.set_ylabel("Proportion of choices", fontsize=pu.FONT_SIZE_AXIS)

    if panel == "both":
        handles, _ = ax_m.get_legend_handles_labels()
        fig.legend(handles, ["Left", "Middle", "Right", "95% CI"],
                   bbox_to_anchor=(1.02, 0.5), loc="center left", fontsize=pu.FONT_SIZE_AXIS)
    plt.tight_layout()
    return fig


def plot_model_vs_subject_performance(performance_df):
    """model vs participant performance bars."""
    if not _require_df(performance_df, ["subject", "condition", "model_performance", "subj_performance"]):
        return None
    long_df = performance_df.melt(
        id_vars=["subject", "condition"],
        value_vars=["model_performance", "subj_performance"],
        var_name="type", value_name="performance")
    g = sns.catplot(data=long_df, x="type", y="performance", hue="type", kind="bar",
                    height=6, aspect=1.5, errorbar="sd",
                    palette=[pu.MODEL_COLOR, pu.SUBJ_COLOR])
    ax = g.ax
    for subj in performance_df["subject"].unique():
        subj_data = long_df[long_df.subject == subj]
        ax.plot(subj_data["type"].map({"model_performance": 0, "subj_performance": 1}),
                subj_data["performance"], color="gray", marker="o", linestyle="-", alpha=0.5)
    chance = performance_df.chance_level_performance.mean() if "chance_level_performance" in performance_df.columns else None
    if chance is not None:
        ax.axhline(chance, color="red", linestyle="dashed")
    ax.set_ylabel("Performance", fontsize=pu.FONT_SIZE_AXIS)
    ax.set_xlabel("")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Model", "Subjects"], fontsize=pu.FONT_SIZE_AXIS)
    ax.set_title("Model vs Subject Performance", fontsize=pu.FONT_SIZE_TITLE)
    mask = performance_df.subj_performance.gt(0.5) if performance_df.subj_performance.notna().any() else slice(None)
    t, p = ttest_rel(performance_df.loc[mask, "model_performance"],
                     performance_df.loc[mask, "subj_performance"])
    print(f"T-test paired: t={t:.2f}, p={p:.3f}")
    return g.fig


def plot_critic_by_value(critic_df, n_bins=N_VALUE_BINS):
    """critic value estimation per value range."""
    if not _require_df(critic_df, ["condition", "fb_group_new", "critic"]):
        return None
    gain_labels = pu.value_bin_labels((-30, 40), n_bins)
    loss_labels = pu.value_bin_labels((-40, 30), n_bins)
    g = sns.catplot(data=critic_df, x="fb_group_new", y="critic", row="condition",
                    kind="point", errorbar="sd", sharey=False, sharex=False, height=4, aspect=1.5)
    for ax, labels in zip(g.axes.flat, [gain_labels, loss_labels]):
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_ylabel("Critic value estimation", fontsize=pu.FONT_SIZE_AXIS)
        ax.set_xlabel("")
    g.set_xlabels("Value range", fontsize=pu.FONT_SIZE_AXIS)
    g.fig.suptitle("Critic value estimation per value range",
                   fontsize=pu.FONT_SIZE_TITLE, y=1.05)
    plt.tight_layout()
    return g.fig




# ========================================================================== #
# ========================================================================== #
def plot_unit_self_correlation(mean_within, consistency_mat, threshold, *, n_lstm=None, out_path=None):
    """Mean self-correlation scatter and consistency heatmap."""
    mean_within = np.asarray(mean_within)
    consistency_mat = np.asarray(consistency_mat)
    if n_lstm is None:
        n_lstm = len(mean_within)
    sorted_idx = np.argsort(mean_within)
    corr_mat = consistency_mat[:, sorted_idx].T

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(10, 6), gridspec_kw={"width_ratios": [1, 0.45]})
    ax_left.scatter(np.arange(n_lstm), mean_within[sorted_idx], c="gray",
                    s=25, edgecolors="black", linewidth=0.3)
    ax_left.hlines(threshold, 0, n_lstm, color="gray", linestyle="--")
    ax_left.set_xticks(np.arange(0, n_lstm, 5))
    ax_left.set_xlabel("LSTM units sorted", fontsize=pu.FONT_SIZE_AXIS)
    ax_left.set_ylabel("Mean Correlation", fontsize=pu.FONT_SIZE_AXIS)
    ax_left.set_title("Mean Within-condition Correlation", fontsize=pu.FONT_SIZE_TITLE)

    sns.heatmap(corr_mat, cmap=sns.color_palette("blend:#7AB,#EDA", as_cmap=True),
                vmin=-1, vmax=1, cbar=True,
                cbar_kws={"label": "Correlation", "shrink": 0.85}, ax=ax_right)
    ax_right.set_xticks([0.5, 1.5, 2.5], ["Across", "Gain", "Loss"], rotation=45, ha="right")
    ax_right.set_yticks([])
    ax_right.set_title("Correlation Heatmap", y=1.05)
    n_selected = int(np.sum(mean_within < threshold))
    ax_right.axhline(n_lstm - n_selected, color="black", linewidth=3, linestyle="--")
    plt.subplots_adjust(wspace=0.35)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
    return fig


def plot_pca_by_threshold(gain_common, mean_within, gain_trials, *,
                               thresholds=(0.2, 0.8), pc_indices=(0, 1)):
    """side-by-side PCA at low vs high unit-selection threshold."""
    from sklearn.decomposition import PCA
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    mean_within = np.asarray(mean_within)
    fig, axes = plt.subplots(1, 2, figsize=(21, 7), sharey=True, constrained_layout=True)
    actions = gain_trials.action.values
    explore = gain_trials.model_exploration.values if "model_exploration" in gain_trials.columns else None

    for ax, th in zip(axes, thresholds):
        units = np.where(mean_within < th)[0]
        pca_xy = PCA(n_components=2).fit_transform(
            np.concatenate(gain_common[:, :, units], axis=0))
        if explore is not None:
            pu.plot_pca_over_decision(pca_xy, actions, explore, ax=ax, pc_indices=pc_indices)
        else:
            pu.plot_pca_over_value(pca_xy, actions, gain_trials.feedback.values, ax=ax,
                                   pc_indices=pc_indices)
        ax.set_title(f"Threshold < {th}", fontsize=pu.FONT_SIZE_AXIS)
    axes[1].set_ylabel("")
    fig.supxlabel("PC 1", fontsize=pu.FONT_SIZE_AXIS + 2)
    fig.supylabel("PC 2", fontsize=pu.FONT_SIZE_AXIS + 2)
    act_handles = [
        Line2D([0], [0], marker="o", lw=0, markerfacecolor="white", markeredgecolor="black"),
        Line2D([0], [0], marker="^", lw=0, markerfacecolor="white", markeredgecolor="black"),
        Line2D([0], [0], marker="x", lw=0, markerfacecolor="none", markeredgecolor="black"),
    ]
    fig.legend(handles=act_handles, labels=["arm 1", "arm 2", "arm 3"], title="Action",
               loc="center left", bbox_to_anchor=(1.02, 0.60), fontsize=pu.FONT_SIZE_AXIS)
    if explore is not None:
        state_handles = [
            Patch(facecolor=pu.EXPLOIT_COLOR, edgecolor="none", label="Exploit"),
            Patch(facecolor=pu.EXPLORE_COLOR, edgecolor="none", label="Explore"),
        ]
        fig.legend(handles=state_handles, loc="center left", bbox_to_anchor=(1.02, 0.35),
                   fontsize=pu.FONT_SIZE_AXIS)
    fig.suptitle("PCA visualization of LSTM representations (Gain) — "
                 "Different thresholds for unit selection",
                 y=1.05, fontsize=pu.FONT_SIZE_TITLE + 2)
    return fig


def plot_rsa_by_threshold(multi_model_df_th, threshold, ttest_df=None,
                         naive_models=None):
    """ROI × model RSA bars at a given unit-selection threshold."""
    if not _require_df(multi_model_df_th, ["roi", "model", "corr2model"]):
        return None
    if naive_models is None:
        naive_models = ["fb", "act_fb", "spatial", "lstm_subj"]
    df = multi_model_df_th[multi_model_df_th.model.isin(naive_models)]
    palette = dict(zip(naive_models, sns.color_palette(n_colors=len(naive_models))))
    g = sns.catplot(data=df, y="corr2model", x="roi", hue="model", kind="bar",
                    height=5.5, aspect=1.5, palette=palette, hue_order=naive_models)
    g._legend.set_title("Model")
    for t in g._legend.texts:
        t.set_text(MODEL_LABELS.get(t.get_text(), t.get_text()))
    g.fig.suptitle(f"Models to ROI correlation — threshold {threshold}",
                   fontsize=pu.FONT_SIZE_AXIS + 1, weight="bold", y=1.02)
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=pu.FONT_SIZE_AXIS)
        ax.set_ylabel("Spearman correlation", fontsize=pu.FONT_SIZE_AXIS)
        ax.set_xlabel("")
    plt.subplots_adjust(top=0.9)
    return g.fig


def plot_unit_count_by_threshold(mean_within, thresholds=UNIT_SELECTION_THRESHOLDS):
    """number of selected units across thresholds."""
    mean_within = np.asarray(mean_within)
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = [int(np.sum(mean_within < th)) for th in thresholds]
    ax.plot(thresholds, counts, marker="o")
    ax.set(xlabel="Consistency threshold", ylabel="# selected units",
           title="Unit-selection threshold sensitivity")
    plt.tight_layout()
    return fig


def plot_selection_threshold_analysis(gain_common, mean_within, consistency_mat, gain_trials,
                       multi_model_df_by_thresh=None, ttest_df=None):
    """unit diagnostic, PCA, and RSA across selection thresholds."""
    if mean_within is None or consistency_mat is None:
        return []
    figs = []
    pca_done = False
    for th in UNIT_SELECTION_THRESHOLDS:
        figs.append(plot_unit_self_correlation(mean_within, consistency_mat, th))
        if not pca_done and gain_common is not None and gain_trials is not None:
            figs.append(plot_pca_by_threshold(
                gain_common, mean_within, gain_trials, thresholds=(0.2, 0.8)))
            pca_done = True
        if multi_model_df_by_thresh and th in multi_model_df_by_thresh:
            figs.append(plot_rsa_by_threshold(multi_model_df_by_thresh[th], th, ttest_df))
    figs.append(plot_unit_count_by_threshold(mean_within))
    return figs


# ========================================================================== #
# ========================================================================== #
def plot_loss_condition_pca(loss_common, selected_units, loss_trials):
    """PCA of selected units for the Loss condition."""
    from sklearn.decomposition import PCA

    if loss_common is None or selected_units is None or loss_trials is None:
        return None, None
    pca = PCA(n_components=10)
    pca_xy = pca.fit_transform(np.concatenate(loss_common[:, :, selected_units], axis=0))
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    pu.plot_pca_over_value(pca_xy, loss_trials.action.values,
                           loss_trials.feedback.values, ax=axes[0])
    if "model_exploration" in loss_trials.columns:
        pu.plot_pca_over_decision(pca_xy, loss_trials.action.values,
                                  loss_trials.model_exploration.values, ax=axes[1])
    fig.suptitle("PCA visualization of LSTM representations - Loss condition",
                 fontsize=pu.FONT_SIZE_TITLE)
    plt.tight_layout()
    return fig, pca


def plot_pca_explained_variance(gain_var=None, loss_var=None, *,
                           pca_gain=None, pca_loss=None):
    """per-PC and cumulative explained variance for Gain and Loss."""
    if gain_var is None and pca_gain is not None:
        gain_var = pca_gain.explained_variance_ratio_
    if loss_var is None and pca_loss is not None:
        loss_var = pca_loss.explained_variance_ratio_
    if gain_var is None or loss_var is None:
        return None
    gain_var = np.asarray(gain_var)
    loss_var = np.asarray(loss_var)
    fig, ax = plt.subplots(figsize=(8, 5))
    x_g = range(1, len(gain_var) + 1)
    x_l = range(1, len(loss_var) + 1)
    ax.plot(x_g, gain_var, marker="o", label="Gain", color=pu.GAIN_COLOR)
    ax.plot(x_l, loss_var, marker="o", label="Loss", color=pu.LOSS_COLOR)
    ax.plot(x_g, np.cumsum(gain_var), marker="x", linestyle="--",
            label="Cumulative Gain", color=pu.GAIN_COLOR)
    ax.plot(x_l, np.cumsum(loss_var), marker="x", linestyle="--",
            label="Cumulative Loss", color=pu.LOSS_COLOR)
    ax.set_xlabel("Principal Component", fontsize=pu.FONT_SIZE_AXIS)
    ax.set_ylabel("Explained Variance Ratio", fontsize=pu.FONT_SIZE_AXIS)
    ax.set_title("Explained Variance Ratio by Principal Component", fontsize=pu.FONT_SIZE_TITLE)
    ax.legend(fontsize=pu.FONT_SIZE_AXIS)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_pca_over_trials(pca_xy, trial_actions, *, condition="Gain",
                        common_num_trials=None):
    """PCA trajectories colored by trial index."""
    if pca_xy is None or trial_actions is None:
        return None
    fig, ax = plt.subplots(figsize=(10, 7))
    pu.plot_pcs_over_trials(pca_xy, trial_actions, ax=ax, common_num_trials=common_num_trials,
                            title=f"All Subj {condition} PCA: actions over time")
    plt.tight_layout()
    return fig


def plot_rsa_by_condition(multi_model_df, ttest_df=None, naive_models=None):
    """per-condition RSA bar plot with FDR stars."""
    if not _require_df(multi_model_df, ["roi", "model", "condition", "corr2model"]):
        return None
    if naive_models is None:
        naive_models = MODEL_ORDER[:4]
    df = multi_model_df.copy()
    if df["condition"].isin(["rew", "pun"]).any():
        df["condition"] = df["condition"].replace({"rew": "Gain", "pun": "Loss"})
    return plot_rsa_correlations(df[df.model.isin(naive_models)], ttest_df=ttest_df)


# ========================================================================== #
# ========================================================================== #
def plot_functional_roi_table(mask_metadata_df):
    """functional ROI peak and voxel-count table."""
    if not _require_df(mask_metadata_df):
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    display_df = mask_metadata_df.copy()
    tbl = ax.table(cellText=display_df.values, colLabels=display_df.columns,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.4)
    ax.set_title("Functional ROI summary", fontsize=pu.FONT_SIZE_TITLE, pad=20)
    plt.tight_layout()
    return fig


def plot_searchlight_conjunction(behavior_map, model_map, fb_modulation_map, *,
                                        fdr_alpha=0.05, cluster_k=10,
                                        display_mode="lyrz"):
    """searchlight conjunction on a glass brain."""
    from nilearn.image import binarize_img, math_img, threshold_stats_img
    from nilearn.plotting import plot_glass_brain

    if behavior_map is None or model_map is None:
        return None

    beh_th = binarize_img(behavior_map)
    model_th = binarize_img(model_map) if hasattr(model_map, "get_fdata") else model_map
    if fb_modulation_map is not None:
        fb_th, _ = threshold_stats_img(
            fb_modulation_map, alpha=fdr_alpha, height_control="fdr",
            cluster_threshold=cluster_k, two_sided=True)
        three_conj = math_img("img1 * img2 * img3",
                              img1=beh_th, img2=model_th, img3=binarize_img(fb_th))
        corr_with_model_conj = math_img("img1 * img2", img1=beh_th, img2=model_th)
        plot_glass_brain(three_conj, colorbar=True, display_mode=display_mode,
                         title="Three-way conjunction (yellow)")
        return plot_glass_brain(corr_with_model_conj, colorbar=True, display_mode=display_mode,
                                title="Behavior × model conjunction")
    return plot_glass_brain(math_img("img1 * img2", img1=beh_th, img2=model_th),
                            colorbar=True, display_mode=display_mode,
                            title="Behavior × model conjunction")


def plot_searchlight_maps(searchlight_group_maps):
    """group searchlight model-fit maps on a glass brain."""
    from nilearn.plotting import plot_glass_brain
    if not searchlight_group_maps:
        return None
    for name, img in searchlight_group_maps.items():
        plot_glass_brain(img, colorbar=True, display_mode="lyrz",
                         title=f"Searchlight: {name}")


def plot_searchlight_contrasts(second_level_df, models_to_plot=None):
    """Plot selected second-level searchlight contrast maps."""
    if not _require_df(second_level_df, ["condition", "model", "map"]):
        return None
    if models_to_plot is None:
        models_to_plot = [m for m in second_level_df.model.unique() if str(m).startswith("lstm")]
    maps = {f"{row.condition}_{row.model}": row.map
            for _, row in second_level_df.iterrows() if row.model in models_to_plot}
    return plot_searchlight_maps(maps)


# ========================================================================== #
# ========================================================================== #
def _multiseed_with_hue(multiseed_corr_df):
    """Add model_vs_act_fb column for bar hue."""
    df = multiseed_corr_df.copy()
    if "model_vs_act_fb" not in df.columns:
        df["model_vs_act_fb"] = np.where(df["model"] == "act_fb", "act_fb", "lstm_subj")
    return df


def multiseed_rsa_ttest(multiseed_corr_df, best_models=None, likelihood_thresh=0.7):
    """t-test of best seeds vs Action-Value per value ROI."""
    from statsmodels.stats.multitest import multipletests

    if not _require_df(multiseed_corr_df, ["seed", "roi", "model", "corr2model"]):
        return None
    if best_models is None:
        if "likelihood" in multiseed_corr_df.columns:
            best_models = (multiseed_corr_df.groupby("seed")["likelihood"]
                           .first() > likelihood_thresh)
            best_models = best_models[best_models].index.tolist()
        else:
            best_models = multiseed_corr_df.seed.unique().tolist()
    res = []
    for roi in FB_ROIS:
        df = multiseed_corr_df[multiseed_corr_df.roi == roi]
        x = df[df.model.isin([str(m) for m in best_models])]["corr2model"].values
        y = df[df.model == "act_fb"]["corr2model"].values
        t, p = ttest_ind(x, y, nan_policy="omit")
        res.append({"roi": roi, "t": t, "p": p, "roi_model_comp": "lstm_subj_vs_act_fb"})
    ttest_df = pd.DataFrame(res)
    _, p_fdr, _, _ = multipletests(ttest_df["p"], alpha=0.05, method="fdr_bh")
    ttest_df["p_fdr"] = p_fdr
    ttest_df["stars_fdr"] = [ru.p_to_stars(p) for p in p_fdr]
    return ttest_df


def plot_multiseed_rsa(multiseed_corr_df, ttest_df=None, best_models=None,
                      likelihood_thresh=0.7):
    """multiseed RSA bar with model_vs_act_fb hue."""
    if not _require_df(multiseed_corr_df):
        return None
    if best_models is None:
        if "likelihood" in multiseed_corr_df.columns:
            best_models = (multiseed_corr_df.groupby("seed")["likelihood"]
                           .first() > likelihood_thresh)
            best_models = [str(m) for m in best_models[best_models].index.tolist()]
        else:
            best_models = [str(s) for s in multiseed_corr_df.seed.unique()]
    df = _multiseed_with_hue(multiseed_corr_df)
    plot_df = df[df.roi.isin(FB_ROIS) & df.model.isin(best_models + ["act_fb"])]
    g = sns.catplot(data=plot_df, x="roi", hue="model_vs_act_fb", y="corr2model",
                    kind="bar", height=5, aspect=1.5,
                    palette=[pu.MODEL_COLOR, sns.color_palette()[0]])
    ax = g.ax
    if ttest_df is not None:
        try:
            from statannotations.Annotator import Annotator
            pairs = [((row["roi"], "lstm_subj"), (row["roi"], "act_fb"))
                     for _, row in ttest_df.iterrows()]
            annotations = ttest_df["stars_fdr"].tolist()
            annotator = Annotator(ax=ax, pairs=pairs, data=plot_df,
                                  x="roi", y="corr2model", hue="model_vs_act_fb")
            annotator.configure(test=None, verbose=False)
            annotator.set_custom_annotations(annotations)
            annotator.annotate()
        except ImportError:
            for i, row in ttest_df.iterrows():
                ax.text(i, ax.get_ylim()[1] * 0.95, row["stars_fdr"], ha="center")
    if g._legend is not None:
        g._legend.remove()
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["Deep-RL", "Action-value"], title="Model",
              fontsize=pu.FONT_SIZE_AXIS, loc="upper right", bbox_to_anchor=(1.3, 0.9))
    ax.set_title("Multi seed models vs. Action-value model",
                 fontsize=pu.FONT_SIZE_AXIS + 1, weight="bold")
    ax.set_ylabel("Spearman correlation", fontsize=pu.FONT_SIZE_AXIS)
    ax.set_xlabel("ROI", fontsize=pu.FONT_SIZE_AXIS)
    plt.xticks(rotation=45)
    return g.fig




def plot_example_seed_rdm(mean_rdm, *, seed_idx=0, condition="rew", n_bins=N_VALUE_BINS):
    """example alternate-seed mean RDM heatmap."""
    if mean_rdm is None:
        return None
    if isinstance(mean_rdm, list):
        if seed_idx >= len(mean_rdm):
            return None
        rdm_obj = mean_rdm[seed_idx].get(condition) if isinstance(mean_rdm[seed_idx], dict) else mean_rdm[seed_idx]
    else:
        rdm_obj = mean_rdm
    plt.figure()
    return pu.plot_rdm(rdm_obj, n_bins,
                       f"Example seed RDM ({condition})",
                       cbar_label="Dissimilarity\n(Euclidean)")


# ========================================================================== #
# ========================================================================== #
def plot_exploration_neural_scatter(rsa_behavior_df, roi="vmPFC"):
    """exploration slope vs diagonal slope, shown separately per condition."""
    if not _require_df(rsa_behavior_df, ["roi", "condition", "exploration_slope", "diag_slope"]):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, cond in zip(axes, ["Gain", "Loss"]):
        d = rsa_behavior_df[(rsa_behavior_df.roi == roi) & (rsa_behavior_df.condition == cond)]
        if d.empty:
            continue
        sns.regplot(data=d, x="exploration_slope", y="diag_slope", ax=ax)
        rho, p = spearmanr(d["exploration_slope"], d["diag_slope"])
        ax.set_title(f"{cond}: rho={rho:.2f}, p={p:.1e}")
    plt.tight_layout()
    return fig


def _build_partial_corr_table(lstm_rsa_df):
    """Partial correlation of exploration_slope with diag_slope per ROI."""
    from statsmodels.stats.multitest import multipletests

    rows = []
    for roi in ALL_ROIS:
        df_roi = lstm_rsa_df[lstm_rsa_df.roi == roi]
        partial = bu.partial_correlations(
            df_roi, outcome="diag_slope",
            predictors=["exploration_slope", "exploration_frequency", "value_slope", "performance"])
        row = partial[partial.predictor == "exploration_slope"].iloc[0]
        rows.append({"roi": roi, "predictor": "exploration_slope",
                     "partial_r": row.partial_r, "p_value": row.p_value})
    out = pd.DataFrame(rows)
    _, p_fdr, _, _ = multipletests(out["p_value"], alpha=0.05, method="fdr_bh")
    out["p_fdr"] = p_fdr
    out["significant_fdr"] = p_fdr < 0.05
    return out


def plot_partial_correlation_table(partial_corr_results_df=None, lstm_rsa_df=None):
    """partial-correlation table figure."""
    if partial_corr_results_df is None:
        if not _require_df(lstm_rsa_df):
            return None
        partial_corr_results_df = _build_partial_corr_table(lstm_rsa_df)
    df = partial_corr_results_df[partial_corr_results_df.predictor == "exploration_slope"]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    show = df[["roi", "partial_r", "p_value", "p_fdr"]].round(4) if "p_fdr" in df.columns else df.round(4)
    tbl = ax.table(cellText=show.values, colLabels=show.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    ax.set_title("Partial correlation: exploration slope → neural diagonal slope",
                 fontsize=pu.FONT_SIZE_TITLE, pad=20)
    plt.tight_layout()
    return fig


def plot_partial_correlation_bar(partial_corr_results_df=None, lstm_rsa_df=None, *, fdr_alpha=0.05):
    """bar of partial r per ROI with FDR significance."""
    if partial_corr_results_df is None:
        if not _require_df(lstm_rsa_df):
            return None
        partial_corr_results_df = _build_partial_corr_table(lstm_rsa_df)
    df = partial_corr_results_df[partial_corr_results_df.predictor == "exploration_slope"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="roi", y="partial_r", ax=ax, palette=[pu.ROI_COLORS.get(r, "gray") for r in df.roi])
    if "p_fdr" in df.columns:
        for i, row in df.iterrows():
            if row.p_fdr < fdr_alpha:
                ax.text(list(df.roi).index(row.roi), row.partial_r + 0.02, "*", ha="center", fontsize=14)
    ax.set_ylabel("Partial correlation (exploration slope)", fontsize=pu.FONT_SIZE_AXIS)
    ax.set_xlabel("")
    ax.set_title("Partial correlation by ROI (FDR-corrected)", fontsize=pu.FONT_SIZE_TITLE)
    plt.tight_layout()
    return fig


def partial_correlations_by_roi(rsa_behavior_df, roi="vmPFC"):
    """partial correlations for one ROI."""
    if not _require_df(rsa_behavior_df):
        return None
    predictors = ["exploration_slope", "exploration_frequency", "value_slope", "performance"]
    df = rsa_behavior_df[rsa_behavior_df.roi == roi]
    return bu.partial_correlations(df, outcome="diag_slope", predictors=predictors)


def plot_whole_brain_exploration_map(correlation_img, roi_masks=None, *, r_thresh=0.252,
                              n_subj=31, cluster_k=20):
    """whole-brain exploration-slope correlation map."""
    from scipy.stats import t as t_dist
    from nilearn.image import binarize_img, math_img, threshold_img
    from nilearn.plotting import plot_glass_brain

    if correlation_img is None:
        return None

    if r_thresh is None:
        t_crit = t_dist.ppf(0.975, df=n_subj * 2 - 2)
        r_thresh = (t_crit ** 2 / (t_crit ** 2 + n_subj * 2 - 2)) ** 0.5

    corr_th = threshold_img(correlation_img, threshold=r_thresh,
                            cluster_threshold=1, two_sided=True, copy=True)
    plot_glass_brain(corr_th, colorbar=True, display_mode="lyrz",
                     title=f"Whole-brain exploration slope correlation (r ≥ {r_thresh:.3f})")

    if roi_masks is not None:
        from nilearn.image import concat_imgs
        import nilearn
        if isinstance(roi_masks, dict):
            mask_imgs = [m.mask_img if hasattr(m, "mask_img") else m for m in roi_masks.values()]
        else:
            mask_imgs = list(roi_masks)
        concat_masks = math_img("np.sum(imgs, axis=3)",
                                imgs=nilearn.image.concat_imgs(mask_imgs))
        overlay = math_img("img1 + (img2 * 2)", img1=concat_masks,
                           img2=binarize_img(corr_th))
        return plot_glass_brain(overlay, colorbar=True, display_mode="lyrz",
                                title="Functional ROIs + thresholded correlation")
    return None


def plot_whole_brain_exploration_maps(correlation_img, pval_img=None, **kwargs):
    """Whole-brain exploration map with optional p-value overlay."""
    plot_whole_brain_exploration_map(correlation_img, **kwargs)
    if pval_img is not None:
        from nilearn.plotting import plot_glass_brain
        plot_glass_brain(pval_img, colorbar=True, display_mode="lyrz", title="p-values")


# ========================================================================== #
# Runner
# ========================================================================== #
def main(force: bool = False):
    """Run supplementary plots."""
    paths.EXTENDED_FIGS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_supplementary_data(force=force)

    plot_gain_loss_comparisons(data.episode_df, data.all_trials_df)
    plot_split_half_independence(data.gain_loss_corr_df, data.half_random_corr_df)

    if data.all_trials_df is not None and data.episode_df is not None:
        plot_exploration_window_robustness(data.all_trials_df, data.episode_df)

    plot_arm_choice_histograms(data.per_trial_df)
    plot_choice_rate_over_trials(data.per_trial_df)
    plot_model_vs_subject_performance(data.performance_df)
    plot_critic_by_value(data.critic_df)

    gain_trials = None
    if data.common_shape_df is not None:
        gain_trials = data.common_shape_df[data.common_shape_df.condition == "Gain"]
    plot_selection_threshold_analysis(data.gain_common, data.mean_within, data.consistency_mat,
                       gain_trials, data.multi_model_df_by_thresh, data.ttest_df)

    pca_gain_obj = None
    pca_loss_obj = None
    if data.gain_common is not None and data.selected_units is not None:
        from sklearn.decomposition import PCA
        pca_gain_obj = PCA(n_components=10).fit(
            np.concatenate(data.gain_common[:, :, data.selected_units], axis=0))
    if data.loss_common is not None and data.selected_units is not None and data.loss_trials is not None:
        _, pca_loss_obj = plot_loss_condition_pca(data.loss_common, data.selected_units, data.loss_trials)
    plot_pca_explained_variance(data.explained_var_gain, data.explained_var_loss,
                           pca_gain=pca_gain_obj, pca_loss=pca_loss_obj)
    if data.pca_gain_xy is not None and data.common_shape_df is not None:
        gain_actions = data.common_shape_df[data.common_shape_df.condition == "Gain"].action.values
        n_trials = len(gain_actions) // bu.N_SUBJ if len(gain_actions) >= bu.N_SUBJ else None
        plot_pca_over_trials(data.pca_gain_xy, gain_actions, condition="Gain",
                            common_num_trials=n_trials)
    plot_rsa_by_condition(data.multi_model_df, data.ttest_df)

    plot_functional_roi_table(data.mask_metadata_df)
    plot_searchlight_conjunction(
        data.correlation_img, data.searchlight_model_map, data.second_level_fb_modulation)
    if data.second_level_contrasts is not None:
        plot_searchlight_contrasts(data.second_level_contrasts)

    ttest_ms = multiseed_rsa_ttest(data.multiseed_corr_df)
    plot_multiseed_rsa(data.multiseed_corr_df, ttest_df=ttest_ms)
    plot_example_seed_rdm(data.mean_rdms_lstm_models_all)

    rsa_df = data.lstm_rsa_df if data.lstm_rsa_df is not None else data.rsa_behavior_df
    plot_exploration_neural_scatter(rsa_df)
    plot_partial_correlation_table(data.partial_corr_results_df, lstm_rsa_df=rsa_df)
    plot_partial_correlation_bar(data.partial_corr_results_df, lstm_rsa_df=rsa_df)
    plot_whole_brain_exploration_map(data.correlation_img, data.functional_maskers)

    plt.show()


if __name__ == "__main__":
    main()

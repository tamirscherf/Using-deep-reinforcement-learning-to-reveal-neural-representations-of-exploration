"""Task behavior and deep-RL model fit.

Loads participant behavior, runs the trained model, and plots example episodes,
performance–exploration relationships, model likelihood, and exploration by value.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, ttest_1samp

import behavior_utils as bu
import model_pipeline as mp
import paths
import plotting_utils as pu


# ========================================================================== #
# Data loading
# ========================================================================== #
def load_behavior():
    """Load per-subject behavioral DataFrames for Gain and Loss conditions.

    Each element is a DataFrame with (at least) columns:
    ``choice`` (1-indexed arm), ``FB`` (outcome value), and ``bandit_1..3``
    (the three arms' value functions).
    """
    gain = pd.read_pickle(paths.RAW_BEHAVIOR_GAIN)
    loss = pd.read_pickle(paths.RAW_BEHAVIOR_LOSS)
    return gain, loss


def load_overall_performance():
    """Load per-subject overall performance (corr. of actual vs optimal value)."""
    gain = pd.read_pickle(paths.RAW_PERFORMANCE_GAIN)
    loss = pd.read_pickle(paths.RAW_PERFORMANCE_LOSS)
    return {"Gain": gain, "Loss": loss}


def build_episode_summaries(all_trials_df, likelihood_df):
    """Collapse trials to one row per episode (subject x condition) with the
    scalar behavioral measures (exploration rate, slope, performance)."""
    df_bins = all_trials_df.groupby(["subject", "condition", "fb_group"]).mean().reset_index()
    df_quant = all_trials_df.groupby(["subject", "condition", "fb_quantile"]).mean().reset_index()

    mean_df = all_trials_df.groupby(["subject", "condition"]).mean().reset_index()
    mean_df = mean_df.merge(likelihood_df, on=["subject", "condition"], how="left")

    slopes = bu.exploration_slope_by_episode(df_bins, "fb_group", "exploration")
    model_slopes = bu.exploration_slope_by_episode(df_bins, "fb_group", "model_exploration")
    quant_slopes = bu.exploration_slope_by_episode(df_quant, "fb_quantile", "exploration")

    mean_df["exploration_slope_bin"] = mean_df.apply(
        lambda r: slopes[r.condition][r.subject], axis=1)
    mean_df["model_exploration_slope_bin"] = mean_df.apply(
        lambda r: model_slopes[r.condition][r.subject], axis=1)
    mean_df["exploration_slope"] = mean_df.apply(
        lambda r: quant_slopes[r.condition][r.subject], axis=1)
    mean_df = mean_df.rename(columns={"overall_performance": "performance"})
    return mean_df, df_bins


# ========================================================================== #
# Example episodes
# ========================================================================== #
def plot_example_episode(ax, actions, arm_values, model_pred=None, subject_pred=None,
                         title=""):
    """Plot one episode: arm value functions, chosen arm markers, optimal arm.

    If ``model_pred``/``subject_pred`` are provided, choices are marked as correct
    (open circle) or incorrect (x) model predictions.
    """
    x = np.arange(len(actions))
    for a, (color, name) in enumerate(zip(pu.ARM_COLORS, ["Left", "Middle", "Right"])):
        ax.plot(x, arm_values[a], label=name, color=color, alpha=0.5)
    ax.plot(x, arm_values.max(axis=0), linestyle="dotted", color="gray", label="Optimal arm")

    if model_pred is None:
        for a in range(3):
            idx = actions == a
            ax.scatter(x[idx], arm_values[a, idx], marker="o", facecolor="none",
                       edgecolor="black", s=30)
    else:
        match = subject_pred == model_pred
        for a in range(3):
            idx = (actions == a + 1)
            ax.scatter(x[idx & match], arm_values[a, idx & match], marker="o",
                       facecolor="none", edgecolor="black")
            ax.scatter(x[idx & ~match], arm_values[a, idx & ~match], marker="x", c="black")
    ax.set_xlabel("Trial", fontsize=pu.FONT_SIZE_AXIS)
    ax.set_ylabel("Value", fontsize=pu.FONT_SIZE_AXIS)
    ax.set_title(title, fontsize=pu.FONT_SIZE_AXIS)


def plot_example_performance_episodes(behavior_gain, behavior_loss, performance):
    """Example Gain and Loss episodes with participant choices."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (subj, cond, behavior) in zip(
        axes, [(12, "Gain", behavior_gain), (30, "Loss", behavior_loss)]
    ):
        arm_values = np.vstack([behavior[subj][b].to_numpy() for b in ["bandit_1", "bandit_2", "bandit_3"]])
        plot_example_episode(ax, behavior[subj].choice.to_numpy() - 1, arm_values,
                             title=f"Subject {subj}, {cond} "
                                   f"(perf={performance[cond][subj]:.2f})")
    fig.suptitle("Participants' overall performance", fontsize=pu.FONT_SIZE_TITLE)
    plt.tight_layout()
    return fig


def plot_model_prediction_episodes(all_trials_df, behavior_gain, behavior_loss):
    """Example episodes with the model's exploration predictions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (subj, cond, behavior) in zip(
        axes, [(12, "Gain", behavior_gain), (30, "Loss", behavior_loss)]
    ):
        trials = all_trials_df.query("subject == @subj and condition == @cond")
        arm_values = np.vstack([behavior[subj][b].to_numpy() for b in ["bandit_1", "bandit_2", "bandit_3"]])
        plot_example_episode(ax, trials.action.values, arm_values,
                             model_pred=trials.model_exploration.values,
                             subject_pred=trials.exploration.values,
                             title=f"Subject {subj}, {cond}")
    fig.suptitle("Exploration prediction", fontsize=pu.FONT_SIZE_TITLE)
    plt.tight_layout()
    return fig


# ========================================================================== #
# Performance and exploration
# ========================================================================== #
def plot_performance_exploration_relations(episode_df):
    """Performance vs exploration rate and exploration slope."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    sns.regplot(data=episode_df, x="exploration", y="performance", order=2, ax=axes[0])
    r2, _, p = bu.quadratic_fit_stats(episode_df, "exploration", "performance")
    axes[0].set(xlabel="Exploration rate", ylabel="Overall performance")
    axes[0].set_title(f"Quadratic fit (adj. R\u00b2={r2:.2f}, p={p:.1e})", fontsize=pu.FONT_SIZE_AXIS)

    (r, p), mask = bu.pearson_zfiltered(
        episode_df["exploration_slope_bin"], episode_df["performance"], return_mask=True)
    sns.regplot(x=episode_df["exploration_slope_bin"][mask],
                y=episode_df["performance"][mask], ax=axes[1])
    axes[1].set(xlabel="Exploration slope", ylabel="")
    axes[1].set_title(f"Pearson r={r:.2f}, p={p:.1e}", fontsize=pu.FONT_SIZE_AXIS)
    plt.tight_layout()
    return fig


# ========================================================================== #
# Model likelihood
# ========================================================================== #
def plot_likelihood_bars(episode_df):
    """Overall and explore-exploit model likelihood."""
    long = episode_df.melt(id_vars=["subject", "condition"],
                           value_vars=["likelihood", "exploration_fit"],
                           var_name="measure", value_name="value")
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.barplot(data=long, x="measure", y="value", hue="measure",
                errorbar="sd", palette="gray", ax=ax)
    sns.stripplot(data=long, x="measure", y="value", color="black", alpha=0.6, jitter=True, ax=ax)
    ax.axhline(0.33, xmin=0, xmax=0.5, color="black", linestyle="dashed", label="Chance")
    ax.axhline(0.50, xmin=0.5, xmax=1, color="black", linestyle="dashed")
    ax.set(ylim=(0, 1.05), xlabel="", ylabel="Likelihood")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Overall likelihood", "Explore-exploit likelihood"], fontsize=pu.FONT_SIZE_AXIS)

    _, p_overall = ttest_1samp(long[long.measure == "likelihood"].value, 0.33)
    _, p_explore = ttest_1samp(long[long.measure == "exploration_fit"].value, 0.5)
    ax.set_title(f"Model fit (overall p={p_overall:.1e}, explore-exploit p={p_explore:.1e})")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


# ========================================================================== #
# Exploration by value
# ========================================================================== #
def plot_exploration_by_value(df_bins, n_bins=bu.N_VALUE_BINS):
    """Exploration rate across value bins, participants vs model."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.pointplot(data=df_bins, x="fb_group", y="exploration", errorbar="sd",
                  color=pu.SUBJ_COLOR, label="Participants", ax=ax)
    sns.pointplot(data=df_bins, x="fb_group", y="model_exploration", errorbar="sd",
                  color=pu.MODEL_COLOR, linestyles="--", label="Model", ax=ax)
    ax.set_ylabel("Exploration rate", fontsize=pu.FONT_SIZE_AXIS)
    ax.set_title("Exploration rate over value", fontsize=pu.FONT_SIZE_TITLE)
    pu.set_dual_xaxis_labels(
        ax,
        pu.value_bin_labels(bu.GAIN_BIN_RANGE, n_bins),
        pu.value_bin_labels(bu.LOSS_BIN_RANGE, n_bins),
    )
    ax.legend(title="")
    plt.tight_layout()
    return fig


# ========================================================================== #
# Behavior tables
# ========================================================================== #
def compute_behavior_tables():
    """Build trial- and episode-level tables from behavior and model outputs."""
    behavior_gain, behavior_loss = load_behavior()
    performance = load_overall_performance()

    trained_model, _ = mp.load_trained_model()
    model_outputs = mp.feed_all_subjects(
        trained_model, behavior_gain, behavior_loss, bu.N_SUBJ)

    rows = []
    for cond, behavior in (("Gain", behavior_gain), ("Loss", behavior_loss)):
        for subj in range(bu.N_SUBJ):
            like = mp.calc_likelihood(
                behavior[subj].choice.to_numpy() - 1,
                behavior[subj].FB.to_numpy(),
                trained_model,
            )
            rows.append({"subject": subj, "condition": cond, "likelihood": like})
    likelihood_df = pd.DataFrame(rows)

    all_trials_df = bu.build_all_trials_df(behavior_gain, behavior_loss, model_outputs)
    all_trials_df["overall_performance"] = all_trials_df.apply(
        lambda r: performance[r.condition][r.subject], axis=1)
    episode_df, df_bins = build_episode_summaries(all_trials_df, likelihood_df)

    paths.save_pickle(paths.ALL_TRIALS_DF, all_trials_df)
    paths.save_pickle(paths.EPISODE_SUMMARIES, episode_df)
    paths.save_pickle(paths.MODEL_OUTPUTS, model_outputs)
    paths.save_pickle(paths.LIKELIHOOD_DF, likelihood_df)
    paths.save_pickle(paths.DF_BINS, df_bins)

    return {
        "all_trials_df": all_trials_df,
        "episode_df": episode_df,
        "likelihood_df": likelihood_df,
        "model_outputs": model_outputs,
        "df_bins": df_bins,
        "behavior_gain": behavior_gain,
        "behavior_loss": behavior_loss,
        "performance": performance,
    }


# ========================================================================== #
# Runner
# ========================================================================== #
def main(force: bool = False):
    import data_pipeline as dp

    bundle = dp.get_behavior_tables(force=force)
    behavior_gain = bundle["behavior_gain"]
    behavior_loss = bundle["behavior_loss"]
    performance = bundle["performance"]
    all_trials_df = bundle["all_trials_df"]
    episode_df = bundle["episode_df"]
    df_bins = bundle["df_bins"]

    plot_example_performance_episodes(behavior_gain, behavior_loss, performance)
    plot_model_prediction_episodes(all_trials_df, behavior_gain, behavior_loss)
    plot_performance_exploration_relations(episode_df)
    plot_likelihood_bars(episode_df)
    plot_exploration_by_value(df_bins)
    plt.show()


if __name__ == "__main__":
    main()

"""Shared plotting utilities: styling, palettes, and reusable figure helpers."""

from __future__ import annotations

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
FONT_SIZE_TITLE = 20
FONT_SIZE_AXIS = 15

# Shared color palette.
SUBJ_COLOR = "#1f77b4"
MODEL_COLOR = "#ff7f0e"
EXPLORE_COLOR = "#9b59b6"
EXPLOIT_COLOR = "#1abc9c"
GAIN_COLOR = "#2ecc71"
LOSS_COLOR = "#e74c3c"

# Per-arm colors (Left / Middle / Right) sampled from gist_rainbow.
_cmap = cm.gist_rainbow
ARM_COLORS = [_cmap(0.05)[:3], _cmap(0.40)[:3], _cmap(0.75)[:3]]

# ROI colors (colorblind-friendly).
ROI_COLORS = {
    "vmPFC": (159 / 255, 50 / 255, 38 / 255),
    "dACC": (117 / 255, 202 / 255, 225 / 255),
    "Insula": (127 / 255, 127 / 255, 127 / 255),
    "V1": (170 / 255, 191 / 255, 153 / 255),
    "FUSF": (166 / 255, 137 / 255, 187 / 255),
}

VALUE_COLORMAP = sns.color_palette("YlOrBr", as_cmap=True)

import paths

FIGS_DIR = paths.FIGS_DIR


def save_fig(fig_dir: str, name: str, fmt: str = "pdf") -> None:
    """Save the current figure with publication settings (editable text)."""
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.family"] = "Arial"
    out_dir = FIGS_DIR / fig_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"{name}.{fmt}", format=fmt, bbox_inches="tight", dpi=300)


# --------------------------------------------------------------------------- #
# Dual value-range x-axis (Gain top, Loss bottom)
# --------------------------------------------------------------------------- #
def value_bin_labels(bin_range, n_bins):
    """Human-readable value-range labels, e.g. '[30, 40)'."""
    edges = np.linspace(bin_range[0], bin_range[1], n_bins + 1)
    return [f"[{int(edges[i])}, {int(edges[i + 1])})" for i in range(n_bins)]


def set_dual_xaxis_labels(ax, gain_labels, loss_labels, font_size_axis=FONT_SIZE_AXIS):
    """Label the shared value-bin x-axis with Gain (primary) and Loss (secondary)."""
    ax.set_xticklabels(gain_labels, rotation=45, fontsize=font_size_axis)
    secax = ax.secondary_xaxis("top")
    secax.set_xticks(ax.get_xticks())
    secax.set_xticklabels(loss_labels, rotation=45, fontsize=font_size_axis)
    secax.set_xlabel("Value range", fontsize=font_size_axis)
    ax.set_xlabel("Value range", fontsize=font_size_axis)
    return secax


# --------------------------------------------------------------------------- #
# PCA scatter plots
# --------------------------------------------------------------------------- #
_ARM_MARKERS = ["o", "^", "x"]


def plot_pca_over_value(pca_xy, actions, values, ax=None, colormap=VALUE_COLORMAP,
                        pc_indices=(0, 1), fontsize=FONT_SIZE_AXIS):
    """Scatter first two PCs colored by value; marker encodes chosen arm."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))
    fig = ax.figure
    vmin, vmax = np.min(values), np.max(values)
    rew_colors = colormap(np.linspace(0, 1, int(vmax - vmin) + 1))
    fb_range = np.arange(vmin, vmax)
    for a, marker in zip([1, 2, 3], _ARM_MARKERS):
        idx = actions.astype(int) == a
        c = rew_colors[np.searchsorted(fb_range, values[idx])]
        ax.scatter(pca_xy[idx, pc_indices[0]], pca_xy[idx, pc_indices[1]], c=c, marker=marker)
    sm = cm.ScalarMappable(cmap=colormap)
    sm.set_clim(vmin=vmin, vmax=vmax)
    fig.colorbar(sm, ax=ax).set_label("Value", fontsize=fontsize)
    ax.set_xlabel(f"PC {pc_indices[0] + 1}", fontsize=fontsize)
    ax.set_ylabel(f"PC {pc_indices[1] + 1}", fontsize=fontsize)
    _add_arm_legend(ax, fontsize)
    return ax


def plot_pca_over_decision(pca_xy, actions, is_exploration, ax=None,
                           palette=(EXPLOIT_COLOR, EXPLORE_COLOR),
                           pc_indices=(0, 1), fontsize=FONT_SIZE_AXIS):
    """Scatter first two PCs colored by exploration/exploitation; marker = arm."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))
    colors = np.where(is_exploration, palette[1], palette[0])
    for a, marker in zip([1, 2, 3], _ARM_MARKERS):
        idx = actions.astype(int) == a
        ax.scatter(pca_xy[idx, pc_indices[0]], pca_xy[idx, pc_indices[1]],
                   c=colors[idx], marker=marker, alpha=0.7)
    handles = [Line2D([0], [0], marker="o", lw=0, markerfacecolor=palette[0], markeredgecolor="black"),
               Line2D([0], [0], marker="o", lw=0, markerfacecolor=palette[1], markeredgecolor="black")]
    ax.legend(handles=handles, labels=["Exploit", "Explore"], fontsize=fontsize, loc="upper right")
    ax.set_xlabel(f"PC {pc_indices[0] + 1}", fontsize=fontsize)
    ax.set_ylabel(f"PC {pc_indices[1] + 1}", fontsize=fontsize)
    return ax


def _add_arm_legend(ax, fontsize):
    handles = [Line2D([0], [0], marker=m, lw=0, markerfacecolor="white", markeredgecolor="black")
               for m in _ARM_MARKERS]
    ax.legend(handles=handles, labels=["Left", "Middle", "Right"], title="Action",
              loc="upper left", fontsize=fontsize)


# --------------------------------------------------------------------------- #
# RDM heatmaps
# --------------------------------------------------------------------------- #
def build_rdm_matrix(rdm_obj):
    """Convert an rsatoolbox RDM object to a symmetric labeled DataFrame."""
    m = rdm_obj.to_df().pivot("act_fb_grp_2", "act_fb_grp_1", "dissimilarity")
    labs = sorted(m.index.union(m.columns))
    m = m.reindex(index=labs, columns=labs).combine_first(m.T)
    np.fill_diagonal(m.values, 0)
    return m


def style_rdm_axes(ax, n_value_bins, group_labels=("Left\nAction", "Middle\nAction", "Right\nAction"),
                   cbar_label=None, fontsize=10):
    """Add value-bin tick labels, action-group boundaries, and an optional colorbar."""
    ax.set_xticklabels([f"Value-{i + 1}" for i in range(n_value_bins)] * 3, rotation=45)
    ax.set_yticklabels([f"Value-{i + 1}" for i in range(n_value_bins)] * 3, rotation=30)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth(2)

    if cbar_label:
        hm = ax.collections[0]
        arr = hm.get_array()
        off_diag = arr[~np.eye(arr.shape[0], dtype=bool)]
        hm.set_clim(off_diag.min(), arr.max())
        cbar = ax.figure.colorbar(hm, ax=ax, fraction=0.03, pad=0.05)
        cbar.set_label(cbar_label, rotation=90, labelpad=10, fontsize=9, weight="bold")

    boundaries = [0, n_value_bins, n_value_bins * 2, n_value_bins * 3]
    centers = [(boundaries[i] + boundaries[i + 1] - 1) / 2 for i in range(len(group_labels))]
    trans_x = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    trans_y = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for c, lab in zip(centers, group_labels):
        ax.text(c, -0.18, lab, ha="center", va="top", transform=trans_x,
                fontsize=fontsize, weight="bold")
        ax.text(-0.185, c, lab, ha="right", va="center", transform=trans_y,
                fontsize=fontsize, weight="bold")
    for b in boundaries:
        ax.axvline(b, color="black", linewidth=2)
        ax.axhline(b, color="black", linewidth=2)


def plot_rdm(rdm_obj, n_value_bins, title, cbar_label="Dissimilarity", cmap="bone_r"):
    """Plot a full RDM heatmap with action/value annotations."""
    ax = sns.heatmap(build_rdm_matrix(rdm_obj), cmap=cmap, square=True, cbar=False)
    style_rdm_axes(ax, n_value_bins, cbar_label=cbar_label)
    plt.suptitle(title, weight="bold")
    return ax


# --------------------------------------------------------------------------- #
# Functional ROI glass brain
# --------------------------------------------------------------------------- #
def plot_functional_rois(maskers, roi_order, display_mode="xz"):
    """Overlay a dict of NiftiMaskers on a glass brain, colored per ROI."""
    import nilearn
    from matplotlib.colors import ListedColormap
    from nilearn.image import math_img
    from nilearn.plotting import plot_glass_brain

    cmap = ListedColormap([ROI_COLORS[r] for r in roi_order])
    roi_images = [math_img(f"img * {i + 1}", img=maskers[r].mask_img) for i, r in enumerate(roi_order)]
    combined = math_img("np.sum(imgs, axis=3)", imgs=nilearn.image.concat_imgs(roi_images))
    plot_glass_brain(combined, vmin=1, vmax=len(roi_order), cmap=cmap,
                     colorbar=False, display_mode=display_mode)


# --------------------------------------------------------------------------- #
# Bar + strip comparisons
# --------------------------------------------------------------------------- #
def bar_strip_pair(df, y, ax, ylab, title, *, ylim=None, chance=None,
                   palette=(GAIN_COLOR, LOSS_COLOR), hue_order=("Gain", "Loss"),
                   subject_col="subject", fontsize=FONT_SIZE_AXIS):
    """Gain vs Loss bar plot with subject strips."""
    from scipy.stats import ttest_rel

    sns.barplot(data=df, x="condition", y=y, hue="condition", palette=palette,
                hue_order=hue_order, errorbar="sd", ax=ax, legend=False)
    sns.stripplot(data=df, x="condition", y=y, color="black", alpha=0.6, jitter=True, ax=ax)
    if chance is not None:
        ax.axhline(chance, color="black", linestyle="dashed", label="Chance level")
    gain = df[df.condition == "Gain"][y]
    loss = df[df.condition == "Loss"][y]
    t, p = ttest_rel(gain, loss)
    if subject_col in df.columns:
        for subj in df[subject_col].unique():
            s = df[df[subject_col] == subj]
            if set(s["condition"]) >= set(hue_order):
                vals = s.set_index("condition").loc[list(hue_order), y].values
                ax.plot([0, 1], vals, color="gray", alpha=0.4, linewidth=0.8)
    ax.set_xlabel("")
    ax.set_ylabel(ylab, fontsize=fontsize)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(list(hue_order), fontsize=fontsize)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=FONT_SIZE_TITLE)
    print(f"{title} – Gain vs Loss: t={t:.2f}, p={p:.3f}")
    return t, p


# --------------------------------------------------------------------------- #
# PCA trajectories over trials
# --------------------------------------------------------------------------- #
def plot_pcs_over_trials(pca_xy, actions, *, ax=None, common_num_trials=None,
                         pc_indices=(0, 1), axis_label="PC", title=None,
                         fontsize=FONT_SIZE_AXIS):
    """PCA scatter over trials (trial color, arm marker)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))
    actions = np.asarray(actions).astype(int)
    if common_num_trials is not None:
        n_trials = common_num_trials
        n_subj = len(actions) // common_num_trials
        trial_idx = np.concatenate([np.arange(common_num_trials) for _ in range(n_subj)])
    else:
        n_trials = len(actions)
        trial_idx = np.arange(n_trials)
    colormap = plt.get_cmap("viridis")
    time_colors = colormap(np.linspace(0, 1, n_trials))
    for arm, marker in zip([1, 2, 3], _ARM_MARKERS):
        idx = actions == arm
        ax.scatter(pca_xy[idx, pc_indices[0]], pca_xy[idx, pc_indices[1]],
                   c=time_colors[trial_idx[idx]], marker=marker)
    sm = cm.ScalarMappable(cmap=colormap)
    sm.set_clim(vmin=0, vmax=n_trials)
    cbar = ax.figure.colorbar(sm, ax=ax)
    cbar.set_label("Trial", fontsize=fontsize)
    ax.set_xlabel(f"{axis_label} {pc_indices[0] + 1}", fontsize=fontsize)
    ax.set_ylabel(f"{axis_label} {pc_indices[1] + 1}", fontsize=fontsize)
    _add_arm_legend(ax, fontsize)
    if title:
        ax.set_title(title, fontsize=fontsize)
    return ax

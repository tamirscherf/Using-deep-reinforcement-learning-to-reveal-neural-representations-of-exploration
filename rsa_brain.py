"""Representational similarity between the deep-RL model and fMRI ROIs.

Functional ROI masks, neural and candidate model RDMs, and ROI-wise model correlations.
RSA categories are three arms × five value quantiles (``A{arm}_fb_{bin}``).
"""

from __future__ import annotations

import copy
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rsatoolbox
import seaborn as sns

import paths
import plotting_utils as pu
import rsa_utils as ru

N_VALUE_BINS = ru.N_VALUE_BINS
FB_ROIS = ru.FB_ROIS
ACT_ROIS = ru.ACT_ROIS
ALL_ROIS = ru.ALL_ROIS
MODEL_ORDER = ru.MODEL_ORDER

MODEL_LABELS = {
    "lstm_subj": "Deep-RL",
    "lstm_subj_shf": "Deep-RL (shuffle)",
    "act_fb": "Action-Value",
    "fb": "Value",
    "spatial": "Choice-location",
}

permute_model = ru.permute_model
evaluate_neural_model_correlations = ru.evaluate_neural_model_correlations
rsa_paired_ttests = ru.rsa_paired_ttests
lstm_rdms_to_models = ru.lstm_rdms_to_models


# ========================================================================== #
# Data loading (precomputed neural RDMs and model-correlation table)
# ========================================================================== #
def load_neural_rdms(distance_measure=paths.NEURAL_RDM_DISTANCE, n_bins=N_VALUE_BINS):
    """Per-ROI, per-subject neural RDMs (dict[cond][roi][subj] -> rsatoolbox RDM)."""
    return paths.load_pickle(paths.neural_rdms_path(n_bins, distance_measure))


def load_model_correlations(n_bins=N_VALUE_BINS):
    """Table of each subject/ROI neural RDM's Spearman correlation to each model."""
    df = paths.load_pickle(paths.MULTI_MODEL_DF)
    df["condition"] = df["condition"].replace({"rew": "Gain", "pun": "Loss"})
    return df


def load_functional_maskers():
    """Load fitted NiftiMaskers for the five functional ROIs (from fmri_pipeline)."""
    return paths.load_pickle(paths.FUNCTIONAL_MASKERS)


# ========================================================================== #
def _category_labels(n_bins=N_VALUE_BINS):
    return [f"A{a}_fb_{f}" for a in [1, 2, 3] for f in range(1, n_bins + 1)]


def _float_rdm(rdm):
    rdm = copy.deepcopy(rdm)
    rdm.dissimilarities = rdm.dissimilarities.astype(float)
    return rdm


def build_naive_models(n_bins=N_VALUE_BINS):
    """Construct the Value, Action-Value and Choice-location model RDMs.

    * Value model: dissimilarity grows with the difference in value bin;
    * Choice-location model: dissimilarity is the distance between arm positions;
    * Action-Value model: sum of the value and action components.
    """
    from rsatoolbox.rdm import RDMs

    conds = _category_labels(n_bins)
    pairs = list(combinations(conds, 2))

    action = [int(c1.split("_")[0] != c2.split("_")[0]) for c1, c2 in pairs]
    value = []
    for c1, c2 in pairs:
        f1, f2 = int(c1.split("_")[2]), int(c2.split("_")[2])
        value.append(1 if f1 == f2 else abs(f1 - f2) + 1)
    spatial = [abs(float(c1.split("_")[0][1]) - float(c2.split("_")[0][1])) for c1, c2 in pairs]

    def _rdm(vec, name):
        return RDMs(np.array(vec, dtype=float)[None, :], dissimilarity_measure="diss",
                    pattern_descriptors={"act_fb_grp": conds}, rdm_descriptors={"type": [name]})

    rdm_value = _rdm(value, "Value model")
    rdm_action = _rdm(action, "Action model")
    rdm_action_value = _rdm(np.array(value) + np.array(action), "Action-Value model")
    rdm_spatial = _rdm(spatial, "Choice-location model")

    return {
        "fb": rsatoolbox.model.ModelFixed("fb", _float_rdm(rdm_value)),
        "act_fb": rsatoolbox.model.ModelFixed("act_fb", _float_rdm(rdm_action_value)),
        "spatial": rsatoolbox.model.ModelFixed("spatial", _float_rdm(rdm_spatial)),
    }, {"Value model": rdm_value, "Action-Value model": rdm_action_value,
        "Choice-location model": rdm_spatial}


def build_lstm_rdm(lstm_categories):
    """Build the deep-RL (LSTM) RDM from mean per-category latent representations.

    ``lstm_categories`` is a DataFrame with columns ``subject``, ``condition``,
    ``act_fb_grp`` and ``LSTM_vals`` (mean activation vector over selected units
    for that category). Uses Euclidean distance.
    """
    import rsatoolbox.rdm as rsr
    from rsatoolbox import data as rsd

    rdm_dict = {}
    for cond in ["Gain", "Loss"]:
        rdm_dict[cond] = {}
        for subj in lstm_categories.subject.unique():
            d = lstm_categories[(lstm_categories.subject == subj) &
                                (lstm_categories.condition == cond)]
            if d.empty:
                continue
            dataset = rsd.Dataset(
                measurements=np.array(d.LSTM_vals.to_list()),
                obs_descriptors={"act_fb_grp": d.act_fb_grp.values},
                descriptors={"subj": int(subj), "condition": cond})
            rdm_dict[cond][subj] = rsr.calc_rdm(dataset, descriptor="act_fb_grp", method="euclidean")
    return rdm_dict


# ========================================================================== #
# Plotting
# ========================================================================== #
def _dict2list(d):
    return [d[k] for k in d.keys()]


def plot_functional_rois(maskers):
    """Functional ROIs on a glass brain."""
    pu.plot_functional_rois(maskers, ALL_ROIS)


def plot_neural_rdms(neural_rdms, rois=("V1", "dACC", "vmPFC")):
    """Mean neural RDMs for example ROIs."""
    import rsatoolbox.rdm as rsr
    for roi in rois:
        mean_rdm = rsr.concat(
            _dict2list(neural_rdms["rew"][roi]) + _dict2list(neural_rdms["pun"][roi])).mean()
        plt.figure()
        pu.plot_rdm(mean_rdm, N_VALUE_BINS, roi, cbar_label="Dissimilarity\n(Mahalanobis)")


def plot_candidate_rdms(model_rdms, lstm_rdm_mean=None):
    """Candidate model RDMs and optional deep-RL RDM."""
    if lstm_rdm_mean is not None:
        plt.figure()
        pu.plot_rdm(lstm_rdm_mean, N_VALUE_BINS, "Deep-RL model",
                    cbar_label="Dissimilarity\n(Euclidean)")
    for name, rdm in model_rdms.items():
        plt.figure()
        pu.plot_rdm(rdm, N_VALUE_BINS, name, cbar_label="Dissimilarity")


def plot_rsa_correlations(model_corr_df, ttest_df=None):
    """Spearman correlation of neural RDMs to candidate models."""
    palette = dict(zip(MODEL_ORDER, sns.color_palette(n_colors=len(MODEL_ORDER))))
    data = model_corr_df[model_corr_df.model.isin(MODEL_ORDER)]
    g = sns.catplot(data=data, x="roi", y="corr2model", hue="model", kind="bar",
                    col="condition", height=5.5, aspect=1.5, palette=palette,
                    hue_order=MODEL_ORDER)
    g._legend.set_title("Model")
    for t in g._legend.texts:
        t.set_text(MODEL_LABELS.get(t.get_text(), t.get_text()))
    if ttest_df is not None:
        try:
            from statannotations.Annotator import Annotator
        except ImportError:
            Annotator = None
        if Annotator is not None:
            for ax in g.axes.flat:
                condition = ax.get_title().split(" = ")[-1]
                pairs, stars = [], []
                for roi in ALL_ROIS:
                    comp = ("lstm_subj_vs_act_fb" if roi in FB_ROIS
                            else "lstm_subj_vs_spatial")
                    row = ttest_df[(ttest_df.condition == condition) &
                                   (ttest_df.roi == roi) &
                                   (ttest_df.roi_model_comp == comp)]
                    if row.empty:
                        continue
                    ref = "act_fb" if roi in FB_ROIS else "spatial"
                    pairs.append(((roi, "lstm_subj"), (roi, ref)))
                    stars.append(row["stars_fdr"].values[0])
                if pairs:
                    annotator = Annotator(
                        ax=ax, pairs=pairs,
                        data=data[data.condition == condition],
                        x="roi", y="corr2model", hue="model", hue_order=MODEL_ORDER)
                    annotator.configure(test=None, verbose=False)
                    annotator.set_custom_annotations(stars)
                    annotator.annotate()
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=pu.FONT_SIZE_AXIS)
        ax.set_ylabel("Spearman correlation", fontsize=pu.FONT_SIZE_AXIS)
        ax.set_xlabel("")
    g.fig.suptitle("Neural RDM correlation to models", weight="bold", y=1.02)
    return g


# ========================================================================== #
# RSA tables
# ========================================================================== #
def compute_rsa_tables(n_bins=N_VALUE_BINS):
    """Evaluate neural-vs-model RSA correlations and paired t-tests."""
    import data_pipeline as dp

    neural_rdms = dp.get_neural_rdms()
    naive_models, _ = build_naive_models(n_bins)

    if paths.LSTM_CATEGORIES.exists():
        lstm_categories = paths.load_pickle(paths.LSTM_CATEGORIES)
        lstm_rdm_dict = build_lstm_rdm(lstm_categories)
        lstm_models = lstm_rdms_to_models(lstm_rdm_dict)
        model_corr_df = evaluate_neural_model_correlations(
            neural_rdms, naive_models, lstm_models)
    else:
        raise FileNotFoundError(f"Required file not found: {paths.LSTM_CATEGORIES}")

    ttest_df = rsa_paired_ttests(model_corr_df)
    paths.save_pickle(paths.MULTI_MODEL_DF, model_corr_df)
    paths.save_pickle(paths.RSA_PAIRED_TTEST_DF, ttest_df)
    return model_corr_df, ttest_df


# ========================================================================== #
# Runner
# ========================================================================== #
def main(force: bool = False):
    import data_pipeline as dp

    neural_rdms = dp.get_neural_rdms(force=force)
    maskers = dp.get_functional_maskers(force=force)
    naive_models, naive_rdms = build_naive_models()

    if force or not paths.MULTI_MODEL_DF.exists():
        model_corr_df, ttest_df = compute_rsa_tables()
    else:
        model_corr_df = load_model_correlations()
        ttest_df = paths.load_pickle(paths.RSA_PAIRED_TTEST_DF)

    lstm_rdm_mean = None
    if paths.LSTM_CATEGORIES.exists():
        lstm_categories = paths.load_pickle(paths.LSTM_CATEGORIES)
        lstm_rdm_dict = build_lstm_rdm(lstm_categories)
        import rsatoolbox.rdm as rsr
        lstm_rdm_mean = rsr.concat(
            [rdm for cond in lstm_rdm_dict for rdm in lstm_rdm_dict[cond].values()]).mean()

    plot_functional_rois(maskers)
    plot_neural_rdms(neural_rdms)
    plot_candidate_rdms(naive_rdms, lstm_rdm_mean=lstm_rdm_mean)
    plot_rsa_correlations(model_corr_df, ttest_df=ttest_df)
    plt.show()


if __name__ == "__main__":
    main()

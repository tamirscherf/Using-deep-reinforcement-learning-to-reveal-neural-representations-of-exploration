"""Shared RSA helpers (model shuffle, correlations, sub-RDM decomposition, stats)."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import rsatoolbox
from rsatoolbox.model import ModelFixed
from scipy.stats import spearmanr, ttest_rel
from statsmodels.stats.multitest import multipletests

N_SUBJ = 31
N_VALUE_BINS = 5
FB_ROIS = ["dACC", "vmPFC", "Insula"]
ACT_ROIS = ["V1", "FUSF"]
ALL_ROIS = FB_ROIS + ACT_ROIS
MODEL_ORDER = ["fb", "act_fb", "spatial", "lstm_subj", "lstm_subj_shf"]

SKIP_SUBJ7_PUN = {("pun", 7)}


def permute_model(model):
    """Shuffle dissimilarities within a fixed model RDM (deep-RL null)."""
    return ModelFixed(name=f"{model.name}_permuted",
                      rdm=np.random.permutation(model.rdm))


def lstm_rdms_to_models(rdm_dict):
    """Wrap per-subject LSTM RDMs as rsatoolbox fixed models (rew/pun keys)."""
    out = {"rew": {}, "pun": {}}
    for gain_loss, cond in (("Gain", "rew"), ("Loss", "pun")):
        for subj, rdm in rdm_dict.get(gain_loss, {}).items():
            out[cond][subj] = ModelFixed(f"lstm_{subj}", rdm)
    return out


def evaluate_neural_model_correlations(neural_rdms, naive_models, lstm_models,
                                       n_subj=N_SUBJ, skip=SKIP_SUBJ7_PUN,
                                       method="spearman"):
    """Spearman correlation of each subject's neural RDM to each candidate model."""
    from rsatoolbox.inference import eval_fixed

    rows = []
    for cond in ("rew", "pun"):
        for roi in ALL_ROIS:
            for subj in range(n_subj):
                if (cond, subj) in skip:
                    continue
                subj_rdm = neural_rdms[cond][roi][subj]
                candidates = [
                    ("fb", naive_models["fb"]),
                    ("act_fb", naive_models["act_fb"]),
                    ("lstm_subj", lstm_models[cond][subj]),
                    ("lstm_subj_shf", permute_model(lstm_models[cond][subj])),
                    ("spatial", naive_models["spatial"]),
                ]
                for model_name, model in candidates:
                    corr = eval_fixed(model, subj_rdm, method=method).evaluations[0][0][0]
                    rows.append({"subject": subj, "condition": cond, "model": model_name,
                                 "corr2model": corr, "roi": roi})
    df = pd.DataFrame(rows)
    df["condition"] = df["condition"].replace({"rew": "Gain", "pun": "Loss"})
    return df


def p_to_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def rsa_paired_ttests(multi_model_df):
    """Paired t-tests: deep-RL vs Action-Value (value ROIs) or vs Choice-location."""
    res = []
    for cond in multi_model_df["condition"].unique():
        for roi in multi_model_df["roi"].unique():
            df = multi_model_df[(multi_model_df.condition == cond) &
                                (multi_model_df.roi == roi)]
            x = df[df.model == "lstm_subj"]["corr2model"].values
            if roi in FB_ROIS:
                y = df[df.model == "act_fb"]["corr2model"].values
                comp = "lstm_subj_vs_act_fb"
            else:
                y = df[df.model == "spatial"]["corr2model"].values
                comp = "lstm_subj_vs_spatial"
            t, p = ttest_rel(x, y, nan_policy="omit")
            res.append(dict(condition=cond, roi=roi, t=t, p=p, roi_model_comp=comp))
    ttest_df = pd.DataFrame(res)
    ttest_df["stars"] = ttest_df["p"].apply(p_to_stars)
    for cond in ttest_df["condition"].unique():
        for comp in ttest_df["roi_model_comp"].unique():
            mask = (ttest_df.condition == cond) & (ttest_df.roi_model_comp == comp)
            _, p_fdr, _, _ = multipletests(ttest_df.loc[mask, "p"], alpha=0.05, method="fdr_bh")
            ttest_df.loc[mask, "p_fdr"] = p_fdr
            ttest_df.loc[mask, "stars_fdr"] = [p_to_stars(p) for p in p_fdr]
    return ttest_df


def _same_action(series):
    arms = series.str.extract(r"^(A\d+)_fb_\d+.*(A\d+)_fb_\d+")
    return arms[0].eq(arms[1])


def rdm_dict_to_long(rdm_dict, cond):
    """Flatten dict[subj]->RDM for one condition to a long DataFrame."""
    frames = [rdm.to_df() for subj, rdm in rdm_dict[cond].items()]
    df = pd.concat(frames, ignore_index=True)
    df["cat1_cat2"] = df["act_fb_grp_1"] + "_" + df["act_fb_grp_2"]
    df["fbgrp1_fbgrp2"] = (df["fb_group_1"].astype(str) + "_" +
                           df["fb_group_2"].astype(str))
    return df


def split_subrdm_tables(df, group_cols):
    """Within- and across-action sub-RDM tables (mean per value-pair)."""
    same = _same_action(df["cat1_cat2"])
    within = (df[same].groupby(group_cols)["dissimilarity"].mean().reset_index())
    across = (df[~same].groupby(group_cols)["dissimilarity"].mean().reset_index())
    return within, across


def get_within_vals(df, cond=None, subj=None, roi=None, value_col="dissimilarity"):
    if cond is not None:
        df = df[df.condition.eq(cond)]
    if subj is not None:
        df = df[df.subj.eq(subj)]
    if roi is not None:
        df = df[df.roi.eq(roi)]
    return df[value_col].values


def get_across_vals(df, cond=None, subj=None, roi=None, value_col="dissimilarity",
                    ravel=False):
    if cond is not None:
        df = df[df.condition.eq(cond)]
    if subj is not None:
        df = df[df.subj.eq(subj)]
    if roi is not None:
        df = df[df.roi.eq(roi)]
    mat = df.groupby(["fb_group_1", "fb_group_2"])[value_col].mean().unstack()
    return mat.values.ravel() if ravel else mat.values


def build_sub_rdm_correlations(neural_rdms, lstm_rdm_dict, act_fb_model):
    """Correlate neural within/across sub-RDMs with deep-RL and Action-Value models."""
    neural_long = []
    for cond in ("rew", "pun"):
        for roi in ALL_ROIS:
            for subj, rdm in neural_rdms[cond][roi].items():
                d = rdm.to_df()
                d["roi"] = roi
                neural_long.append(d)
    neural_df = pd.concat(neural_long, ignore_index=True)
    neural_df["cat1_cat2"] = neural_df["act_fb_grp_1"] + "_" + neural_df["act_fb_grp_2"]
    neural_df["fbgrp1_fbgrp2"] = (neural_df["fb_group_1"].astype(str) + "_" +
                                  neural_df["fb_group_2"].astype(str))
    within_subjects, across_subjects = split_subrdm_tables(
        neural_df, ["roi", "condition", "fbgrp1_fbgrp2", "subj"])

    lstm_long = pd.concat([rdm_dict_to_long(lstm_rdm_dict, c) for c in ("rew", "pun")],
                          ignore_index=True)
    within_lstm, across_lstm = split_subrdm_tables(
        lstm_long, ["condition", "fbgrp1_fbgrp2", "subj"])

    act_fb_rdm = act_fb_model.rdm if hasattr(act_fb_model, "rdm") else act_fb_model.rdm_obj
    act_fb_df = act_fb_rdm.to_df()
    act_fb_df["cat1_cat2"] = act_fb_df["act_fb_grp_1"] + "_" + act_fb_df["act_fb_grp_2"]
    act_fb_df["fbgrp1_fbgrp2"] = (act_fb_df["fb_group_1"].astype(str) + "_" +
                                  act_fb_df["fb_group_2"].astype(str))
    within_act_fb, across_act_fb = split_subrdm_tables(act_fb_df, ["fbgrp1_fbgrp2"])

    model_tables = {
        "within_subjects": within_subjects,
        "across_subjects": across_subjects,
        "within_lstm_subj": within_lstm,
        "across_lstm_subj": across_lstm,
        "within_act_fb": within_act_fb,
        "across_act_fb": across_act_fb,
    }

    rows = []
    for cond in ("rew", "pun"):
        for roi in ALL_ROIS:
            subjs = neural_rdms[cond][roi].keys()
            for subj in subjs:
                for sub_rdm_name in ("across", "within"):
                    neural_tbl = model_tables[f"{sub_rdm_name}_subjects"]
                    if sub_rdm_name == "within":
                        subj_rdm = get_within_vals(neural_tbl, cond, subj, roi)
                    else:
                        subj_rdm = get_across_vals(neural_tbl, cond, subj, roi, ravel=True)
                    for model_name in ("lstm_subj", "act_fb"):
                        m_tbl = model_tables[f"{sub_rdm_name}_{model_name}"]
                        if model_name == "lstm_subj":
                            model_vals = (get_within_vals(m_tbl, cond, subj) if sub_rdm_name == "within"
                                          else get_across_vals(m_tbl, cond, subj, ravel=True))
                        else:
                            model_vals = (get_within_vals(m_tbl) if sub_rdm_name == "within"
                                          else get_across_vals(m_tbl, ravel=True))
                        rho, _ = spearmanr(subj_rdm, model_vals)
                        rows.append({"model": model_name, "subject": subj, "condition": cond,
                                     "roi": roi, "sub_rdm_type": sub_rdm_name,
                                     "SUB_RDM_corr2model": rho})
    df = pd.DataFrame(rows)
    df["condition"] = df["condition"].replace({"rew": "Gain", "pun": "Loss"})
    return df


def sub_rdm_paired_ttests(sub_rdm_df, sub_type="within"):
    """Paired t-test deep-RL vs Action-Value on sub-RDM correlations per value ROI."""
    results = []
    for roi in FB_ROIS:
        data = sub_rdm_df[(sub_rdm_df.roi == roi) &
                          (sub_rdm_df.sub_rdm_type == sub_type)]
        models = data.model.unique()
        v1 = data[data.model == models[0]]["SUB_RDM_corr2model"].values
        v2 = data[data.model == models[1]]["SUB_RDM_corr2model"].values
        t, p = ttest_rel(v1, v2)
        results.append({"roi": roi, "sub_rdm_type": sub_type, "t": t, "p": p})
    return pd.DataFrame(results)

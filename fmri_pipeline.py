"""fMRI preprocessing and representational-analysis pipeline.

Builds functional ROI masks from group GLMs, per-subject beta maps and Mahalanobis
RDMs, and whole-brain searchlight maps (model fit and diagonal slope).
Uses Nilearn first/second-level GLMs with the SPM HRF.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import rsatoolbox.data as rsd
import rsatoolbox.rdm as rsr
from nilearn.glm import threshold_stats_img
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.image import binarize_img, index_img, load_img, math_img
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.maskers import NiftiMasker
from nilearn.regions import connected_regions
from scipy.stats import zscore

import paths

warnings.filterwarnings("ignore")

N_SUBJ = 31
N_VALUE_BINS = paths.N_VALUE_BINS
T_R = 2.0
CONFOUND_STRATEGY = ["motion", "high_pass", "wm_csf", "scrub"]

BRAIN_MASK_IMG = paths.BRAIN_MASK_IMG
STIMULUS_GROUP_MAP = paths.MASKS_DIR / "feedback_minus_choice_group.pkl"

SKIP = {("pun", 7)}

# Cluster-extraction settings; set these before running the mask pipeline.
MIN_CLUSTER_SIZE = None
VALUE_CLUSTER_IDX = {
    "dACC": None,
    "vmPFC": None,
    "Insula": (None, None),
}
STIMULUS_CLUSTER_IDX = {
    "V1": (None, None),
    "FUSF": (None, None),
}


# ========================================================================== #
# Event construction
# ========================================================================== #
def get_invalid_events(events, behavior):
    """Relabel trials with no response (choice == -1) as 'invalid' regressors."""
    out = events[["onset", "duration"]].copy()
    invalid_idx = np.where(behavior.choice == -1)[0]
    new_types, counter = [], 0
    for trial_type in events["trial_type"]:
        if trial_type == "stimuli":
            label = ["invalid"] * 3 if counter in invalid_idx else ["stimuli", "choice", "feedback"]
            new_types += label
            counter += 1
    out.insert(2, "trial_type", new_types)
    return out


def add_value_modulation(behavior, events, fill=1.0):
    """Add a z-scored feedback (value) parametric modulator at feedback onset.

    Used for the group value-modulation functional mask.
    """
    valid = np.where(behavior.choice > -1)[0]
    fb = behavior.FB[valid].to_numpy()
    zfb = (fb - fb.mean()) / fb.std()
    events = get_invalid_events(events, behavior)
    new = events[["onset", "duration"]][events["trial_type"] == "feedback"].copy()
    new.insert(2, "trial_type", ["fb_modul"] * len(zfb))
    new.insert(3, "modulation", zfb)
    out = pd.concat([events, new], ignore_index=True)
    out["modulation"] = out["modulation"].fillna(fill)
    return out


def make_category_events(behavior, events, n_bins=N_VALUE_BINS, duration=1):
    """Relabel each valid feedback event as its RSA category ``A{arm}_fb_{bin}``.

    Value bins are equal-count quantiles computed *within each arm*, giving a
    balanced number of trials per category.
    """
    valid = behavior[behavior.choice > 0][["choice", "FB"]].copy()
    valid["fb_grp"] = valid.groupby("choice")["FB"].transform(
        lambda x: pd.qcut(x, q=n_bins, labels=False, retbins=False).to_numpy() + 1)
    valid["category"] = valid.apply(lambda r: f"A{int(r.choice)}_fb_{int(r.fb_grp)}", axis=1)
    out = get_invalid_events(events, behavior)
    out.loc[out["trial_type"] == "feedback", "duration"] = duration
    out.loc[out["trial_type"] == "feedback", "trial_type"] = valid["category"].to_numpy()
    return out


# ========================================================================== #
# 1. Functional masks (group-level GLM)
# ========================================================================== #
def _first_level_model():
    return FirstLevelModel(t_r=T_R, hrf_model="spm", slice_time_ref=0.5,
                           signal_scaling=0, drift_model=None,
                           mask_img=str(BRAIN_MASK_IMG), minimize_memory=False)


def value_modulation_group_map(images, behavior, events, tasks=("Gain", "Loss")):
    """Second-level z-map of the value (feedback) modulation, pooled over conditions.

    ``images``/``behavior``/``events`` are dicts keyed by task, each a per-subject
    list. The second-level design is a single group column plus per-subject
    dummies (paired across the two conditions).
    """
    subject_maps = []
    for task in tasks:
        for subj in range(N_SUBJ):
            fl = _first_level_model()
            subj_events = add_value_modulation(behavior[task][subj], events[task][subj])
            file_path = _bold_path(subj, task)
            conf, sample_mask = load_confounds(str(file_path), strategy=CONFOUND_STRATEGY,
                                               fd_threshold=0.5)
            fl = fl.fit(images[task][subj], events=subj_events,
                        sample_masks=sample_mask, confounds=conf)
            subject_maps.append(fl.compute_contrast("fb_modul", output_type="effect_size"))

    condition_effect = np.ones(len(subject_maps))
    subject_effect = np.vstack([np.eye(N_SUBJ)] * len(tasks))
    design = pd.DataFrame(
        np.hstack([condition_effect[:, None], subject_effect]),
        columns=["all"] + [f"Subj{i}" for i in range(N_SUBJ)])
    second_level = SecondLevelModel().fit(subject_maps, design_matrix=design)
    return second_level.compute_contrast("all", output_type="z_score")


def _merge_clusters(regions, indices):
    """Binarize and union one or more connected-component clusters."""
    imgs = [binarize_img(index_img(regions, i)) for i in np.atleast_1d(indices)]
    out = imgs[0]
    for img in imgs[1:]:
        out = math_img("i1 + i2", i1=out, i2=img)
    return binarize_img(out)


def make_value_masks(value_group_zmap, alpha=0.005,
                     min_region_size=MIN_CLUSTER_SIZE,
                     cluster_idx=VALUE_CLUSTER_IDX):
    """Value ROIs (dACC, vmPFC, Insula) from the value-modulation group map.

    Thresholds at FDR<``alpha`` (two-sided), extracts connected clusters, and
    combines the clusters listed in ``cluster_idx`` into each anatomical ROI.
    """
    thresholded, _ = threshold_stats_img(value_group_zmap, alpha=alpha,
                                          height_control="fdr", two_sided=True)
    regions, _ = connected_regions(thresholded, min_region_size=min_region_size,
                                   extract_type="connected_components")
    return {name: _merge_clusters(regions, idx)
            for name, idx in cluster_idx.items()}


def make_stimulus_masks(feedback_minus_choice_zmap, alpha=0.005,  min_region_size=MIN_CLUSTER_SIZE,
                        cluster_idx=STIMULUS_CLUSTER_IDX):
    """Stimulus ROIs (V1, FUSF) from the feedback>choice contrast group map."""
    thresholded, _ = threshold_stats_img(feedback_minus_choice_zmap, alpha=alpha,
                                          height_control="fdr", two_sided=True)
    regions, _ = connected_regions(thresholded, min_region_size=min_region_size,
                                   extract_type="connected_components")
    return {name: _merge_clusters(regions, idx)
            for name, idx in cluster_idx.items()}


def build_functional_maskers(value_masks, stimulus_masks):
    """Fit a NiftiMasker for every functional ROI and return the combined dict."""
    maskers = {}
    for name, img in {**value_masks, **stimulus_masks}.items():
        maskers[name] = NiftiMasker(img).fit()
    return maskers


# ========================================================================== #
# 2. First-level GLM: per-category beta maps
# ========================================================================== #
def _bold_path(subj, task):
    return paths.FMRIPREP_DIR / (
        f"sub-{subj + 1:02}/func/"
        f"sub-{subj + 1:02}_task-bandit{task}"
        f"_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")


def _events_path(subj, task):
    return paths.FMRIPREP_DIR / (
        f"sub-{subj + 1:02}/func/"
        f"sub-{subj + 1:02}_task-bandit{task}_events.tsv")


def load_fmri_inputs():
    """Load fMRIPrep BOLD images, event files, and behavioral pickles."""
    import task_behavior as tb

    images = {"Gain": [], "Loss": []}
    events = {"Gain": [], "Loss": []}
    for task in ("Gain", "Loss"):
        for subj in range(N_SUBJ):
            images[task].append(load_img(str(_bold_path(subj, task))))
            events[task].append(pd.read_csv(_events_path(subj, task), sep="\t"))
    behavior_gain, behavior_loss = tb.load_behavior()
    behavior = {"Gain": behavior_gain, "Loss": behavior_loss}
    return images, behavior, events


def _cluster_indices_configured(cluster_idx) -> bool:
    return all(v is not None for v in cluster_idx.values())


def _compute_masks(images, behavior, events, force: bool = False):
    """Functional ROI masks and maskers."""
    if not force and paths.FUNCTIONAL_MASKERS.exists():
        return paths.load_pickle(paths.FUNCTIONAL_MASKERS)

    if not _cluster_indices_configured(VALUE_CLUSTER_IDX):
        raise ValueError(
            "Set VALUE_CLUSTER_IDX in fmri_pipeline.py before computing masks.")

    fb_mod = value_modulation_group_map(images, behavior, events)
    paths.save_pickle(paths.SECOND_LEVEL_FB_MODULATION, fb_mod)

    value_masks = make_value_masks(fb_mod)
    stimulus_masks = {}
    if STIMULUS_GROUP_MAP.exists():
        stim_zmap = paths.load_pickle(STIMULUS_GROUP_MAP)
        if _cluster_indices_configured(STIMULUS_CLUSTER_IDX):
            stimulus_masks = make_stimulus_masks(stim_zmap)

    maskers = build_functional_maskers(value_masks, stimulus_masks)
    paths.save_pickle(paths.FUNCTIONAL_MASKERS, maskers)
    return maskers


def compute_beta_maps(images, behavior, events, n_bins=N_VALUE_BINS,
                      maps_type="effect_size", save=True):
    """Fit per-subject first-level GLMs and estimate one beta map per RSA category.

    Also returns each subject's residual time series and GLM degrees of freedom,
    which are required for Mahalanobis noise whitening.
    """
    categories = [f"A{a}_fb_{f}" for a in [1, 2, 3] for f in range(1, n_bins + 1)]
    beta_maps = {c: {cat: {} for cat in categories} for c in ("rew", "pun")}
    residuals = {c: {} for c in ("rew", "pun")}
    dof = {c: {} for c in ("rew", "pun")}

    for task, cond in (("Gain", "rew"), ("Loss", "pun")):
        for subj in range(N_SUBJ):
            if (cond, subj) in SKIP:
                continue
            subj_events = make_category_events(behavior[task][subj], events[task][subj], n_bins)
            conf, sample_mask = load_confounds(str(_bold_path(subj, task)),
                                               strategy=CONFOUND_STRATEGY,
                                               fd_threshold=0.5)
            fl = _first_level_model().fit(images[task][subj], events=subj_events,
                                          sample_masks=sample_mask, confounds=conf)
            residuals[cond][subj] = fl.residuals[0].get_fdata()
            dof[cond][subj] = fl.design_matrices_[0].shape[1]
            for cat in categories:
                beta_maps[cond][cat][subj] = fl.compute_contrast(cat, output_type=maps_type)

    if save:
        paths.save_pickle(paths.BETA_MAPS_PKL, beta_maps)
        paths.save_pickle(paths.BETA_RESIDUALS_PKL, residuals)
        paths.save_pickle(paths.BETA_DOF_PKL, dof)
        _save_beta_maps(beta_maps, n_bins)
    return beta_maps, residuals, dof


def _load_or_compute_beta_maps(images, behavior, events, force: bool = False):
    if not force and paths.BETA_MAPS_PKL.exists():
        return (paths.load_pickle(paths.BETA_MAPS_PKL),
                paths.load_pickle(paths.BETA_RESIDUALS_PKL),
                paths.load_pickle(paths.BETA_DOF_PKL))
    return compute_beta_maps(images, behavior, events, save=True)


def _save_beta_maps(beta_maps, n_bins):
    out_dir = paths.BETA_MAPS_DIR / f"bins_{n_bins}"
    for cond in beta_maps:
        for cat in beta_maps[cond]:
            (out_dir / cat).mkdir(parents=True, exist_ok=True)
            for subj, img in beta_maps[cond][cat].items():
                img.to_filename(out_dir / cat / f"{cat}_{cond}_subj{subj}.nii.gz")


# ========================================================================== #
# 3. Per-ROI Mahalanobis RDMs
# ========================================================================== #
def _clip_betas_zscore(betas):
    """Winsorize per-ROI betas at +/-3 z."""
    z = zscore(betas, nan_policy="omit")
    return np.where(z > 3, np.mean(betas[z <= 3]),
                    np.where(z < -3, np.mean(betas[z >= -3]), betas))


def build_roi_datasets(beta_maps, maskers, n_bins=N_VALUE_BINS):
    """Extract z-scored, clipped beta patterns per ROI into rsatoolbox Datasets."""
    categories = [f"A{a}_fb_{f}" for a in [1, 2, 3] for f in range(1, n_bins + 1)]
    datasets = {c: {roi: {} for roi in maskers} for c in ("rew", "pun")}
    for cond in ("rew", "pun"):
        for roi, masker in maskers.items():
            for subj in beta_maps[cond][categories[0]]:
                if (cond, subj) in SKIP:
                    continue
                vals = np.stack([masker.transform(beta_maps[cond][cat][subj]).ravel()
                                 for cat in categories])
                vals = np.stack([zscore(_clip_betas_zscore(v)) for v in vals])
                datasets[cond][roi][subj] = rsd.Dataset(
                    measurements=vals,
                    obs_descriptors={"act_fb_grp": np.array(categories)},
                    descriptors={"subj": subj, "roi": roi, "condition": cond})
    return datasets


def compute_rdms(datasets, residuals, dof, maskers, distance="mahalanobis", save=True,
                 n_bins=N_VALUE_BINS):
    """Per-ROI, per-subject RDMs; Mahalanobis uses residual-derived precision."""
    rdm_dict = {c: {roi: {} for roi in maskers} for c in ("rew", "pun")}
    for cond in ("rew", "pun"):
        for roi, masker in maskers.items():
            roi_mask = masker.mask_img_.get_fdata().astype(bool)
            for subj, dataset in datasets[cond][roi].items():
                if distance == "mahalanobis":
                    resid = residuals[cond][subj][roi_mask, :].T
                    precision = rsd.noise.prec_from_residuals(resid, dof[cond][subj])
                    rdm_dict[cond][roi][subj] = rsr.calc_rdm(
                        dataset, descriptor="act_fb_grp", method=distance, noise=precision)
                else:
                    rdm_dict[cond][roi][subj] = rsr.calc_rdm(
                        dataset, descriptor="act_fb_grp", method=distance)
    if save:
        paths.RSA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        paths.save_pickle(paths.neural_rdms_path(n_bins, distance), rdm_dict)
    return rdm_dict


def _beta_maps_per_subj(beta_maps, n_bins=N_VALUE_BINS):
    categories = [f"A{a}_fb_{f}" for a in [1, 2, 3] for f in range(1, n_bins + 1)]
    out = {"rew": {}, "pun": {}}
    for cond in ("rew", "pun"):
        subjs = beta_maps[cond][categories[0]].keys()
        for subj in subjs:
            out[cond][subj] = [beta_maps[cond][cat][subj] for cat in categories]
    return out


def _exploration_slopes_by_cond(episode_df):
    slopes = {"rew": {}, "pun": {}}
    for _, row in episode_df.iterrows():
        key = "rew" if row.condition == "Gain" else "pun"
        slopes[key][int(row.subject)] = row.exploration_slope
    return slopes


def _searchlight_candidate_models():
    """RSA candidate models for searchlight analysis."""
    from rsa_brain import build_lstm_rdm, build_naive_models
    from rsa_utils import lstm_rdms_to_models

    naive_models, _ = build_naive_models()
    models = {k: v for k, v in naive_models.items()}
    if paths.LSTM_CATEGORIES.exists():
        lstm_rdm_dict = build_lstm_rdm(paths.load_pickle(paths.LSTM_CATEGORIES))
        lstm_models = lstm_rdms_to_models(lstm_rdm_dict)
        models["lstm_subj"] = lstm_models
    return models


def _run_searchlight_stage(beta_maps, maskers, episode_df, force: bool = False):
    if not force and all(
        p.exists() for p in (
            paths.SEARCHLIGHT_MODEL_MAPS,
            paths.SEARCHLIGHT_SECOND_LEVEL,
            paths.SEARCHLIGHT_DIAG_SLOPE_MAPS,
            paths.EXPLORATION_CORRELATION_IMG,
        )
    ):
        return

    brain_mask = load_img(str(BRAIN_MASK_IMG))
    beta_per_subj = _beta_maps_per_subj(beta_maps)
    models = _searchlight_candidate_models()

    sl_df = searchlight_model_maps(beta_per_subj, models, brain_mask)
    paths.save_pickle(paths.SEARCHLIGHT_MODEL_MAPS, sl_df)

    second_level = searchlight_second_level_contrasts(sl_df)
    paths.save_pickle(paths.SEARCHLIGHT_SECOND_LEVEL, second_level)

    diag_df = searchlight_diag_slope_maps(beta_per_subj, brain_mask)
    paths.save_pickle(paths.SEARCHLIGHT_DIAG_SLOPE_MAPS, diag_df)

    cortex_masker = NiftiMasker(mask_img=brain_mask).fit()
    exploration_slopes = _exploration_slopes_by_cond(episode_df)
    corr_img, pval_img = whole_brain_exploration_correlation(
        diag_df, exploration_slopes, cortex_masker)
    paths.save_pickle(paths.EXPLORATION_CORRELATION_IMG, {
        "correlation_img": corr_img,
        "correlation_pval": pval_img,
    })


def run_fmri_pipeline(force: bool = False, through: str = "searchlight"):
    """Run the fMRI pipeline through ``through`` ('masks', 'rdms', or 'searchlight')."""
    import data_pipeline as dp

    images, behavior, events = load_fmri_inputs()
    maskers = _compute_masks(images, behavior, events, force=force)

    if through == "masks":
        return {"maskers": maskers}

    beta_maps, residuals, dof = _load_or_compute_beta_maps(
        images, behavior, events, force=force)
    datasets = build_roi_datasets(beta_maps, maskers)
    rdm_dict = compute_rdms(datasets, residuals, dof, maskers, save=True)

    if through == "rdms":
        return {"maskers": maskers, "neural_rdms": rdm_dict}

    episode_df = dp.get_episode_summaries()
    _run_searchlight_stage(beta_maps, maskers, episode_df, force=force)
    return {
        "maskers": maskers,
        "neural_rdms": rdm_dict,
        "searchlight": dp.get_searchlight_bundle(force=False),
    }


# ========================================================================== #
# 4. Whole-brain searchlight RSA
# ========================================================================== #
def _prep_beta_stack(beta_list):
    data = np.stack([im.get_fdata().squeeze() for im in beta_list])
    return np.nan_to_num(data.reshape(data.shape[0], -1))


def searchlight_model_maps(beta_maps_per_subj, models, brain_mask_img,
                           radius=3, rdm_method="euclidean"):
    """Per-subject whole-brain maps of model-RDM fit (Spearman) for each model.

    ``beta_maps_per_subj`` is dict[cond][subj] -> ordered list of category beta
    images. ``models`` maps a name to an rsatoolbox fixed model (may be
    subject-specific, e.g. the deep-RL model).
    """
    from rsatoolbox.inference import eval_fixed
    from rsatoolbox.util.searchlight import (evaluate_models_searchlight,
                                             get_searchlight_RDMs,
                                             get_volume_searchlight)

    mask_data = brain_mask_img.get_fdata()
    centers, neighbors = get_volume_searchlight(mask_data, radius=radius, threshold=0.5)
    x, y, z = mask_data.shape

    rows = []
    for cond in beta_maps_per_subj:
        for subj, beta_list in beta_maps_per_subj[cond].items():
            sl_rdms = get_searchlight_RDMs(_prep_beta_stack(beta_list), centers, neighbors,
                                           np.arange(len(beta_list)), method=rdm_method)
            sl_rdms.dissimilarities = np.nan_to_num(sl_rdms.dissimilarities)
            subj_models = {k: (v[cond][subj] if isinstance(v, dict) else v)
                           for k, v in models.items()}
            evals = evaluate_models_searchlight(
                sl_rdms, list(subj_models.values()), eval_fixed, method="spearman")
            scores = np.array([e.evaluations.flatten() for e in evals]).T
            for i, name in enumerate(subj_models):
                brain = np.zeros(x * y * z)
                brain[list(sl_rdms.rdm_descriptors["voxel_index"])] = scores[i]
                rows.append({"subject": subj, "condition": cond, "model": name,
                             "sl_map": math_img("a",
                                                a=_reshape_to_img(brain, brain_mask_img))})
    return pd.DataFrame(rows)


def _reshape_to_img(flat, ref_img):
    from nilearn.image import new_img_like
    x, y, z = ref_img.get_fdata().shape
    return new_img_like(ref_img, flat.reshape(x, y, z))


def _create_paired_design_matrix(n_subjects):
    condition_effect = np.hstack(([1] * n_subjects, [-1] * n_subjects))
    subject_effect = np.vstack((np.eye(n_subjects), np.eye(n_subjects)))
    return pd.DataFrame(
        np.hstack((condition_effect[:, np.newaxis], subject_effect)),
        columns=["cond1_VS_cond2"] + [f"Subj{i + 1}" for i in range(n_subjects)])


def compute_second_level_map_from_maps_list(maps, output_type="z_score"):
    design = pd.DataFrame([1] * len(maps), columns=["intercept"])
    model = SecondLevelModel().fit(maps, design_matrix=design)
    return model.compute_contrast(output_type=output_type)


def compute_second_level_diff_paired_design(list_maps_1, list_maps_2, output_type="z_score"):
    design = _create_paired_design_matrix(len(list_maps_1))
    model = SecondLevelModel().fit(list_maps_1 + list_maps_2, design_matrix=design)
    return model.compute_contrast("cond1_VS_cond2", output_type=output_type)


def searchlight_second_level_contrasts(sl_df, naive_model_names=("act_fb", "fb", "spatial"),
                                       lstm_name="lstm_subj"):
    """Group-level searchlight maps and deep-RL vs naive paired contrasts."""
    rows = []
    for cond in ("rew", "pun"):
        for model_name in sl_df.model.unique():
            maps = sl_df[(sl_df.model == model_name) & (sl_df.condition == cond)].sl_map.tolist()
            rows.append({"condition": cond, "model": model_name,
                           "map": compute_second_level_map_from_maps_list(maps)})
        for naive in naive_model_names:
            lstm_maps = sl_df[(sl_df.model == lstm_name) & (sl_df.condition == cond)].sl_map.tolist()
            naive_maps = sl_df[(sl_df.model == naive) & (sl_df.condition == cond)].sl_map.tolist()
            rows.append({"condition": cond, "model": f"{lstm_name}_vs_{naive}",
                           "map": compute_second_level_diff_paired_design(lstm_maps, naive_maps)})
    return pd.DataFrame(rows)


def _diag_index_pairs(n_bins=N_VALUE_BINS):
    labels = [f"A{a}_fb_{f}" for a in (1, 2, 3) for f in range(1, n_bins + 1)]
    pairs = []
    for i1 in range(n_bins):
        pairs.append((i1, i1 + n_bins))
        pairs.append((i1, i1 + 2 * n_bins))
        pairs.append((i1 + n_bins, i1 + 2 * n_bins))
    return labels, pairs


def _slope_from_xy_vec(y):
    x = np.arange(1, y.shape[0] + 1)
    xm, ym = x.mean(), y.mean(axis=0)
    return ((x[:, None] - xm) * (y - ym)).sum(axis=0) / ((x - xm) ** 2).sum()


def _calc_diag_slope_mean(rdm_obj, n_bins=N_VALUE_BINS):
    _, pairs = _diag_index_pairs(n_bins)
    mats = rdm_obj.get_matrices()
    diags = np.stack([mats[:, i1, i2] for i1, i2 in pairs], axis=0)
    return _slope_from_xy_vec(diags.mean(axis=0))


def searchlight_diag_slope_maps(beta_maps_per_subj, brain_mask_img, radius=3,
                                rdm_method="euclidean", skip=SKIP):
    """Whole-brain map of across-action RDM diagonal slope per voxel."""
    from rsatoolbox.util.searchlight import get_searchlight_RDMs, get_volume_searchlight

    mask_data = brain_mask_img.get_fdata()
    centers, neighbors = get_volume_searchlight(mask_data, radius=radius, threshold=0.5)
    x, y, z = mask_data.shape
    rows = []
    for cond in beta_maps_per_subj:
        for subj, beta_list in beta_maps_per_subj[cond].items():
            if (cond, subj) in skip:
                continue
            sl_rdms = get_searchlight_RDMs(
                _prep_beta_stack(beta_list), centers, neighbors,
                np.arange(len(beta_list)), method=rdm_method, verbose=False)
            sl_rdms.dissimilarities = np.nan_to_num(sl_rdms.dissimilarities)
            brain = np.zeros(x * y * z)
            brain[list(sl_rdms.rdm_descriptors["voxel_index"])] = _calc_diag_slope_mean(sl_rdms)
            rows.append({"subj": subj, "condition": cond,
                         "sl_map": math_img("a", a=_reshape_to_img(brain, brain_mask_img))})
    return pd.DataFrame(rows)


def whole_brain_exploration_correlation(sl_df, exploration_slopes, cortex_masker, skip=SKIP):
    """Correlate exploration slope with searchlight diagonal slope per voxel."""
    from nilearn.image import new_img_like
    from scipy.stats import pearsonr

    slope_rows = []
    exp_rows = []
    for cond in ("rew", "pun"):
        for subj in sl_df[(sl_df.condition == cond)].subj.unique():
            if (cond, subj) in skip:
                continue
            slope_rows.append(cortex_masker.transform(
                sl_df[(sl_df.subj == subj) & (sl_df.condition == cond)].sl_map.values[0]).squeeze())
            exp_rows.append(exploration_slopes[cond][subj])
    all_slopes = np.array(slope_rows)
    exploration_arr = np.array(exp_rows)

    correlations, pvals = [], []
    for i in range(all_slopes.shape[1]):
        r, p = pearsonr(all_slopes[:, i], exploration_arr)
        correlations.append(r)
        pvals.append(p)
    correlations, pvals = np.array(correlations), np.array(pvals)

    mask_bool = cortex_masker.mask_img_.get_fdata().astype(bool)
    corr_map = np.zeros(cortex_masker.mask_img_.shape)
    corr_map[mask_bool] = correlations
    pval_map = np.zeros(cortex_masker.mask_img_.shape)
    pval_map[mask_bool] = pvals
    return (new_img_like(cortex_masker.mask_img_, corr_map),
            new_img_like(cortex_masker.mask_img_, pval_map))


# ========================================================================== #
# Runner
# ========================================================================== #
def main(force: bool = False):
    run_fmri_pipeline(force=force, through="searchlight")


if __name__ == "__main__":
    main()

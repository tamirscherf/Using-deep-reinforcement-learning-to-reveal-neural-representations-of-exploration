"""Load saved analysis outputs or compute them if missing."""

from __future__ import annotations

from typing import Any

import paths


def _load_behavior_bundle() -> dict[str, Any]:
    import task_behavior as tb

    return {
        "all_trials_df": paths.load_pickle(paths.ALL_TRIALS_DF),
        "episode_df": paths.load_pickle(paths.EPISODE_SUMMARIES),
        "likelihood_df": paths.load_pickle(paths.LIKELIHOOD_DF),
        "model_outputs": paths.load_pickle(paths.MODEL_OUTPUTS),
        "df_bins": paths.load_pickle(paths.DF_BINS),
        "behavior_gain": tb.load_behavior()[0],
        "behavior_loss": tb.load_behavior()[1],
        "performance": tb.load_overall_performance(),
    }


def get_behavior_tables(force: bool = False) -> dict[str, Any]:
    """Trial- and episode-level behavior tables."""
    if not force and paths.ALL_TRIALS_DF.exists():
        return _load_behavior_bundle()
    import task_behavior as tb

    return tb.compute_behavior_tables()


def get_all_trials_df(force: bool = False):
    return get_behavior_tables(force=force)["all_trials_df"]


def get_episode_summaries(force: bool = False):
    return get_behavior_tables(force=force)["episode_df"]


def get_model_outputs(force: bool = False):
    return get_behavior_tables(force=force)["model_outputs"]


def get_lstm_activations(force: bool = False) -> dict[str, Any]:
    """Trimmed LSTM activations and unit-selection results."""
    if not force and paths.LSTM_ACTIVATIONS.exists():
        return paths.load_pickle(paths.LSTM_ACTIVATIONS)
    import lstm_representations as lr

    return lr.compute_lstm_activations()


def get_neural_rdms(force: bool = False, n_bins: int = paths.N_VALUE_BINS,
                    distance: str = paths.NEURAL_RDM_DISTANCE):
    """Per-ROI neural RDMs."""
    path = paths.neural_rdms_path(n_bins, distance)
    if not force and path.exists():
        return paths.load_pickle(path)
    import fmri_pipeline as fp

    fp.run_fmri_pipeline(force=force, through="rdms")
    return paths.load_pickle(path)


def get_functional_maskers(force: bool = False):
    if not force and paths.FUNCTIONAL_MASKERS.exists():
        return paths.load_pickle(paths.FUNCTIONAL_MASKERS)
    import fmri_pipeline as fp

    fp.run_fmri_pipeline(force=force, through="masks")
    return paths.load_pickle(paths.FUNCTIONAL_MASKERS)


def get_second_level_fb_modulation(force: bool = False):
    if not force and paths.SECOND_LEVEL_FB_MODULATION.exists():
        return paths.load_pickle(paths.SECOND_LEVEL_FB_MODULATION)
    import fmri_pipeline as fp

    fp.run_fmri_pipeline(force=force, through="masks")
    return paths.load_pickle(paths.SECOND_LEVEL_FB_MODULATION)


def get_searchlight_bundle(force: bool = False) -> dict[str, Any]:
    """Searchlight maps and whole-brain exploration correlation."""
    needed = (
        paths.SEARCHLIGHT_MODEL_MAPS,
        paths.SEARCHLIGHT_SECOND_LEVEL,
        paths.SEARCHLIGHT_DIAG_SLOPE_MAPS,
        paths.EXPLORATION_CORRELATION_IMG,
    )
    if not force and all(p.exists() for p in needed):
        return _load_searchlight_bundle()
    import fmri_pipeline as fp

    fp.run_fmri_pipeline(force=force, through="searchlight")
    return _load_searchlight_bundle()


def _load_searchlight_bundle() -> dict[str, Any]:
    sl_maps = paths.load_pickle(paths.SEARCHLIGHT_MODEL_MAPS)
    second_level = paths.load_pickle(paths.SEARCHLIGHT_SECOND_LEVEL)
    corr_bundle = paths.load_pickle(paths.EXPLORATION_CORRELATION_IMG)
    correlation_img = corr_bundle["correlation_img"]
    correlation_pval = corr_bundle.get("correlation_pval")
    searchlight_model_map = _extract_searchlight_model_map(second_level)
    return {
        "searchlight_model_maps": sl_maps,
        "second_level_contrasts": second_level,
        "searchlight_diag_slope_maps": paths.load_pickle(paths.SEARCHLIGHT_DIAG_SLOPE_MAPS),
        "correlation_img": correlation_img,
        "correlation_pval": correlation_pval,
        "searchlight_model_map": searchlight_model_map,
    }


def _extract_searchlight_model_map(second_level_df):
    """Group-level deep-RL searchlight map used in conjunction plots."""
    if second_level_df is None or second_level_df.empty:
        return None
    rows = second_level_df[
        (second_level_df.model == "lstm_subj") & (second_level_df.condition == "rew")]
    if rows.empty:
        rows = second_level_df[second_level_df.model == "lstm_subj"]
    return rows.iloc[0]["map"] if not rows.empty else None


def get_multi_model_df(force: bool = False):
    """Neural-vs-model RSA correlation table."""
    if not force and paths.MULTI_MODEL_DF.exists():
        return paths.load_pickle(paths.MULTI_MODEL_DF)
    import rsa_brain as rb

    rb.compute_rsa_tables()
    return paths.load_pickle(paths.MULTI_MODEL_DF)


def get_rsa_ttest_df(force: bool = False):
    if not force and paths.RSA_PAIRED_TTEST_DF.exists():
        return paths.load_pickle(paths.RSA_PAIRED_TTEST_DF)
    import rsa_brain as rb

    rb.compute_rsa_tables()
    return paths.load_pickle(paths.RSA_PAIRED_TTEST_DF)


def get_sub_rdm_df(force: bool = False):
    """Within/across sub-RDM correlation table."""
    if not force and paths.SUB_RDM_DF.exists():
        return paths.load_pickle(paths.SUB_RDM_DF)
    import exploration_neural_dynamics as end

    end.compute_sub_rdm_df()
    return paths.load_pickle(paths.SUB_RDM_DF)

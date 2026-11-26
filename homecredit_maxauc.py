#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Home Credit Default Risk — application-only AUC pipeline.

STRICT INPUT POLICY:
  - Reads train.csv and test.csv only.
  - Does not read sample_submission.csv or any auxiliary/history table.
  - All engineered features are derived from columns in train.csv/test.csv.

Example:
  python homecredit_maxauc.py --data input --out results/hc_run
  python homecredit_maxauc.py --data input --out results/hc_run --models lgbm,catboost,xgboost --seeds 42,2024
  python homecredit_maxauc.py --data input --out results/hc_run --quick --models lgbm

The pipeline combines application-level domain features, leakage-aware OOF target encoding,
shared StratifiedKFold splits, fold-level prediction caches, and log-odds blending.
"""
from __future__ import annotations

import argparse
import atexit
import gc
import hashlib
import importlib.util
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit
from scipy.special import expit, logit
from scipy.special import ndtri
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier

import lightgbm as lgb
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

CODE_VERSION = "HC_APPLICATION_ONLY_MAXAUC_V4_7_META_APP_FEATURES_2026-10-23"
ID = "SK_ID_CURR"
TARGET = "TARGET"
INPUT_FILES = ["train.csv", "test.csv"]


def parse_args():
    p = argparse.ArgumentParser(description="Train/test-only Home Credit AUC pipeline")
    p.add_argument("--data", default="input", help="folder containing competition CSV files")
    p.add_argument("--out", default="results/hc_run", help="output folder for this run")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seeds", default="42,2024", help="comma-separated CV seeds; use 42 for a faster first run")
    p.add_argument("--models", default="lgbm,catboost", help="comma-separated: lgbm,lgbm_et,catboost,xgboost,xgb_rank,extratrees")
    p.add_argument("--lgbm-profile", choices=["current", "autolab"], default="current",
                   help="use current LGBM settings or the saved AutoLab champion settings")
    p.add_argument("--quick", action="store_true", help="fewer iterations, intended only for smoke tests")
    p.add_argument("--no-feature-cache", action="store_true", help="rebuild train/test-only features instead of loading feature cache")
    p.add_argument("--no-fold-cache", action="store_true", help="ignore cached fold predictions")
    p.add_argument("--catboost-task-type", choices=["CPU", "GPU"], default="CPU")
    p.add_argument("--catboost-native-keys", action="store_true",
                   help="let CatBoost learn ordered CTRs from tariff/key categoricals instead of manual TE")
    p.add_argument("--inner-te-folds", type=int, default=5)
    p.add_argument("--te-alpha", type=float, default=40.0)
    p.add_argument("--threads", type=int, default=-1, help="threads for model training; -1 means all available")
    p.add_argument("--lgbm-estimators", type=int, default=10000)
    p.add_argument("--lgbm-es-rounds", type=int, default=None,
                   help="LGBM early-stopping patience (defaults to 250, or 80 with --quick)")
    p.add_argument("--optuna-trials", type=int, default=0,
                   help="run an Optuna search for AutoLab LGBM only; use --folds 3 and --seeds 42 for screening")
    p.add_argument("--lgbm-params-json", default=None,
                   help="JSON file of fixed LGBM parameter overrides, e.g. Optuna best_params.json")
    p.add_argument("--catboost-params-json", default=None, help="JSON CatBoost parameter overrides")
    p.add_argument("--xgb-params-json", default=None, help="JSON XGBoost parameter overrides")
    p.add_argument("--refit-outer", action="store_true",
                   help="select tree count on inner ES, then refit on all outer-training rows")
    p.add_argument("--refit-iteration-scale", type=float, default=1.0,
                   help="multiply selected tree count before the outer-training refit")
    p.add_argument("--lgbm-monotone", choices=["auto", "off", "ext"], default="auto")
    p.add_argument("--catboost-iterations", type=int, default=5000)
    p.add_argument("--xgb-estimators", type=int, default=7000)
    p.add_argument("--submission-name", default="submission.csv")
    p.add_argument("--no-uid", action="store_true", help="drop same-person (UID) features and UID label features")
    p.add_argument("--no-ext-impute", action="store_true", help="skip OOF imputation of EXT_SOURCE_* (saves several minutes)")
    p.add_argument("--no-product-x", action="store_true", help="skip exact-amount / goods==credit target-encoding keys")
    p.add_argument("--autolab-age-end", action="store_true", help="add AutoLab loan-end-age features")
    p.add_argument("--autolab-days-diffs", action="store_true", help="add missing AutoLab date-difference features")
    p.add_argument("--client-dates", action="store_true", help="add application-only ages and intervals between client date fields")
    p.add_argument("--client-id-renewal", action="store_true",
                   help="add ID renewal delay, punctuality, and age-normalized delay features")
    p.add_argument("--id-features", action="store_true",
                   help="add train/test-only SK_ID_CURR trend, fixed-width ID blocks, and fold-safe block target encodings")
    p.add_argument("--id-te-only", action="store_true",
                   help="add only fold-safe target encodings of fixed-width SK_ID_CURR blocks (no raw/rank/block ID features)")
    p.add_argument("--geometry", choices=["off", "umap", "umap_klein", "umap_supervised"], default="off",
                   help="fold-local UMAP embedding; umap_supervised uses only that fold's model-fit labels")
    p.add_argument("--umap-fit-sample", type=int, default=30000,
                   help="maximum model-fit rows used to fit UMAP (default: 30000); all rows are transformed")
    p.add_argument("--umap-components", type=int, default=2,
                   help="number of UMAP coordinates to add (Klein mapping requires exactly 2)")
    p.add_argument("--umap-neighbors", type=int, default=30)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--umap-metric", choices=["cosine", "euclidean"], default="cosine")
    p.add_argument("--umap-target-weight", type=float, default=0.5,
                   help="supervised UMAP balance between feature topology and target topology (0..1)")
    p.add_argument("--knn-features", action="store_true",
                   help="add fold-safe local-neighbor distances and smoothed neighbor-default rates")
    p.add_argument("--knn-fit-sample", type=int, default=25000,
                   help="maximum reference rows per fold-local approximate kNN index (default: 25000)")
    p.add_argument("--knn-components", type=int, default=32,
                   help="PCA dimensions used for the local-neighbor index (default: 32)")
    p.add_argument("--knn-inner-folds", type=int, default=3,
                   help="cross-fit folds for training-row neighbor features (default: 3)")
    p.add_argument("--autolab-rank-extra", action="store_true", help="add the AutoLab DAYS_EMPLOYED rank feature")
    p.add_argument("--autolab-tariff", action="store_true", help="add AutoLab tariff-size counts and cross-fitted tariff TE keys")
    p.add_argument("--autolab-value-counts", action="store_true", help="add AutoLab exact-value counts for repeated numeric application values")
    p.add_argument("--autolab-uid", action="store_true", help="add the three AutoLab invariant-UID groups and fold-safe label features")
    p.add_argument("--ext-copula", action="store_true", help="add train/test-only empirical copula and joint-tail features for EXT_SOURCE_1/2/3")
    p.add_argument("--ratio-income-stress", action="store_true",
                   help="add the example_code annuity x household-size / income ratio")
    p.add_argument("--peer-context", action="store_true",
                   help="add label-free peer-relative amount, affordability and external-score features")
    p.add_argument("--finance-extra-keys", action="store_true",
                   help="add payment/goods and contextual tariff keys for fold-safe TE or CatBoost CTR")
    p.add_argument("--amortization-scenarios", action="store_true",
                   help="add fixed-rate hypothetical amortization terms and balance paths from input credit/annuity")
    p.add_argument("--financial-gap", action="store_true", help="add explicit financial-gap and hypothetical payment-stress features from application inputs")
    p.add_argument('--domain-russian-age',choices=['off','id','pension','both'],default='off',help='post-cache input-only legal-age hypothesis features')
    p.add_argument('--domain-behavior',choices=['off','retirement','inquiry','product','all'],default='off',help='post-cache input-only product/inquiry/retirement interactions')
    p.add_argument("--financial-income-period", choices=["annual", "monthly"], default="annual", help="income-period hypothesis for --financial-gap; annuity assumed monthly")
    p.add_argument("--financial-gap-group", choices=["all", "funding", "surplus", "shock", "thresholds", "ext", "core"], default="all", help="isolate a financial feature family without changing legacy all-feature caches")
    p.add_argument("--equation-disposable-income", action="store_true",
                   help="add input-only disposable-income equations: surplus per household member and credit months of surplus")
    p.add_argument("--equation-disposable-component",
                   choices=["surplus", "per_member", "credit_months", "log_credit_months"],
                   default=None,
                   help="ablate one disposable-income equation at a time; implies that equation is enabled")
    p.add_argument("--meta-application-features", action="store_true",
                   help="add reproducible application-only ratios/interactions inspired by meta.json; history-table features are excluded")
    p.add_argument("--meta-only-group", choices=["finance", "dates", "ext", "bureau", "missing"], default=None,
                   help="with --meta-application-features, add only this meta feature family (for controlled ablation)")
    p.add_argument("--es-holdout", type=float, default=0.0,
                   help="fraction of each training fold used for early stopping (0 = stop on the validation fold, as before). "
                        "Use 0.1 for an unbiased OOF estimate")
    p.add_argument("--pseudo-label", action="store_true",
                   help="LGBM-only transductive pseudo-labeling: label confident test rows from a fold-local model, then refit")
    p.add_argument("--pseudo-label-threshold", type=float, default=0.99,
                   help="confidence cutoff for pseudo-labeling (default: use p<=0.01 or p>=0.99)")
    p.add_argument("--pseudo-label-weight", type=float, default=0.10,
                   help="sample weight for pseudo-labeled test rows (default: 0.10)")
    p.add_argument("--group-cv", action="store_true", help="StratifiedGroupKFold by detected person (pessimistic for UID features)")
    p.add_argument('--feature-subspace-json',default=None,help='Explicit base-feature/key column manifest; fold-safe UID label features remain a shared core')
    p.add_argument('--application-cache',default=None,help='Existing input-derived application cache with matching fingerprint/flags; read-only reuse')
    p.add_argument("--leak-check", action="store_true", help="train on permuted TARGET and report AUC (should be ~0.5), then exit")
    parsed = p.parse_args()
    if parsed.application_cache and parsed.no_feature_cache:
        p.error('--application-cache cannot be combined with --no-feature-cache')
    if parsed.id_features and parsed.id_te_only:
        p.error("--id-features and --id-te-only are mutually exclusive")
    if parsed.meta_only_group and not parsed.meta_application_features:
        p.error("--meta-only-group requires --meta-application-features")
    if not 0.5 < parsed.pseudo_label_threshold < 1.0:
        p.error("--pseudo-label-threshold must be between 0.5 and 1.0")
    if parsed.pseudo_label_weight <= 0:
        p.error("--pseudo-label-weight must be positive")
    if parsed.pseudo_label and not all(m.strip() == "lgbm" for m in parsed.models.split(",")):
        p.error("--pseudo-label currently supports only --models lgbm")
    if parsed.refit_outer and (parsed.es_holdout <= 0 or parsed.pseudo_label or
                               parsed.geometry != "off" or parsed.knn_features):
        p.error("--refit-outer requires inner ES and is incompatible with pseudo-label/geometry/kNN")
    if not 0.25 <= parsed.refit_iteration_scale <= 3:
        p.error("--refit-iteration-scale must be in [0.25, 3]")
    return parsed


args = parse_args()
SUBSPACE_CONFIG = json.loads(Path(args.feature_subspace_json).read_text(encoding='utf-8')) if args.feature_subspace_json else None
DOMAIN_CONFIG={}
if args.domain_russian_age!='off':
    from scripts.russian_law_features import VERSION as RU_DOMAIN_VERSION
    DOMAIN_CONFIG['russian_age']=[args.domain_russian_age,RU_DOMAIN_VERSION]
if args.domain_behavior!='off':
    from scripts.application_behavior_features import VERSION as BEHAVIOR_DOMAIN_VERSION
    DOMAIN_CONFIG['behavior']=[args.domain_behavior,BEHAVIOR_DOMAIN_VERSION]
DATA_DIR = Path(args.data)
OUT_DIR = Path(args.out)
if not OUT_DIR.is_absolute() and (not OUT_DIR.parts or OUT_DIR.parts[0].lower() != "results"):
    OUT_DIR = Path("results") / OUT_DIR
CACHE_DIR = OUT_DIR / "cache"
FOLD_CACHE_DIR = CACHE_DIR / "folds"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FOLD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
T0 = time.time()
FIXED_LGBM_OVERRIDES: dict = {}
FIXED_CB_OVERRIDES: dict = {}
FIXED_XGB_OVERRIDES: dict = {}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def acquire_research_gpu_slot(model_names: list[str]) -> None:
    """Serialize GPU jobs from this bounded research window on the 4-GB GPU."""
    if "research_5h_20261025" not in OUT_DIR.parts or os.name != "nt":
        return
    wants_gpu = ("catboost" in model_names and
                 str(FIXED_CB_OVERRIDES.get("task_type", args.catboost_task_type)).upper() == "GPU")
    wants_gpu |= bool(set(model_names) & {"xgboost", "xgb_rank"}) and str(
        FIXED_XGB_OVERRIDES.get("device", "cpu")).startswith("cuda")
    if not wants_gpu:
        return
    import msvcrt
    slot_path = OUT_DIR.parent.parent / "gpu.slot"
    slot_path.parent.mkdir(parents=True, exist_ok=True)
    slot = slot_path.open("a+b")
    if slot.seek(0, os.SEEK_END) == 0:
        slot.write(b"0")
        slot.flush()
    waiting = False
    while True:
        try:
            slot.seek(0)
            msvcrt.locking(slot.fileno(), msvcrt.LK_NBLCK, 1)
            break
        except OSError:
            if not waiting:
                log("Waiting for the bounded research GPU slot")
                waiting = True
            time.sleep(5)
    atexit.register(slot.close)
    log("Acquired bounded research GPU slot")


def sha10(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]


def locate_file(*names: str) -> Path | None:
    if not DATA_DIR.exists():
        return None
    lookup = {p.name.lower(): p for p in DATA_DIR.iterdir() if p.is_file()}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def fingerprint_inputs() -> str:
    rows = []
    for name in INPUT_FILES:
        p = locate_file(name)
        if p is None:
            rows.append((name, "MISSING"))
        else:
            st = p.stat()
            rows.append((name, st.st_size, st.st_mtime_ns))
    return sha10({"version": CODE_VERSION, "files": rows})

DATA_TAG = fingerprint_inputs()
_feature_flag_config = {"uid": not args.no_uid, "ext": not args.no_ext_impute, "px": not args.no_product_x,
                        "auto_age_end": args.autolab_age_end, "auto_days": args.autolab_days_diffs,
                        "client_dates": args.client_dates,
                        "auto_rank": args.autolab_rank_extra, "auto_tariff": args.autolab_tariff,
                        "auto_vc": args.autolab_value_counts, "auto_uid": args.autolab_uid,
                        "lgbm_profile": args.lgbm_profile}
# Keep existing feature-cache keys stable; this option only changes the cache key when enabled.
if args.ratio_income_stress:
    _feature_flag_config["ratio_income_stress"] = True
if args.peer_context:
    _feature_flag_config["peer_context"] = True
if args.finance_extra_keys:
    _feature_flag_config["finance_extra_keys"] = True
if args.amortization_scenarios:
    from scripts.finance_scenarios import VERSION as SCENARIO_VERSION
    _feature_flag_config["amortization_scenarios"] = SCENARIO_VERSION
if args.financial_gap:
    from scripts.financial_gap_features import VERSION as GAP_VERSION
    _feature_flag_config["financial_gap"] = [GAP_VERSION, args.financial_income_period]
    if args.financial_gap_group != "all":
        _feature_flag_config["financial_gap_group"] = args.financial_gap_group
if args.equation_disposable_income or args.equation_disposable_component:
    _feature_flag_config["equation_disposable_income"] = (
        args.equation_disposable_component or "all")
if args.client_id_renewal:
    _feature_flag_config["client_id_renewal"] = True
if args.ext_copula:
    _feature_flag_config["ext_copula"] = True
if args.meta_application_features:
    _feature_flag_config["meta_application_features"] = args.meta_only_group or "all"
# Keep legacy no-ID cache keys stable; ID experiments receive their own feature/model cache.
if args.id_features or args.id_te_only:
    _feature_flag_config["id_feature_mode"] = "full" if args.id_features else "te_only"
FLAG_TAG = sha10(_feature_flag_config)


def safe_div(a, b):
    aa = pd.to_numeric(a, errors="coerce").astype("float64")
    bb = pd.to_numeric(b, errors="coerce").astype("float64").replace(0, np.nan)
    out = aa / bb
    return out.replace([np.inf, -np.inf], np.nan)


def finite_float32(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return x.astype("float32")


def find_distribution_peaks(values: np.ndarray, max_peaks: int = 8) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    values = values[np.isfinite(values) & (values > 0) & (values < 120)]
    if len(values) < 100:
        return np.array([6., 12., 18., 24., 36.], dtype="float64")
    q_lo, q_hi = np.quantile(values, [0.01, 0.99])
    clipped = values[(values >= q_lo) & (values <= q_hi)]
    if len(clipped) < 50 or not np.isfinite(q_lo + q_hi) or q_hi <= q_lo:
        return np.unique(np.quantile(values, [0.1, 0.3, 0.5, 0.7, 0.9]))
    hist, edges = np.histogram(clipped, bins=160, range=(float(q_lo), float(q_hi)))
    peak_ix, _ = find_peaks(hist, prominence=max(1.0, hist.max() * 0.025), distance=3)
    if len(peak_ix) == 0:
        return np.unique(np.quantile(clipped, [0.1, 0.3, 0.5, 0.7, 0.9]))
    selected = sorted(peak_ix[np.argsort(hist[peak_ix])[-max_peaks:]])
    return np.unique((edges[:-1][selected] + edges[1:][selected]) / 2)


UID_LABEL_COLS = [
    "HC_UIDY_N_LAB", "HC_UIDY_POS", "HC_UIDY_ANYPOS", "HC_UIDY_RATE",
    "HC_UIDY_PREV_Y", "HC_UIDY_PREV_GAP", "HC_UIDY_NEXT_Y", "HC_UIDY_NEXT_GAP",
]


def uid_structure(raw: pd.DataFrame):
    """Same-person detection from invariants of the DAYS_* columns.

    Every DAYS_* value is measured backwards from the application date, so for one person who applies
    twice, (DAYS_BIRTH - DAYS_ID_PUBLISH) and (DAYS_BIRTH - DAYS_REGISTRATION) stay constant while each
    raw column shifts by the same number of days. Raw values only match same-day duplicates; these
    differences also match repeat applications on different dates. Uses no TARGET.

    Returns (feature dict of float arrays in raw order, integer group id per row, time per row).
    Time = -DAYS_BIRTH (age in days): larger means a later application.
    """
    n = len(raw)
    need = {"DAYS_BIRTH", "DAYS_ID_PUBLISH", "DAYS_REGISTRATION"}
    if not need.issubset(raw.columns):
        return {}, np.arange(n, dtype="int64"), np.zeros(n)

    def num(c):
        return pd.to_numeric(raw[c], errors="coerce").to_numpy(dtype="float64")

    birth = num("DAYS_BIRTH")
    b_id = np.round(birth - num("DAYS_ID_PUBLISH"))
    b_reg = np.round(birth - num("DAYS_REGISTRATION"))
    if "CODE_GENDER" in raw.columns:
        gender = raw["CODE_GENDER"].astype("string").fillna("?").astype(str).to_numpy()
    else:
        gender = np.full(n, "?")
    valid = np.isfinite(b_id) & np.isfinite(b_reg)
    gid = np.arange(n, dtype="int64") + n  # unique id for rows that cannot be matched
    if valid.any():
        kdf = pd.DataFrame({"g": gender[valid], "a": b_id[valid], "b": b_reg[valid]})
        gid[valid] = kdf.groupby(["g", "a", "b"], sort=False).ngroup().to_numpy()
    _, gid = np.unique(gid, return_inverse=True)
    gid = np.asarray(gid, dtype="int64").reshape(-1)
    time_ = np.where(np.isfinite(birth), -birth, 0.0)

    credit = num("AMT_CREDIT") if "AMT_CREDIT" in raw.columns else np.full(n, np.nan)
    annuity = num("AMT_ANNUITY") if "AMT_ANNUITY" in raw.columns else np.full(n, np.nan)
    d = pd.DataFrame({"g": gid, "t": time_, "c": credit, "a": annuity, "i": np.arange(n)})
    d = d.sort_values(["g", "t", "i"], kind="mergesort").reset_index(drop=True)
    gb = d.groupby("g", sort=False)
    size = gb["i"].transform("size").to_numpy(dtype="float64")
    rank = gb.cumcount().to_numpy(dtype="float64")
    cols = {"HC_UID_N": size, "HC_UID_N_EARLIER": rank, "HC_UID_N_LATER": size - 1 - rank}
    a_safe = d["a"].replace(0, np.nan)
    term = d["c"] / a_safe
    for side, sh in (("PREV", 1), ("NEXT", -1)):
        t_o, c_o, a_o = gb["t"].shift(sh), gb["c"].shift(sh), gb["a"].shift(sh)
        a_o_safe = a_o.replace(0, np.nan)
        cols[f"HC_UID_DAYS_{side}"] = (d["t"] - t_o).abs()
        cols[f"HC_UID_CREDIT_RATIO_{side}"] = d["c"] / c_o.replace(0, np.nan)
        cols[f"HC_UID_ANNUITY_RATIO_{side}"] = d["a"] / a_o_safe
        cols[f"HC_UID_TERM_DIFF_{side}"] = term - c_o / a_o_safe
    order = d["i"].to_numpy()
    out = {}
    for k, v in cols.items():
        res = np.empty(n, dtype="float64")
        res[order] = np.asarray(v, dtype="float64")
        out[k] = res
    return out, gid, time_


def autolab_uid_groups(raw: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Recreate AutoLab's three invariant UID keys using application rows only.

    These are deliberately kept separate from uid_structure(): the latter includes gender
    in one stricter identity proxy, while these keys mirror the older AutoLab candidates.
    Invalid keys are assigned unique row IDs so missing values never merge applicants.
    """
    n = len(raw)
    required = {"DAYS_BIRTH", "DAYS_ID_PUBLISH", "DAYS_REGISTRATION", "DAYS_EMPLOYED", "AMT_INCOME_TOTAL"}
    if not required.issubset(raw.columns):
        return {}, {}

    def vals(col: str) -> np.ndarray:
        return pd.to_numeric(raw[col], errors="coerce").to_numpy(dtype="float64")

    birth = vals("DAYS_BIRTH")
    b_id = birth - vals("DAYS_ID_PUBLISH")
    b_reg = birth - vals("DAYS_REGISTRATION")
    emp_b = vals("DAYS_EMPLOYED") - birth
    income = vals("AMT_INCOME_TOTAL")
    specs = {
        "HC_AUID1": (b_id, b_reg),
        "HC_AUID2": (b_id, b_reg, income),
        "HC_AUID3": (b_id, b_reg, emp_b),
    }
    groups: dict[str, np.ndarray] = {}
    size_features: dict[str, np.ndarray] = {}
    for name, arrays in specs.items():
        frame = pd.DataFrame({f"k{i}": a for i, a in enumerate(arrays)})
        valid = np.isfinite(frame.to_numpy(dtype="float64")).all(axis=1)
        gid = np.arange(n, dtype="int64") + n
        sizes = np.ones(n, dtype="float32")
        if valid.any():
            valid_frame = frame.loc[valid]
            grouped = valid_frame.groupby(list(valid_frame.columns), sort=False, dropna=False)
            local_gid = grouped.ngroup().to_numpy(dtype="int64")
            gid[valid] = local_gid
            sizes[valid] = grouped[valid_frame.columns[0]].transform("size").to_numpy(dtype="float32")
        _, gid = np.unique(gid, return_inverse=True)
        groups[name] = gid.astype("int64", copy=False)
        size_features[f"{name}_SIZE"] = sizes
    return groups, size_features


def uid_label_features(gid: np.ndarray, time_: np.ndarray, lab: np.ndarray, prior: float, alpha: float = 2.0):
    """Outcomes of OTHER applications by the same person, using only labels present in `lab`.

    `lab` holds TARGET for the rows of the current outer-training fold and NaN everywhere else
    (validation + test rows). A row never sees its own label: counts exclude self, prev/next use
    strictly earlier/later rows. Train, validation and test rows therefore see partner labels drawn
    from the same pool (the outer-training fold), so the feature has the same meaning in all three.
    """
    n = len(gid)
    counts = np.bincount(gid)
    multi = counts[gid] >= 2
    out = {name: np.full(n, np.nan, dtype="float32") for name in UID_LABEL_COLS}
    for name in ("HC_UIDY_N_LAB", "HC_UIDY_POS", "HC_UIDY_ANYPOS"):
        out[name][:] = 0.0
    if not multi.any():
        return out
    idx = np.flatnonzero(multi)
    s = pd.DataFrame({"g": gid[idx], "t": time_[idx], "y": lab[idx], "i": idx})
    s = s.sort_values(["g", "t", "i"], kind="mergesort").reset_index(drop=True)
    g = s["g"]
    has = s["y"].notna().astype("float64")
    yv = s["y"].fillna(0.0)
    cnt = has.groupby(g).transform("sum") - has
    pos = yv.groupby(g).transform("sum") - yv
    rate = ((pos + alpha * prior) / (cnt + alpha)).where(cnt > 0)
    gy = s["y"].groupby(g)
    prev_y = gy.shift(1).groupby(g).ffill()
    next_y = gy.shift(-1).groupby(g).bfill()
    tl = s["t"].where(has > 0)
    gt = tl.groupby(g)
    prev_t = gt.shift(1).groupby(g).ffill()
    next_t = gt.shift(-1).groupby(g).bfill()
    vals = {
        "HC_UIDY_N_LAB": cnt, "HC_UIDY_POS": pos, "HC_UIDY_ANYPOS": (pos > 0).astype("float64"),
        "HC_UIDY_RATE": rate, "HC_UIDY_PREV_Y": prev_y, "HC_UIDY_PREV_GAP": (s["t"] - prev_t).abs(),
        "HC_UIDY_NEXT_Y": next_y, "HC_UIDY_NEXT_GAP": (next_t - s["t"]).abs(),
    }
    order = s["i"].to_numpy()
    for k, v in vals.items():
        out[k][order] = np.asarray(v, dtype="float32")
    return out


def uid_coverage(gid: np.ndarray, n_train: int, folds) -> dict:
    """How often a row has at least one labeled partner: validation rows vs test rows.

    If the CV rows see far fewer (or more) partners than test rows do for a comparable labeled pool,
    the CV gain from the UID label features will not transfer to the leaderboard.
    """
    size = int(gid.max()) + 1
    cnt_full = np.bincount(gid[:n_train], minlength=size)
    out = {"test_with_labeled_partner_full_train": float((cnt_full[gid[n_train:]] > 0).mean())}
    val_cov, test_cov = [], []
    for tr, va in folds:
        cnt = np.bincount(gid[tr], minlength=size)
        val_cov.append(float((cnt[gid[va]] > 0).mean()))
        test_cov.append(float((cnt[gid[n_train:]] > 0).mean()))
    out["val_with_labeled_partner_in_fold_train"] = float(np.mean(val_cov))
    out["test_with_labeled_partner_in_fold_train"] = float(np.mean(test_cov))
    return out


def ext_impute_features(features: pd.DataFrame, ext_cols: list, seed: int = 2026, n_splits: int = 3) -> dict:
    """Unsupervised OOF imputation of EXT_SOURCE_* from the rest of the application (no TARGET).

    Known rows get out-of-fold predictions (so the model never predicts a row it trained on); missing rows
    get the mean of the fold models. Gives EXTP (expected value), EXTI (imputed value), EXTR (actual - expected).
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold

    P = features.select_dtypes(include=[np.number]).copy()
    P = P[[c for c in P.columns if "EXT" not in c.upper()]]
    for c in ["CODE_GENDER", "NAME_CONTRACT_TYPE", "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE",
              "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE"]:
        if c in features.columns:
            P[f"_CODE_{c}"] = features[c].astype("category").cat.codes.astype("float32")
    P = P.astype("float32")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = {}
    for c in ext_cols:
        yv = pd.to_numeric(features[c], errors="coerce").to_numpy(dtype="float64")
        known = np.isfinite(yv)
        pred = np.full(len(yv), np.nan)
        ik, iu = np.flatnonzero(known), np.flatnonzero(~known)
        if len(ik) < 1000:
            continue
        acc = np.zeros(len(iu))
        for a, b in kf.split(ik):
            m = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.08, max_leaf_nodes=31,
                                              min_samples_leaf=100, l2_regularization=1.0, random_state=seed)
            m.fit(P.iloc[ik[a]], yv[ik[a]])
            pred[ik[b]] = m.predict(P.iloc[ik[b]])
            if len(iu):
                acc += m.predict(P.iloc[iu]) / n_splits
        if len(iu):
            pred[iu] = acc
        short = c.replace("EXT_SOURCE_", "E")
        out[f"HC_EXTP_{short}"] = pred
        out[f"HC_EXTI_{short}"] = np.where(known, yv, pred)
        out[f"HC_EXTR_{short}"] = np.where(known, yv - pred, np.nan)
    return out


def application_features(app: pd.DataFrame, n_train: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = app.copy()
    id_values = pd.to_numeric(raw[ID], errors="coerce") if ID in raw else None
    if ID in raw:
        raw = raw.drop(columns=[ID])
    raw = raw.drop(columns=[TARGET], errors="ignore")

    # Preserve the employment anomaly as a separate signal.
    if "DAYS_EMPLOYED" in raw.columns:
        raw["DAYS_EMPLOYED_ANOM"] = (raw["DAYS_EMPLOYED"] == 365243).astype("int8")
        raw["DAYS_EMPLOYED"] = raw["DAYS_EMPLOYED"].replace(365243, np.nan)
    if "OWN_CAR_AGE" in raw.columns:
        raw["OWN_CAR_AGE_ANOM"] = (raw["OWN_CAR_AGE"] >= 60).astype("int8")
        raw.loc[raw["OWN_CAR_AGE"] >= 60, "OWN_CAR_AGE"] = np.nan

    num = {}
    def put(name, series):
        num[name] = finite_float32(pd.Series(series, index=raw.index))
    put("HC_ROW_MISSING_COUNT", raw.isna().sum(axis=1))
    if args.financial_gap:
        from scripts.financial_gap_features import financial_gap_features
        for name, series in financial_gap_features(raw, args.financial_income_period, args.financial_gap_group).items():
            put(name, series)
    if args.amortization_scenarios:
        from scripts.finance_scenarios import scenario_features
        for name, series in scenario_features(raw).items():
            put(name, series)

    if args.meta_application_features:
        # Reconstruct only the meta.json ideas supported by the permitted
        # application table. Names are HC_META_* to distinguish these
        # explicitly defined proxies from the competitor's undocumented
        # feature formulas. No bureau/POS/installment/previous/CC table is read.
        income = pd.to_numeric(raw["AMT_INCOME_TOTAL"], errors="coerce") if "AMT_INCOME_TOTAL" in raw else None
        credit = pd.to_numeric(raw["AMT_CREDIT"], errors="coerce") if "AMT_CREDIT" in raw else None
        annuity = pd.to_numeric(raw["AMT_ANNUITY"], errors="coerce") if "AMT_ANNUITY" in raw else None
        age_years = -pd.to_numeric(raw["DAYS_BIRTH"], errors="coerce") / 365.25 if "DAYS_BIRTH" in raw else None
        employed_days = pd.to_numeric(raw["DAYS_EMPLOYED"], errors="coerce") if "DAYS_EMPLOYED" in raw else None
        if income is not None and credit is not None:
            put("HC_META_CREDIT_INCOME", safe_div(credit, income))
        if income is not None and annuity is not None:
            put("HC_META_ANNUITY_INCOME", safe_div(annuity, income))
        if credit is not None and annuity is not None:
            put("HC_META_CREDIT_ANNUITY", safe_div(credit, annuity))
        if "DAYS_EMPLOYED" in raw:
            anomaly = raw.get("DAYS_EMPLOYED_ANOM", pd.Series(0, index=raw.index))
            put("HC_META_FLAG_ANOM", anomaly)
        if age_years is not None and employed_days is not None:
            emp_years = -employed_days / 365.25
            put("HC_META_EMPLOYED_YEARS", emp_years)
            put("HC_META_DAYS_EMPLOYED_PERC", safe_div(employed_days, raw["DAYS_BIRTH"]))
            put("HC_META_EMPLOYED_BIRTH_RATIO", safe_div(employed_days, raw["DAYS_BIRTH"]))
            put("HC_META_EMPLOYED_X_AGE", emp_years * age_years)
        if income is not None and "CNT_FAM_MEMBERS" in raw:
            put("HC_META_INCOME_PER_PERSON", safe_div(income, raw["CNT_FAM_MEMBERS"]))
        ext_names = [c for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if c in raw]
        if ext_names:
            ext = raw[ext_names].apply(pd.to_numeric, errors="coerce")
            ext_mean = ext.mean(axis=1)
            ext_std = ext.std(axis=1)
            ext_min, ext_max = ext.min(axis=1), ext.max(axis=1)
            put("HC_META_EXT_MEAN", ext_mean)
            put("HC_META_EXT_STD", ext_std)
            put("HC_META_EXT_MIN", ext_min)
            put("HC_META_EXT_MAX", ext_max)
            put("HC_META_EXT_MEDIAN", ext.median(axis=1))
            put("HC_META_EXT_RANGE", ext_max - ext_min)
            put("HC_META_EXT_PRODUCT", ext.prod(axis=1, min_count=len(ext_names)))
            put("HC_META_EXT_KNOWN_COUNT", ext.notna().sum(axis=1))
            put("HC_META_EXT_ANY_MISSING", ext.isna().any(axis=1).astype("int8"))
            put("HC_META_EXT_ALL_MISSING", ext.isna().all(axis=1).astype("int8"))
            if age_years is not None:
                put("HC_META_EXT_RANGE_X_AGE", (ext_max - ext_min) * age_years)
                put("HC_META_EXT_MEAN_X_AGE", ext_mean * age_years)
            if credit is not None and income is not None:
                put("HC_META_EXT_MEAN_X_CREDIT_INCOME", ext_mean * safe_div(credit, income))
            if "REGION_RATING_CLIENT" in raw:
                put("HC_META_RATING_X_EXT_MEAN", pd.to_numeric(raw["REGION_RATING_CLIENT"], errors="coerce") * ext_mean)
            for c in ext_names:
                x = pd.to_numeric(raw[c], errors="coerce")
                put(f"HC_META_{c}_SQUARED", x * x)
                if age_years is not None:
                    put(f"HC_META_{c}_X_AGE", x * age_years)
                if credit is not None:
                    put(f"HC_META_{c}_X_CREDIT", x * credit)
            for i, c1 in enumerate(ext_names):
                for c2 in ext_names[i + 1:]:
                    x1, x2 = pd.to_numeric(raw[c1], errors="coerce"), pd.to_numeric(raw[c2], errors="coerce")
                    put(f"HC_META_{c1}_X_{c2}", x1 * x2)
                    put(f"HC_META_{c1}_OVER_{c2}", safe_div(x1, x2))
            ext_sum = ext.sum(axis=1, min_count=1)
            for c in ext_names:
                put(f"HC_META_{c}_SHARE", safe_div(raw[c], ext_sum))
        if {"AMT_GOODS_PRICE", "AMT_CREDIT"}.issubset(raw.columns):
            put("HC_META_GOODS_CREDIT_RATIO", safe_div(raw["AMT_GOODS_PRICE"], raw["AMT_CREDIT"]))
            put("HC_META_CREDIT_GOODS_DIFF", raw["AMT_CREDIT"] - raw["AMT_GOODS_PRICE"])
        if {"DAYS_REGISTRATION", "DAYS_BIRTH"}.issubset(raw.columns):
            put("HC_META_REGISTRATION_BIRTH_RATIO", safe_div(raw["DAYS_REGISTRATION"], raw["DAYS_BIRTH"]))
        if {"DAYS_ID_PUBLISH", "DAYS_BIRTH"}.issubset(raw.columns):
            put("HC_META_ID_PUBLISH_BIRTH_RATIO", safe_div(raw["DAYS_ID_PUBLISH"], raw["DAYS_BIRTH"]))
        bureau_req_cols = [c for c in raw.columns if c.startswith("AMT_REQ_CREDIT_BUREAU_")]
        if bureau_req_cols:
            req = raw[bureau_req_cols].apply(pd.to_numeric, errors="coerce")
            put("HC_META_BUREAU_REQ_MISSING_COUNT", req.isna().sum(axis=1))
            put("HC_META_BUREAU_REQ_SUM", req.sum(axis=1, min_count=1))
            put("HC_META_BUREAU_REQ_NONZERO", (req.fillna(0) != 0).sum(axis=1))
        put("HC_META_NA_COUNT", raw.isna().sum(axis=1))
        if args.meta_only_group:
            def belongs_to_meta_group(name, group):
                if not name.startswith("HC_META_"):
                    return True
                if group == "finance":
                    return name.startswith(("HC_META_CREDIT_", "HC_META_ANNUITY_", "HC_META_GOODS_")) or name == "HC_META_EXT_MEAN_X_CREDIT_INCOME"
                if group == "dates":
                    return name in {"HC_META_FLAG_ANOM", "HC_META_EMPLOYED_YEARS", "HC_META_DAYS_EMPLOYED_PERC",
                                    "HC_META_EMPLOYED_BIRTH_RATIO", "HC_META_EMPLOYED_X_AGE",
                                    "HC_META_REGISTRATION_BIRTH_RATIO", "HC_META_ID_PUBLISH_BIRTH_RATIO"}
                if group == "ext":
                    return name.startswith("HC_META_EXT_") or name.startswith("HC_META_RATING_X_EXT_")
                if group == "bureau":
                    return name.startswith("HC_META_BUREAU_REQ_")
                if group == "missing":
                    return name == "HC_META_NA_COUNT" or name in {"HC_META_EXT_ANY_MISSING", "HC_META_EXT_ALL_MISSING", "HC_META_EXT_KNOWN_COUNT"}
                return True
            num = {name: values for name, values in num.items()
                   if belongs_to_meta_group(name, args.meta_only_group)}

    if args.id_features and id_values is not None:
        # SK_ID_CURR is the only additional signal here: no auxiliary/history files are read.
        # Raw/rank features express broad order; fixed-width bins expose local ID neighborhoods.
        id_rank = id_values.rank(method="average", pct=True)
        put("HC_ID_RAW", id_values)
        put("HC_ID_RANK", id_rank)
        for width in (5_000, 20_000):
            put(f"HC_ID_BLOCK_{width}", np.floor(id_values / width))

    credit_term = safe_div(raw["AMT_CREDIT"], raw["AMT_ANNUITY"]) if {"AMT_CREDIT", "AMT_ANNUITY"}.issubset(raw.columns) else pd.Series(np.nan, index=raw.index)
    if {"AMT_CREDIT", "AMT_ANNUITY"}.issubset(raw.columns):
        put("HC_CREDIT_TERM", credit_term)
        put("HC_PAYMENT_RATE", safe_div(raw["AMT_ANNUITY"], raw["AMT_CREDIT"]))

        # Learn loan-term distribution peaks from application TRAIN features only; TARGET is never used.
        term_train = pd.to_numeric(credit_term.iloc[:n_train], errors="coerce").to_numpy(dtype="float64")
        peaks = find_distribution_peaks(term_train)
        term_values = pd.to_numeric(credit_term, errors="coerce").to_numpy(dtype="float64")
        valid = np.isfinite(term_values)
        nearest = np.full(len(term_values), np.nan, dtype="float64")
        dist = np.full(len(term_values), np.nan, dtype="float64")
        seg_id = np.full(len(term_values), np.nan, dtype="float64")
        if len(peaks):
            distances = np.abs(term_values[valid, None] - peaks[None, :])
            ix = distances.argmin(axis=1)
            nearest[valid], dist[valid], seg_id[valid] = peaks[ix], distances[np.arange(len(ix)), ix], ix
        put("HC_SEG_PEAK", nearest)
        put("HC_SEG_ID", seg_id)
        put("HC_SEG_DIST", term_values - nearest)
        put("HC_SEG_ABSDIST", dist)
        put("HC_SEG_RATIO_TO_PEAK", safe_div(pd.Series(term_values, index=raw.index), pd.Series(nearest, index=raw.index)))
        # Position between neighbouring peaks is a separate geometric coordinate.
        pos = np.full(len(term_values), np.nan, dtype="float64")
        if len(peaks) >= 2:
            interval = np.searchsorted(peaks, term_values, side="right") - 1
            use = valid & (interval >= 0) & (interval < len(peaks) - 1)
            ii = interval[use]
            pos[use] = (term_values[use] - peaks[ii]) / (peaks[ii + 1] - peaks[ii])
            pos[valid & (term_values < peaks[0])] = (term_values[valid & (term_values < peaks[0])] - peaks[0]) / max(peaks[1] - peaks[0], 1e-6)
            pos[valid & (term_values >= peaks[-1])] = 1.0 + (term_values[valid & (term_values >= peaks[-1])] - peaks[-1]) / max(peaks[-1] - peaks[-2], 1e-6)
        put("HC_SEG_POS", pos)
        # Contract-type-specific peak representation; peaks are learned from that type's TRAIN rows only.
        contract_col = "NAME_CONTRACT_TYPE"
        contract_near = np.full(len(term_values), np.nan, dtype="float64")
        contract_dist = np.full(len(term_values), np.nan, dtype="float64")
        contract_id = np.full(len(term_values), np.nan, dtype="float64")
        if contract_col in raw.columns:
            contracts = raw[contract_col].astype("string").fillna("__MISSING__")
            train_contracts = contracts.iloc[:n_train]
            for contract in train_contracts.unique():
                mask_all = (contracts == contract).to_numpy()
                mask_train = (train_contracts == contract).to_numpy()
                group_terms = pd.to_numeric(credit_term.iloc[:n_train], errors="coerce").to_numpy(dtype="float64")[mask_train]
                if np.isfinite(group_terms).sum() >= 300:
                    group_peaks = find_distribution_peaks(group_terms)
                else:
                    group_peaks = peaks
                active = mask_all & np.isfinite(term_values)
                if active.any():
                    dmat = np.abs(term_values[active, None] - group_peaks[None, :])
                    near_ix = dmat.argmin(axis=1)
                    contract_near[active] = group_peaks[near_ix]
                    contract_dist[active] = dmat[np.arange(len(near_ix)), near_ix]
                    contract_id[active] = near_ix
        else:
            contract_near, contract_dist, contract_id = nearest.copy(), dist.copy(), seg_id.copy()
        put("HC_SEG_CONTRACT_PEAK", contract_near)
        put("HC_SEG_CONTRACT_ID", contract_id)
        put("HC_SEG_CONTRACT_DIST", term_values - contract_near)
        put("HC_SEG_CONTRACT_ABSDIST", contract_dist)
        # Approximate monthly implied rates for fixed and observed term assumptions.
        credit = pd.to_numeric(raw["AMT_CREDIT"], errors="coerce").to_numpy(dtype="float64")
        annuity = pd.to_numeric(raw["AMT_ANNUITY"], errors="coerce").to_numpy(dtype="float64")
        def solve_implied_rate(principal, payment, months, iterations=38):
            principal = np.asarray(principal, dtype="float64")
            payment = np.asarray(payment, dtype="float64")
            months = np.asarray(months, dtype="float64")
            valid_r = np.isfinite(principal) & np.isfinite(payment) & np.isfinite(months) & (principal > 0) & (payment > 0) & (months > 0)
            lo = np.full(len(principal), 1e-7)
            hi = np.full(len(principal), 0.10)
            safe_months = np.clip(months, 1, 120)
            for _ in range(iterations):
                mid = (lo + hi) / 2
                denom = 1.0 - np.power(1.0 + mid, -safe_months)
                expected = principal * mid / np.maximum(denom, 1e-12)
                too_high = expected > payment
                hi = np.where(too_high, mid, hi)
                lo = np.where(too_high, lo, mid)
            rate = (lo + hi) / 2
            rate[~valid_r] = np.nan
            return rate
        for months in (12, 24, 36, 48, 60):
            put(f"HC_IMPLIED_RATE_T{months}", solve_implied_rate(credit, annuity, np.full(len(raw), months)))
        put("HC_IMPLIED_RATE_OBS_TERM", solve_implied_rate(credit, annuity, term_values))
        # Heuristic "most plausible catalogue term": the candidate term whose annualised rate is closest to 25%
        # among rates in [2%, 90%]. An argmin over 10 candidates is hard for trees to rebuild from the raw rates.
        best_term = np.full(len(raw), np.nan)
        best_rate = np.full(len(raw), np.nan)
        best_gap = np.full(len(raw), np.inf)
        for months in range(6, 61, 6):
            r_m = solve_implied_rate(credit, annuity, np.full(len(raw), months))
            r_ann = np.power(1.0 + r_m, 12) - 1.0
            ok = np.isfinite(r_ann) & (r_ann >= 0.02) & (r_ann <= 0.90)
            gap = np.abs(r_ann - 0.25)
            better = ok & (gap < best_gap)
            best_term[better], best_rate[better], best_gap[better] = months, r_ann[better], gap[better]
        put("HC_IMPL_TERM", best_term)
        put("HC_IMPL_RATE_ANN", best_rate)
        put("HC_IMPL_TERM_RESID", term_values - best_term)
    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(raw.columns):
        put("HC_CREDIT_INCOME_RATIO", safe_div(raw["AMT_CREDIT"], raw["AMT_INCOME_TOTAL"]))
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(raw.columns):
        put("HC_ANNUITY_INCOME_RATIO", safe_div(raw["AMT_ANNUITY"], raw["AMT_INCOME_TOTAL"]))
    if {"AMT_GOODS_PRICE", "AMT_CREDIT"}.issubset(raw.columns):
        put("HC_GOODS_CREDIT_RATIO", safe_div(raw["AMT_GOODS_PRICE"], raw["AMT_CREDIT"]))
        put("HC_CREDIT_GOODS_RATIO", safe_div(raw["AMT_CREDIT"], raw["AMT_GOODS_PRICE"]))
        put("HC_CREDIT_GOODS_DIFF", raw["AMT_CREDIT"] - raw["AMT_GOODS_PRICE"])
    if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"}.issubset(raw.columns):
        put("HC_INCOME_PER_FAMILY_MEMBER", safe_div(raw["AMT_INCOME_TOTAL"], raw["CNT_FAM_MEMBERS"]))
    if {"AMT_INCOME_TOTAL", "CNT_CHILDREN"}.issubset(raw.columns):
        put("HC_INCOME_PER_CHILD_PLUS1", safe_div(raw["AMT_INCOME_TOTAL"], raw["CNT_CHILDREN"] + 1))
    if args.ratio_income_stress and {"AMT_ANNUITY", "CNT_FAM_MEMBERS", "AMT_INCOME_TOTAL"}.issubset(raw.columns):
        # This is the only materially new ratio from example_code.py; its other
        # suggested ratios already exist here as HC_CREDIT_TERM,
        # HC_ANNUITY_INCOME_RATIO, HC_CREDIT_INCOME_RATIO,
        # HC_GOODS_CREDIT_RATIO, HC_EMPLOYED_BIRTH_RATIO, and
        # HC_INCOME_PER_CHILD_PLUS1.
        put("HC_INCOME_STRESS_INDEX",
            safe_div(raw["AMT_ANNUITY"] * (raw["CNT_FAM_MEMBERS"] + 1), raw["AMT_INCOME_TOTAL"]))
    if (args.equation_disposable_income or args.equation_disposable_component) and {
        "AMT_INCOME_TOTAL", "AMT_ANNUITY", "AMT_CREDIT", "CNT_FAM_MEMBERS"
    }.issubset(raw.columns):
        # Approximate monthly cash remaining after the scheduled installment.
        # These are deterministic basis features derived only from input columns;
        # no history-table or target information is used.
        income = pd.to_numeric(raw["AMT_INCOME_TOTAL"], errors="coerce")
        annuity = pd.to_numeric(raw["AMT_ANNUITY"], errors="coerce")
        credit = pd.to_numeric(raw["AMT_CREDIT"], errors="coerce")
        surplus = income - annuity
        positive_surplus = surplus.where(surplus > 0)
        credit_months = safe_div(credit, positive_surplus)
        selected_equations = ({args.equation_disposable_component}
                              if args.equation_disposable_component else
                              {"surplus", "per_member", "credit_months", "log_credit_months"})
        if "surplus" in selected_equations:
            put("HC_DISPOSABLE_INCOME", surplus)
        if "per_member" in selected_equations:
            put("HC_DISPOSABLE_INCOME_PER_MEMBER", safe_div(surplus, raw["CNT_FAM_MEMBERS"]))
        if "credit_months" in selected_equations:
            put("HC_CREDIT_MONTHS_OF_SURPLUS", credit_months)
        if "log_credit_months" in selected_equations:
            put("HC_LOG_CREDIT_MONTHS_OF_SURPLUS", np.log1p(credit_months.clip(lower=0)))
    if "DAYS_BIRTH" in raw.columns:
        put("HC_AGE_YEARS", -raw["DAYS_BIRTH"] / 365.25)
        if args.autolab_age_end and {"AMT_CREDIT", "AMT_ANNUITY"}.issubset(raw.columns):
            age_years = -pd.to_numeric(raw["DAYS_BIRTH"], errors="coerce") / 365.25
            put("HC_AGE_END_LOAN", age_years + credit_term / 12.0)
            put("HC_AGE_END_TERM_PEAK", age_years + pd.Series(nearest, index=raw.index) / 12.0)
    if {"DAYS_EMPLOYED", "DAYS_BIRTH"}.issubset(raw.columns):
        put("HC_EMPLOYED_BIRTH_RATIO", safe_div(raw["DAYS_EMPLOYED"], raw["DAYS_BIRTH"]))
        put("HC_AGE_AT_JOB_START", (raw["DAYS_BIRTH"].abs() - raw["DAYS_EMPLOYED"].abs()) / 365.25)
    if args.autolab_days_diffs:
        if {"DAYS_ID_PUBLISH", "DAYS_BIRTH"}.issubset(raw.columns):
            put("HC_AGE_AT_ID_PUBLISH", (raw["DAYS_ID_PUBLISH"] - raw["DAYS_BIRTH"]) / 365.25)
        if {"DAYS_REGISTRATION", "DAYS_EMPLOYED"}.issubset(raw.columns):
            put("HC_REG_MINUS_EMPLOYED", raw["DAYS_REGISTRATION"] - raw["DAYS_EMPLOYED"])
    if args.client_dates:
        if {"DAYS_BIRTH", "DAYS_REGISTRATION"}.issubset(raw.columns):
            put("HC_AGE_AT_REGISTRATION", (raw["DAYS_BIRTH"].abs() - raw["DAYS_REGISTRATION"].abs()) / 365.25)
        if {"DAYS_BIRTH", "DAYS_LAST_PHONE_CHANGE"}.issubset(raw.columns):
            put("HC_AGE_AT_PHONE_CHANGE", (raw["DAYS_BIRTH"].abs() - raw["DAYS_LAST_PHONE_CHANGE"].abs()) / 365.25)
        if {"DAYS_LAST_PHONE_CHANGE", "DAYS_REGISTRATION"}.issubset(raw.columns):
            put("HC_PHONE_REGISTRATION_DIFF", raw["DAYS_LAST_PHONE_CHANGE"] - raw["DAYS_REGISTRATION"])
        employed_date = None
        if "DAYS_EMPLOYED" in raw.columns:
            employed_date = pd.to_numeric(raw["DAYS_EMPLOYED"], errors="coerce")
            employed_date = employed_date.where(employed_date <= 0)
        if employed_date is not None and "DAYS_ID_PUBLISH" in raw.columns:
            put("HC_ID_EMPLOYED_DIFF", raw["DAYS_ID_PUBLISH"] - employed_date)
        if employed_date is not None and "DAYS_LAST_PHONE_CHANGE" in raw.columns:
            put("HC_PHONE_EMPLOYED_DIFF", raw["DAYS_LAST_PHONE_CHANGE"] - employed_date)
    if args.client_id_renewal and {"DAYS_BIRTH", "DAYS_ID_PUBLISH"}.issubset(raw.columns):
        age_years = -pd.to_numeric(raw["DAYS_BIRTH"], errors="coerce") / 365.25
        id_years = -pd.to_numeric(raw["DAYS_ID_PUBLISH"], errors="coerce") / 365.25
        milestone = np.where(age_years >= 45, 45.0, np.where(age_years >= 20, 20.0, 14.0))
        renewal_delay = age_years - milestone - id_years
        delay_abs = pd.Series(renewal_delay, index=raw.index).abs().clip(0.0, 5.0)
        punctual = np.where(delay_abs.notna(), (delay_abs <= 0.25).astype("float32"), np.nan)
        put("HC_ID_RENEWAL_DELAY", renewal_delay)
        put("HC_ID_DELAY_ABS", delay_abs)
        put("HC_ID_IS_PUNCTUAL", punctual)
        put("HC_ID_DELAY_X_AGE", safe_div(delay_abs, (age_years - 13.0).clip(1.0, 100.0)))
    if {"DAYS_REGISTRATION", "DAYS_ID_PUBLISH"}.issubset(raw.columns):
        put("HC_REG_ID_PUBLISH_DIFF", raw["DAYS_REGISTRATION"] - raw["DAYS_ID_PUBLISH"])
    if {"DAYS_LAST_PHONE_CHANGE", "DAYS_ID_PUBLISH"}.issubset(raw.columns):
        put("HC_PHONE_ID_DIFF", raw["DAYS_LAST_PHONE_CHANGE"] - raw["DAYS_ID_PUBLISH"])
    if {"AMT_CREDIT", "AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(raw.columns):
        put("HC_CREDIT_PLUS_ANNUITY_OVER_INCOME", safe_div(raw["AMT_CREDIT"] + raw["AMT_ANNUITY"], raw["AMT_INCOME_TOTAL"]))

    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in raw.columns]
    if ext_cols:
        ext = raw[ext_cols]
        put("HC_EXT_MEAN", ext.mean(axis=1))
        put("HC_EXT_MIN", ext.min(axis=1))
        put("HC_EXT_MAX", ext.max(axis=1))
        put("HC_EXT_STD", ext.std(axis=1))
        put("HC_EXT_MISSING_COUNT", ext.isna().sum(axis=1))
        put("HC_EXT_PRODUCT", ext.prod(axis=1, min_count=len(ext_cols)))
        for i, c1 in enumerate(ext_cols):
            for c2 in ext_cols[i + 1:]:
                put(f"HC_{c1}_{c2}_DIFF", raw[c1] - raw[c2])
                put(f"HC_{c1}_{c2}_PRODUCT", raw[c1] * raw[c2])
        for c in ext_cols:
            if "DAYS_BIRTH" in raw.columns:
                put(f"HC_{c}_X_AGE", raw[c] * (-raw["DAYS_BIRTH"] / 365.25))
            if "AMT_CREDIT" in raw.columns and "AMT_INCOME_TOTAL" in raw.columns:
                put(f"HC_{c}_X_CREDIT_INCOME", raw[c] * safe_div(raw["AMT_CREDIT"], raw["AMT_INCOME_TOTAL"]))

    bureau_req = [c for c in raw.columns if c.startswith("AMT_REQ_CREDIT_BUREAU_")]
    if bureau_req:
        put("HC_BUREAU_REQ_SUM", raw[bureau_req].sum(axis=1, min_count=1))
        put("HC_BUREAU_REQ_MISSING", raw[bureau_req].isna().sum(axis=1))

    # Ratios among amounts; intentionally omit duplicate identical ratios.
    monetary = [c for c in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"] if c in raw.columns]
    for i, c1 in enumerate(monetary):
        for c2 in monetary[i + 1:]:
            put(f"HC_RATIO_{c1}_{c2}", safe_div(raw[c1], raw[c2]))

    uid_feats, uid_gid, uid_time = uid_structure(raw)
    uid_meta = pd.DataFrame({"gid": uid_gid, "time": uid_time}, index=raw.index)
    if args.autolab_uid and not args.no_uid:
        auto_groups, auto_sizes = autolab_uid_groups(raw)
        for size_name, arr in auto_sizes.items():
            group_name = size_name.removesuffix("_SIZE")
            put(size_name, arr)
            uid_meta[group_name + "_GID"] = auto_groups[group_name]
    if not args.no_uid:
        for name, arr in uid_feats.items():
            put(name, arr)

    features = pd.concat([raw, pd.DataFrame(num, index=raw.index)], axis=1)
    if args.autolab_value_counts:
        for col in ("AMT_INCOME_TOTAL", "AMT_GOODS_PRICE", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH", "DAYS_REGISTRATION"):
            if col in features:
                counts = features[col].value_counts(dropna=False)
                features[f"HC_AUTO_VALUE_COUNT_{col}"] = features[col].map(counts).astype("float32")
    if ext_cols and not args.no_ext_impute:
        log("Imputing EXT_SOURCE_* from the rest of the application (OOF, unsupervised) ...")
        ext_feats = ext_impute_features(features, ext_cols)
        if ext_feats:
            ext_df = pd.DataFrame(ext_feats, index=features.index).astype("float32")
            features = pd.concat([features, ext_df], axis=1)
            imputed = [c for c in features.columns if c.startswith("HC_EXTI_")]
            if imputed:
                features["HC_EXTI_MEAN"] = features[imputed].mean(axis=1).astype("float32")
    # Unsupervised rank/frequency views; fit over application train+test only, never uses TARGET.
    rank_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "HC_EXT_MEAN", "AMT_CREDIT", "AMT_ANNUITY", "AMT_INCOME_TOTAL", "DAYS_BIRTH", "HC_CREDIT_TERM"] if c in features]
    if args.autolab_rank_extra and "DAYS_EMPLOYED" in features:
        rank_cols.append("DAYS_EMPLOYED")
    for c in rank_cols:
        features[f"HC_RANK_{c}"] = features[c].rank(pct=True).astype("float32")
    if args.ext_copula and all(c in features for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")):
        # Empirical-copula coordinates are label-free ranks across the permitted
        # train+test application rows. Joint-tail indicators expose co-extremes
        # directly to trees instead of asking them to discover rare conjunctions.
        source_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        uniforms = np.column_stack([features[c].rank(pct=True).to_numpy(dtype="float64") for c in source_cols])
        normal_scores = ndtri(np.clip(uniforms, 1e-5, 1.0 - 1e-5))
        finite = np.isfinite(uniforms)
        all_present = finite.all(axis=1)
        features["HC_EXT_COPULA_MIN"] = np.where(all_present, normal_scores.min(axis=1), np.nan).astype("float32")
        features["HC_EXT_COPULA_MAX"] = np.where(all_present, normal_scores.max(axis=1), np.nan).astype("float32")
        features["HC_EXT_COPULA_MEAN"] = np.where(all_present, normal_scores.mean(axis=1), np.nan).astype("float32")
        features["HC_EXT_COPULA_RANGE"] = np.where(all_present, normal_scores.max(axis=1) - normal_scores.min(axis=1), np.nan).astype("float32")
        for tail, condition in (("LOW", uniforms <= 0.10), ("HIGH", uniforms >= 0.90)):
            features[f"HC_EXT_{tail}_COUNT"] = (condition & finite).sum(axis=1).astype("float32")
            features[f"HC_EXT_{tail}_ALL3"] = (condition & finite).all(axis=1).astype("int8")
            for i, j, name in ((0, 1, "12"), (0, 2, "13"), (1, 2, "23")):
                features[f"HC_EXT_{tail}_PAIR_{name}"] = (condition[:, i] & condition[:, j] & finite[:, i] & finite[:, j]).astype("int8")
        features["HC_EXT_COPULA_MISSING_COUNT"] = (~finite).sum(axis=1).astype("float32")
    for c in [x for x in features.columns if features[x].dtype == "object"]:
        features[f"HC_FREQ_{c}"] = features[c].map(features[c].value_counts(dropna=False)).astype("float32")

    # Target-encoding keys are kept separate; never exposed directly as arbitrary integer codes.
    keys = pd.DataFrame(index=raw.index)
    credit_goods = safe_div(raw["AMT_CREDIT"], raw["AMT_GOODS_PRICE"]) if {"AMT_CREDIT", "AMT_GOODS_PRICE"}.issubset(raw.columns) else pd.Series(np.nan, index=raw.index)
    term = safe_div(raw["AMT_CREDIT"], raw["AMT_ANNUITY"]) if {"AMT_CREDIT", "AMT_ANNUITY"}.issubset(raw.columns) else pd.Series(np.nan, index=raw.index)
    pay_rate = safe_div(raw["AMT_ANNUITY"], raw["AMT_CREDIT"]) if {"AMT_CREDIT", "AMT_ANNUITY"}.issubset(raw.columns) else pd.Series(np.nan, index=raw.index)

    def as_key(s: pd.Series) -> pd.Series:
        return s.astype("string").fillna("__MISSING__").astype(str)

    keys["TAR_CREDIT_GOODS_2D"] = as_key(credit_goods.round(2))
    keys["TAR_TERM_1D"] = as_key(term.round(1))
    keys["TAR_TERM_2D"] = as_key(term.round(2))
    keys["TAR_PAYRATE_4D"] = as_key(pay_rate.round(4))
    keys["TAR_CG_TERM"] = as_key(credit_goods.round(2)) + "|" + as_key(term.round(1))
    if "HC_SEG_PEAK" in features:
        keys["TAR_TERM_PEAK"] = as_key(features["HC_SEG_PEAK"].round(1))
        keys["TAR_CONTRACT_TERM_PEAK"] = as_key(features["HC_SEG_CONTRACT_PEAK"].round(1))
    if "AMT_CREDIT" in raw:
        keys["TAR_CREDIT_ROUND_1K"] = as_key(pd.to_numeric(raw["AMT_CREDIT"], errors="coerce").round(-3))
        keys["TAR_CREDIT_ROUND_10K"] = as_key(pd.to_numeric(raw["AMT_CREDIT"], errors="coerce").round(-4))
    if "AMT_INCOME_TOTAL" in raw:
        keys["TAR_INCOME_ROUND_1K"] = as_key(pd.to_numeric(raw["AMT_INCOME_TOTAL"], errors="coerce").round(-3))
    keys["TAR_CONTRACT_TERM"] = as_key(raw["NAME_CONTRACT_TYPE"] if "NAME_CONTRACT_TYPE" in raw else pd.Series("__MISSING__", index=raw.index)) + "|" + as_key(term.round(1))
    if (args.id_features or args.id_te_only) and id_values is not None:
        # These label statistics are later cross-fitted inside each outer training fold.
        for width in (1_000, 5_000, 20_000):
            keys[f"TAR_ID_BLOCK_{width}"] = as_key(np.floor(id_values / width))
    if args.autolab_tariff:
        # AutoLab's discrete tariff keys. The repeated CA1/CA2 signals already exist as
        # TAR_TERM_1D/2D above, so only distinct keys are added to supervised TE.
        goods_annuity = safe_div(raw["AMT_GOODS_PRICE"], raw["AMT_ANNUITY"]) if {"AMT_GOODS_PRICE", "AMT_ANNUITY"}.issubset(raw.columns) else pd.Series(np.nan, index=raw.index)
        credit_minus_annuity = pd.to_numeric(raw["AMT_CREDIT"], errors="coerce") - pd.to_numeric(raw["AMT_ANNUITY"], errors="coerce")
        tariff_keys = {
            "TAR_AUTO_PR3": as_key(pay_rate.round(3)),
            "TAR_AUTO_CA1_PR3": as_key(term.round(1)) + "|" + as_key(pay_rate.round(3)),
            "TAR_AUTO_PR3_GA1": as_key(pay_rate.round(3)) + "|" + as_key(goods_annuity.round(1)),
            "TAR_AUTO_CMA": as_key(credit_minus_annuity.round(-2)),
            "TAR_AUTO_CG3": as_key(credit_goods.round(3)),
        }
        if "HC_SEG_PEAK" in features and {"AMT_CREDIT", "AMT_GOODS_PRICE"}.issubset(raw.columns):
            ratio_values = pd.to_numeric(credit_goods, errors="coerce").to_numpy(dtype="float64")
            ratio_train = ratio_values[:n_train]
            goods_peaks = find_distribution_peaks(ratio_train)
            goods_nearest = np.full(len(ratio_values), np.nan, dtype="float64")
            goods_valid = np.isfinite(ratio_values)
            if len(goods_peaks) and goods_valid.any():
                dm = np.abs(ratio_values[goods_valid, None] - goods_peaks[None, :])
                goods_nearest[goods_valid] = goods_peaks[dm.argmin(axis=1)]
            goods_nearest_s = pd.Series(goods_nearest, index=raw.index)
            term_nearest_s = pd.Series(nearest, index=raw.index)
            features["HC_AUTO_GOODS_PEAK"] = finite_float32(goods_nearest_s)
            features["HC_AUTO_GOODS_PEAK_DIST"] = finite_float32(pd.Series(ratio_values - goods_nearest, index=raw.index))
            features["HC_AUTO_GOODS_PEAK_RATIO"] = finite_float32(safe_div(pd.Series(ratio_values, index=raw.index), goods_nearest_s))
            seg1 = as_key(term_nearest_s.round(2))
            contract_key = as_key(raw["NAME_CONTRACT_TYPE"] if "NAME_CONTRACT_TYPE" in raw else pd.Series("__MISSING__", index=raw.index))
            goods_key = as_key(goods_nearest_s.round(2))
            tariff_keys["TAR_AUTO_SEG1"] = seg1
            tariff_keys["TAR_AUTO_SEG2"] = seg1 + "|" + contract_key
            tariff_keys["TAR_AUTO_SEG3"] = seg1 + "|" + goods_key
        for key_name, key_values in tariff_keys.items():
            keys[key_name] = key_values
            # Counts are unsupervised and computed only from the supplied train+test applications.
            counts = key_values.value_counts(dropna=False)
            features[f"HC_{key_name}_SIZE"] = key_values.map(counts).to_numpy(dtype="float32")
        size_only_keys = {
            "TAR_AUTO_CA1": as_key(term.round(1)),
            "TAR_AUTO_CA2": as_key(term.round(2)),
            "TAR_AUTO_CG2": as_key(credit_goods.round(2)),
        }
        for key_name, key_values in size_only_keys.items():
            counts = key_values.value_counts(dropna=False)
            features[f"HC_{key_name}_SIZE"] = key_values.map(counts).to_numpy(dtype="float32")
    if not args.no_product_x:
        # Exact-amount keys: identical prices/instalments identify the same product package (smoothed by alpha).
        for col, name in [("AMT_GOODS_PRICE", "TAR_GOODS_EXACT"), ("AMT_CREDIT", "TAR_CREDIT_EXACT"), ("AMT_ANNUITY", "TAR_ANNUITY_EXACT")]:
            if col in raw:
                keys[name] = as_key(pd.to_numeric(raw[col], errors="coerce"))
        if {"AMT_CREDIT", "AMT_GOODS_PRICE"}.issubset(raw.columns):
            eq = (pd.to_numeric(raw["AMT_CREDIT"], errors="coerce") == pd.to_numeric(raw["AMT_GOODS_PRICE"], errors="coerce")).astype(int).astype(str)
            keys["TAR_T1_x_GOODS_EQ_CREDIT"] = as_key(term.round(1)) + "|" + eq
    age = -pd.to_numeric(raw["DAYS_BIRTH"], errors="coerce") / 365.25 if "DAYS_BIRTH" in raw else pd.Series(np.nan, index=raw.index)
    age_bin = (np.floor(age / 5) * 5).astype("Int64").astype("string")
    for col, name in [("ORGANIZATION_TYPE", "ORG_OCC"), ("NAME_EDUCATION_TYPE", "EDU_INCOME"), ("OCCUPATION_TYPE", "OCC_AGE")]:
        if col == "ORGANIZATION_TYPE":
            left, right = raw.get("ORGANIZATION_TYPE", pd.Series("__MISSING__", index=raw.index)), raw.get("OCCUPATION_TYPE", pd.Series("__MISSING__", index=raw.index))
        elif col == "NAME_EDUCATION_TYPE":
            left, right = raw.get("NAME_EDUCATION_TYPE", pd.Series("__MISSING__", index=raw.index)), raw.get("NAME_INCOME_TYPE", pd.Series("__MISSING__", index=raw.index))
        else:
            left, right = raw.get("OCCUPATION_TYPE", pd.Series("__MISSING__", index=raw.index)), age_bin
        keys[name] = as_key(left) + "|" + as_key(right)

    if args.finance_extra_keys:
        # Algebraic product/term keys; target statistics are still learned only
        # inside the matching outer-training fold by manual TE or CatBoost.
        if {"AMT_GOODS_PRICE", "AMT_ANNUITY"}.issubset(raw.columns):
            goods_annuity = safe_div(raw["AMT_GOODS_PRICE"], raw["AMT_ANNUITY"])
            annuity_goods = safe_div(raw["AMT_ANNUITY"], raw["AMT_GOODS_PRICE"])
            keys["FIN_GOODS_ANNUITY_1D"] = as_key(goods_annuity.round(1))
            keys["FIN_ANNUITY_GOODS_3D"] = as_key(annuity_goods.round(3))
            if "NAME_CONTRACT_TYPE" in raw:
                keys["FIN_GA1_CONTRACT"] = as_key(goods_annuity.round(1)) + "|" + as_key(raw["NAME_CONTRACT_TYPE"])
        keys["FIN_CREDIT_GOODS_4D"] = as_key(credit_goods.round(4))
        keys["FIN_PR3_AGE5"] = as_key(pay_rate.round(3)) + "|" + as_key(age_bin)
        keys["FIN_TERM_AGE5"] = as_key(term.round(1)) + "|" + as_key(age_bin)
        for column, label in (("NAME_CONTRACT_TYPE", "CONTRACT"), ("NAME_INCOME_TYPE", "INCOME")):
            if column in raw:
                keys[f"FIN_PR3_{label}"] = as_key(pay_rate.round(3)) + "|" + as_key(raw[column])
        if "AMT_CREDIT" in raw:
            credit_bin = as_key(pd.to_numeric(raw["AMT_CREDIT"], errors="coerce").round(-4))
            keys["FIN_PR3_CREDIT10K"] = as_key(pay_rate.round(3)) + "|" + credit_bin
        if "AMT_GOODS_PRICE" in raw:
            goods_bin = as_key(pd.to_numeric(raw["AMT_GOODS_PRICE"], errors="coerce").round(-4))
            keys["FIN_TERM_GOODS10K"] = as_key(term.round(1)) + "|" + goods_bin

    if args.peer_context:
        # Peer population summaries use application covariates only, following
        # the same transductive policy as existing rank/frequency features.
        groups = {}
        for label, cols in (("EDU_OCC", ["NAME_EDUCATION_TYPE", "OCCUPATION_TYPE"]),
                            ("ORG_OCC", ["ORGANIZATION_TYPE", "OCCUPATION_TYPE"]),
                            ("EDU_INCOME", ["NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE"])):
            if all(c in raw for c in cols):
                groups[label] = as_key(raw[cols[0]]) + "|" + as_key(raw[cols[1]])
        if "CODE_GENDER" in raw:
            groups["AGE_SEX"] = age_bin.fillna("__MISSING__") + "|" + as_key(raw["CODE_GENDER"])
        if "NAME_EDUCATION_TYPE" in raw:
            groups["AGE_EDU"] = age_bin.fillna("__MISSING__") + "|" + as_key(raw["NAME_EDUCATION_TYPE"])
        values = [c for c in ("AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "HC_CREDIT_TERM",
                              "HC_CREDIT_INCOME_RATIO", "HC_ANNUITY_INCOME_RATIO", "HC_EXT_MEAN") if c in features]
        for label, group in groups.items():
            group = group.astype(str)
            count = group.map(group.value_counts()).astype("float64")
            reliability = count / (count + 30.0)
            features[f"HC_PEER_{label}_COUNT"] = count
            for col in values:
                value = pd.to_numeric(features[col], errors="coerce")
                median = value.groupby(group, observed=True).transform("median")
                global_median = float(value.median())
                center = reliability * median + (1.0 - reliability) * global_median
                if col.startswith("AMT_"):
                    relative = np.log1p(value.clip(lower=0)) - np.log1p(center.clip(lower=0))
                else:
                    relative = value - center
                features[f"HC_PEER_{label}_{col}_REL"] = relative.astype("float32")

    # Categories are defined over the combined train+test application rows, so fold matrices share vocabularies.
    for c in features.select_dtypes(include=["object", "string"]).columns:
        features[c] = features[c].astype("string").fillna("__MISSING__").astype("category")
    for c in features.columns:
        if pd.api.types.is_bool_dtype(features[c]):
            features[c] = features[c].astype("int8")
        elif not isinstance(features[c].dtype, pd.CategoricalDtype):
            features[c] = pd.to_numeric(features[c], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
    features = features.loc[:, features.nunique(dropna=False) > 1]
    return features, keys, uid_meta


# ---------------------------------------------------------------------------
# Train/test-only feature construction

def build_features(app_all: pd.DataFrame, ids_all: np.ndarray, n_train: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build features exclusively from the supplied train.csv and test.csv rows."""
    feat, te_keys, uid_meta = application_features(app_all, n_train)
    feat.index = pd.Index(ids_all, name=ID)
    te_keys.index = pd.Index(ids_all, name=ID)
    uid_meta.index = pd.Index(ids_all, name=ID)

    feat = feat.replace([np.inf, -np.inf], np.nan)
    const = [c for c in feat.columns if feat[c].nunique(dropna=False) <= 1]
    if const:
        feat = feat.drop(columns=const)
    for c in feat.columns:
        if isinstance(feat[c].dtype, pd.CategoricalDtype):
            if "__MISSING__" not in feat[c].cat.categories:
                feat[c] = feat[c].cat.add_categories(["__MISSING__"])
            feat[c] = feat[c].fillna("__MISSING__")
        elif feat[c].dtype == "object" or pd.api.types.is_string_dtype(feat[c].dtype):
            feat[c] = feat[c].astype("string").fillna("__MISSING__").astype("category")
        else:
            feat[c] = pd.to_numeric(feat[c], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
    log(f"Application-only feature matrix: {feat.shape[0]:,} rows x {feat.shape[1]:,} features; categoricals={len(feat.select_dtypes(include=['category']).columns)}")
    return feat, te_keys, uid_meta


# ---------------------------------------------------------------------------
# Target encoding, rigorously cross-fitted within each outer training fold

def normalize_key(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("__MISSING__").astype(str)


def te_apply(k_fit: pd.Series, y_fit: np.ndarray, k_apply: pd.Series, alpha: float) -> np.ndarray:
    keys_fit = normalize_key(k_fit)
    keys_apply = normalize_key(k_apply)
    stats = pd.DataFrame({"k": keys_fit.to_numpy(), "y": y_fit.astype("float32")}).groupby("k", sort=False, observed=True)["y"].agg(["sum", "count"])
    prior = float(np.mean(y_fit))
    sums = keys_apply.map(stats["sum"]).fillna(0).to_numpy(dtype="float64")
    counts = keys_apply.map(stats["count"]).fillna(0).to_numpy(dtype="float64")
    return ((sums + alpha * prior) / (counts + alpha)).astype("float32")


def add_crossfit_te(
    Xtr: pd.DataFrame,
    Xva: pd.DataFrame,
    Xte: pd.DataFrame,
    key_tr: pd.DataFrame,
    key_va: pd.DataFrame,
    key_te: pd.DataFrame,
    ytr: np.ndarray,
    seed: int,
    inner_folds: int,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if inner_folds < 2:
        return Xtr, Xva, Xte
    pos, neg = int(ytr.sum()), int(len(ytr) - ytr.sum())
    n_inner = min(inner_folds, pos, neg)
    if n_inner < 2:
        return Xtr, Xva, Xte
    inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + 1009)
    key_cols = list(key_tr.columns)
    for j, name in enumerate(key_cols):
        kt = key_tr[name].reset_index(drop=True)
        oof = np.zeros(len(ytr), dtype="float32")
        for it, iv in inner.split(np.zeros(len(ytr)), ytr):
            oof[iv] = te_apply(kt.iloc[it], ytr[it], kt.iloc[iv], alpha)
        full_prior = float(ytr.mean())
        full_key = normalize_key(kt)
        stat = pd.DataFrame({"k": full_key.to_numpy(), "y": ytr.astype("float32")}).groupby("k", sort=False, observed=True)["y"].agg(["sum", "count"])
        def apply_full(k):
            ka = normalize_key(k)
            sums = ka.map(stat["sum"]).fillna(0).to_numpy(dtype="float64")
            counts = ka.map(stat["count"]).fillna(0).to_numpy(dtype="float64")
            return ((sums + alpha * full_prior) / (counts + alpha)).astype("float32")
        feat_name = f"HC_TE_{name}"
        Xtr[feat_name] = oof
        Xva[feat_name] = apply_full(key_va[name])
        Xte[feat_name] = apply_full(key_te[name])
    return Xtr, Xva, Xte


# ---------------------------------------------------------------------------
# Model training / caching

def parse_seeds(raw: str) -> list[int]:
    seeds = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return list(dict.fromkeys(seeds))


def lgb_params(seed: int, extra_trees: bool = False) -> dict:
    if args.quick:
        n_estimators, lr, es_rounds = min(args.lgbm_estimators, 800), 0.04, 80
        leaves, min_child = 24, 100
    else:
        n_estimators, lr, es_rounds = args.lgbm_estimators, 0.02, 250
        leaves, min_child = 31, 120
    p = {
        "objective": "binary", "n_estimators": n_estimators, "learning_rate": lr,
        "num_leaves": leaves, "max_depth": -1, "min_child_samples": min_child,
        "subsample": 0.82, "subsample_freq": 1, "colsample_bytree": 0.82,
        "reg_alpha": 0.1, "reg_lambda": 8.0, "min_split_gain": 0.0,
        "max_bin": 255, "n_jobs": args.threads, "random_state": seed,
        "verbosity": -1, "force_row_wise": True, "deterministic": True,
    }
    if args.lgbm_profile == "autolab" and not extra_trees:
        # Settings from the saved AutoLab champion configuration; all data and fold
        # handling remain controlled by this train/test-only pipeline.
        p.update({
            "num_leaves": 8, "min_child_samples": 250, "subsample": 0.7,
            "colsample_bytree": 0.3, "reg_alpha": 6.7, "reg_lambda": 0.0,
            "min_split_gain": 0.2,
        })
    if not extra_trees:
        p.update(FIXED_LGBM_OVERRIDES)
    if extra_trees:
        # Randomised split thresholds: weaker single model, but decorrelated from the standard LGBM in the blend.
        p.update({"extra_trees": True, "num_leaves": 48, "min_child_samples": 80, "colsample_bytree": 0.6,
                  "learning_rate": lr * 1.5, "random_state": seed + 101})
    return p


def cb_params(seed: int) -> dict:
    if args.quick:
        iters, lr, depth, od = min(args.catboost_iterations, 700), 0.05, 6, 80
    else:
        iters, lr, depth, od = args.catboost_iterations, 0.035, 7, 300
    return {
        "iterations": iters, "learning_rate": lr, "depth": depth, "l2_leaf_reg": 8.0,
        "loss_function": "Logloss", "eval_metric": "AUC", "random_seed": seed,
        "thread_count": args.threads, "verbose": 200, "allow_writing_files": False,
        "task_type": args.catboost_task_type, "od_type": "Iter", "od_wait": od,
    }


def xgb_params(seed: int) -> dict:
    if args.quick:
        n_estimators, lr = min(args.xgb_estimators, 700), 0.04
    else:
        n_estimators, lr = args.xgb_estimators, 0.025
    return {
        "objective": "binary:logistic", "eval_metric": "auc", "n_estimators": n_estimators,
        "learning_rate": lr, "max_depth": 6, "min_child_weight": 30,
        "subsample": 0.82, "colsample_bytree": 0.82, "reg_alpha": 0.1, "reg_lambda": 10.0,
        "tree_method": "hist", "enable_categorical": True, "max_cat_to_onehot": 8,
        "n_jobs": args.threads, "random_state": seed, "early_stopping_rounds": 250 if not args.quick else 80,
    }


def as_catboost_frame(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    # CatBoost handles string categoricals; make missing values explicit and stable across splits.
    Xc = X.copy()
    for c in cat_cols:
        Xc[c] = Xc[c].astype("string").fillna("__MISSING__").astype(str)
    return Xc


def as_extra_trees_matrix(X: pd.DataFrame) -> np.ndarray:
    """Ordinal-encode categoricals and keep a stable numeric column layout."""
    matrix = np.empty((len(X), X.shape[1]), dtype="float32")
    for j, col in enumerate(X.columns):
        series = X[col]
        if isinstance(series.dtype, pd.CategoricalDtype):
            values = series.cat.codes.to_numpy(dtype="float32", copy=True)
            values[values < 0] = np.nan
        elif pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
            values = pd.Categorical(series.astype("string")).codes.astype("float32", copy=True)
            values[values < 0] = np.nan
        else:
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float32", copy=True)
        values[~np.isfinite(values)] = np.nan
        matrix[:, j] = values
    return matrix


def extra_trees_params(seed: int) -> dict:
    return {
        "n_estimators": 150 if args.quick else 400,
        "max_depth": 12,
        "random_state": seed,
        "n_jobs": args.threads,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "class_weight": None,
    }


def fold_cache_path(model_name: str, seed: int, fold: int, model_hash: str) -> Path:
    return FOLD_CACHE_DIR / f"{model_name}_s{seed}_f{fold}_{DATA_TAG}_{model_hash}.npz"


def load_fold_cache(path: Path, expected_val: np.ndarray, n_test: int, n_features: int):
    if args.no_fold_cache or not path.exists():
        return None
    try:
        z = np.load(path, allow_pickle=False)
        vi = z["val_idx"]
        vp = z["val_pred"]
        tp = z["test_pred"]
        bi = int(z["best_iter"])
        auc = float(z["auc"])
        imp = z["importance"] if "importance" in z.files else np.zeros(n_features, dtype="float64")
        if np.array_equal(vi, expected_val) and len(vp) == len(expected_val) and len(tp) == n_test and len(imp) == n_features:
            return vi, vp, tp, bi, auc, imp
    except Exception as exc:
        log(f"Ignoring corrupt fold cache {path.name}: {exc}")
    return None


def save_fold_cache(path: Path, val_idx: np.ndarray, val_pred: np.ndarray, test_pred: np.ndarray,
                    best_iter: int, auc: float, importance: np.ndarray) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, val_idx=val_idx.astype("int32"), val_pred=val_pred.astype("float32"),
             test_pred=test_pred.astype("float32"), best_iter=np.int32(best_iter), auc=np.float64(auc),
             importance=np.asarray(importance, dtype="float32"))
    tmp.replace(path)


def add_fold_geometry(Xtr: pd.DataFrame, Xva: pd.DataFrame, Xte: pd.DataFrame,
                      Xfit: pd.DataFrame, Xes: pd.DataFrame, source_cols: list[str],
                      y_fit: np.ndarray, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                          pd.DataFrame, pd.DataFrame]:
    """Fit numeric preprocessing and UMAP on model-fit rows only; transform all other rows."""
    if importlib.util.find_spec("umap") is None:
        raise ImportError("--geometry requires umap-learn; install it with `python -m pip install umap-learn`")
    import umap

    fit_arr = Xfit.loc[:, source_cols].to_numpy(dtype="float64", copy=True)
    fit_arr[~np.isfinite(fit_arr)] = np.nan
    medians = np.nanmedian(fit_arr, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    fit_missing = ~np.isfinite(fit_arr)
    fit_arr[fit_missing] = np.broadcast_to(medians, fit_arr.shape)[fit_missing]
    means = np.mean(fit_arr, axis=0)
    scales = np.std(fit_arr, axis=0)
    scales = np.where(np.isfinite(scales) & (scales > 1e-8), scales, 1.0)

    def matrix(frame: pd.DataFrame) -> np.ndarray:
        arr = frame.loc[:, source_cols].to_numpy(dtype="float64", copy=True)
        missing = ~np.isfinite(arr)
        if missing.any():
            arr[missing] = np.broadcast_to(medians, arr.shape)[missing]
        scaled = ((arr - means) / scales).astype("float32")
        return np.concatenate([scaled, missing.astype("float32")], axis=1)

    fit_matrix = matrix(Xfit)
    max_fit = min(len(fit_matrix), args.umap_fit_sample)
    if max_fit < len(fit_matrix):
        rng = np.random.default_rng(seed)
        sample_idx = np.sort(rng.choice(len(fit_matrix), size=max_fit, replace=False))
        fit_sample = fit_matrix[sample_idx]
    else:
        fit_sample = fit_matrix
    reducer = umap.UMAP(n_components=args.umap_components, n_neighbors=args.umap_neighbors,
                        min_dist=args.umap_min_dist, metric=args.umap_metric,
                        random_state=seed, transform_seed=seed, n_jobs=1,
                        target_metric="categorical", target_weight=args.umap_target_weight)
    if args.geometry == "umap_supervised":
        fit_y = np.asarray(y_fit)
        sample_y = fit_y[sample_idx] if max_fit < len(fit_matrix) else fit_y
        reducer.fit(fit_sample, y=sample_y)
    else:
        reducer.fit(fit_sample)
    log(f"  UMAP fit: mode={args.geometry}, numeric={len(source_cols)}, fit_rows={len(fit_sample):,}, metric={args.umap_metric}, neighbors={args.umap_neighbors}")

    # Transform each unique pool once; Xfit/Xes are subsets of Xtr or Xva.
    tr_embed = reducer.transform(matrix(Xtr)).astype("float32")
    va_embed = reducer.transform(matrix(Xva)).astype("float32")
    te_embed = reducer.transform(matrix(Xte)).astype("float32")
    fit_pos = Xtr.index.get_indexer(Xfit.index)
    es_pos_in_tr = Xtr.index.get_indexer(Xes.index)
    es_pos_in_va = Xva.index.get_indexer(Xes.index)
    es_in_tr = bool((es_pos_in_tr >= 0).all())
    es_in_va = bool((es_pos_in_va >= 0).all())
    if (fit_pos < 0).any() or not (es_in_tr or es_in_va):
        raise RuntimeError("Could not align early-stopping/model-fit rows to UMAP-transformed fold rows")
    fit_embed = tr_embed[fit_pos]
    es_embed = tr_embed[es_pos_in_tr] if es_in_tr else va_embed[es_pos_in_va]
    frames = (Xtr, Xva, Xte, Xfit, Xes)
    embeddings = (tr_embed, va_embed, te_embed, fit_embed, es_embed)
    geometry_names = [f"HC_UMAP_{i + 1}" for i in range(args.umap_components)]
    for frame, emb in zip(frames, embeddings):
        frame[geometry_names] = emb
    if args.geometry == "umap_klein":
        # Map UMAP coordinates to a compact chart using quantiles of model-fit rows only.
        fit_embed = embeddings[3]
        lo, hi = np.quantile(fit_embed, [0.005, 0.995], axis=0)
        span = np.where(hi > lo, hi - lo, 1.0)
        for frame, emb in zip(frames, embeddings):
            uv = np.clip((emb - lo) / span, 0.0, 1.0) * (2.0 * np.pi)
            u, v = uv[:, 0], uv[:, 1]
            ring = 2.0 + np.cos(u / 2.0) * np.sin(v) - np.sin(u / 2.0) * np.sin(2.0 * v)
            frame["HC_KLEIN_X"] = (ring * np.cos(u)).astype("float32")
            frame["HC_KLEIN_Y"] = (ring * np.sin(u)).astype("float32")
            frame["HC_KLEIN_Z"] = (np.sin(u / 2.0) * np.sin(v) + np.cos(u / 2.0) * np.sin(2.0 * v)).astype("float32")
    return Xtr, Xva, Xte, Xfit, Xes


KNN_FEATURE_NAMES = ["HC_KNN_DIST_MEAN_20", "HC_KNN_DIST_MEAN_100",
                     "HC_KNN_RATE_20", "HC_KNN_RATE_100", "HC_KNN_RATE_300"]


def add_fold_knn_features(Xfit: pd.DataFrame, Xes: pd.DataFrame, Xva: pd.DataFrame,
                          Xte: pd.DataFrame, source_cols: list[str], yfit: np.ndarray,
                          seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add fold-local kNN features. Fit rows get cross-fitted label rates; all query pools use Xfit labels only."""
    try:
        from pynndescent import NNDescent
    except ImportError as exc:
        raise ImportError("--knn-features requires pynndescent; install it with `python -m pip install pynndescent`") from exc
    from sklearn.decomposition import PCA

    fit_values = Xfit.loc[:, source_cols].to_numpy(dtype="float32", copy=True)
    fit_values[~np.isfinite(fit_values)] = np.nan
    medians = np.nanmedian(fit_values, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype("float32")
    missing_fit = ~np.isfinite(fit_values)
    fit_values[missing_fit] = np.broadcast_to(medians, fit_values.shape)[missing_fit]
    means = fit_values.mean(axis=0)
    scales = fit_values.std(axis=0)
    scales = np.where(np.isfinite(scales) & (scales > 1e-6), scales, 1.0).astype("float32")

    def raw_matrix(frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, source_cols].to_numpy(dtype="float32", copy=True)
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.broadcast_to(medians, values.shape)[missing]
        values = (values - means) / scales
        return np.concatenate([values, missing.astype("float32")], axis=1)

    raw_fit = raw_matrix(Xfit)
    n_components = min(args.knn_components, raw_fit.shape[1], max(2, len(raw_fit) - 1))
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    fit_matrix = pca.fit_transform(raw_fit).astype("float32", copy=False)
    del raw_fit

    rng = np.random.default_rng(seed)

    def reference_indices(pool: np.ndarray) -> np.ndarray:
        cap = min(len(pool), args.knn_fit_sample)
        if cap < len(pool):
            return np.sort(rng.choice(pool, size=cap, replace=False))
        return pool

    def make_index(ref_idx: np.ndarray):
        if len(ref_idx) < 2:
            raise ValueError("kNN feature reference pool must contain at least two rows")
        max_k = min(300, len(ref_idx))
        index = NNDescent(fit_matrix[ref_idx], n_neighbors=min(max_k + 1, len(ref_idx)),
                          metric="euclidean", random_state=seed, n_jobs=-1,
                          low_memory=True, compressed=True)
        return index, ref_idx, max_k

    def query_features(index_bundle, query_frame: pd.DataFrame) -> np.ndarray:
        index, ref_idx, max_k = index_bundle
        query_matrix = pca.transform(raw_matrix(query_frame)).astype("float32", copy=False)
        neighbor_idx, neighbor_dist = index.query(query_matrix, k=max_k)
        neighbor_y = np.asarray(yfit)[ref_idx][neighbor_idx]
        prior = float(np.mean(yfit))
        features = np.zeros((len(query_frame), len(KNN_FEATURE_NAMES)), dtype="float32")
        for col, k in enumerate((20, 100)):
            kk = min(k, max_k)
            features[:, col] = neighbor_dist[:, :kk].mean(axis=1)
        for col, k in enumerate((20, 100, 300), start=2):
            kk = min(k, max_k)
            # Bayesian smoothing stabilizes small neighborhoods at the 8% event rate.
            features[:, col] = ((neighbor_y[:, :kk].sum(axis=1) + 20.0 * prior) /
                                (kk + 20.0)).astype("float32")
        del query_matrix, neighbor_idx, neighbor_dist, neighbor_y
        return features

    # Training features are cross-fitted: each training partition queries an index built
    # only from the other partitions, so its own labels cannot enter its neighbor rates.
    inner = StratifiedKFold(n_splits=min(args.knn_inner_folds, int(np.bincount(yfit.astype(int)).min())),
                            shuffle=True, random_state=seed)
    fit_features = np.zeros((len(Xfit), len(KNN_FEATURE_NAMES)), dtype="float32")
    for inner_no, (ref_local, query_local) in enumerate(inner.split(np.zeros(len(yfit)), yfit)):
        ref_global = reference_indices(ref_local)
        bundle = make_index(ref_global)
        fit_features[query_local] = query_features(bundle, Xfit.iloc[query_local])
        log(f"  kNN cross-fit {inner_no + 1}/{inner.n_splits}: reference={len(ref_global):,}, query={len(query_local):,}")

    full_ref = reference_indices(np.arange(len(Xfit)))
    full_bundle = make_index(full_ref)
    query_frames = (Xes, Xva, Xte)
    query_features_list = [query_features(full_bundle, frame) for frame in query_frames]
    Xfit = Xfit.copy()
    Xfit[KNN_FEATURE_NAMES] = fit_features
    outputs = []
    for frame, values in zip(query_frames, query_features_list):
        frame = frame.copy()
        frame[KNN_FEATURE_NAMES] = values
        outputs.append(frame)
    log(f"  kNN features added: PCA={n_components}, reference={len(full_ref):,}, inner_folds={inner.n_splits}")
    return Xfit, outputs[0], outputs[1], outputs[2]


def train_one_model(
    model_name: str,
    seed: int,
    X: pd.DataFrame,
    key_frame: pd.DataFrame,
    y: np.ndarray,
    n_train: int,
    folds: list[tuple[np.ndarray, np.ndarray]],
    uid_meta: dict | None = None,
    tag: str = "",
    lgb_overrides: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    X_test_base = X.iloc[n_train:].copy()
    n_total = len(X)
    cat_cols = list(X.select_dtypes(include=["category"]).columns)
    native_keys = model_name == "catboost" and args.catboost_native_keys
    key_feature_names = [("HC_NATIVEKEY_" if native_keys else "HC_TE_") + str(c) for c in key_frame.columns]
    if native_keys:
        cat_cols += key_feature_names
    geometry_source_cols = list(X.select_dtypes(include=[np.number]).columns)
    use_uidy = (uid_meta is not None) and (not args.no_uid)
    auto_uid_prefixes = ["HC_AUID1", "HC_AUID2", "HC_AUID3"] if (use_uidy and args.autolab_uid and "auto_gids" in uid_meta) else []
    extra_names = (list(UID_LABEL_COLS) + [f"{prefix}_{name}" for prefix in auto_uid_prefixes for name in UID_LABEL_COLS]) if use_uidy else []
    geometry_names = ([f"HC_UMAP_{i + 1}" for i in range(args.umap_components)] +
                      (["HC_KLEIN_X", "HC_KLEIN_Y", "HC_KLEIN_Z"] if args.geometry == "umap_klein" else [])) if args.geometry != "off" else []
    knn_names = KNN_FEATURE_NAMES if args.knn_features else []
    oof = np.zeros(n_train, dtype="float32")
    test_pred_logit_fold = []
    fold_scores, best_iters, pseudo_counts = [], [], []
    importance_sum = np.zeros(X.shape[1] + len(key_frame.columns) + len(extra_names) +
                              len(geometry_names) + len(knn_names), dtype="float64")

    if model_name in ("lgbm", "lgbm_et"):
        params = lgb_params(seed, extra_trees=(model_name == "lgbm_et"))
        if model_name == "lgbm" and lgb_overrides:
            params.update(lgb_overrides)
    elif model_name == "catboost":
        if importlib.util.find_spec("catboost") is None:
            raise ImportError("catboost is not installed; run pip install catboost or omit it from --models")
        params = cb_params(seed)
        params.update(FIXED_CB_OVERRIDES)
    elif model_name in ("xgboost", "xgb_rank"):
        if importlib.util.find_spec("xgboost") is None:
            raise ImportError("xgboost is not installed; run pip install xgboost or omit it from --models")
        params = xgb_params(seed)
        if model_name == "xgb_rank":
            params.update(objective="rank:pairwise", min_child_weight=5,
                          lambdarank_pair_method="mean", lambdarank_num_pair_per_sample=8,
                          lambdarank_normalization=False)
        params.update(FIXED_XGB_OVERRIDES)
    elif model_name == "extratrees":
        params = extra_trees_params(seed)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model_hash = sha10({"version": CODE_VERSION, "model": model_name, "params": params,
                        "te_alpha": args.te_alpha, "inner_te_folds": args.inner_te_folds, "folds": args.folds,
                        "lgbm_profile": args.lgbm_profile,
                        "geometry": args.geometry, "umap_fit_sample": args.umap_fit_sample,
                        "umap_components": args.umap_components,
                        "umap_neighbors": args.umap_neighbors, "umap_min_dist": args.umap_min_dist,
                        "umap_metric": args.umap_metric, "umap_target_weight": args.umap_target_weight,
                        "knn_features": args.knn_features, "knn_fit_sample": args.knn_fit_sample,
                        "knn_components": args.knn_components, "knn_inner_folds": args.knn_inner_folds,
                        "lgbm_es_rounds": args.lgbm_es_rounds or (80 if args.quick else 250),
                        "pseudo_label": args.pseudo_label,
                        "pseudo_label_threshold": args.pseudo_label_threshold,
                        "pseudo_label_weight": args.pseudo_label_weight,
                        "flags": FLAG_TAG, "es_holdout": args.es_holdout, "group_cv": args.group_cv,
                        "uidy": use_uidy, "tag": tag,
                        **({"feature_subspace": sha10(SUBSPACE_CONFIG)} if SUBSPACE_CONFIG is not None else {}),
                        **({'domain_features':sha10(DOMAIN_CONFIG)} if DOMAIN_CONFIG else {}),
                        **({"catboost_native_keys": True} if native_keys else {}),
                        **({"refit_outer": True, "refit_iteration_scale": args.refit_iteration_scale}
                           if args.refit_outer else {}),
                        **({"lgbm_monotone": args.lgbm_monotone} if args.lgbm_monotone != "auto" else {})})
    log(f"Model={model_name} seed={seed}; cache key={model_hash}")

    for fold_no, (tr, va) in enumerate(folds, start=0):
        cache_path = fold_cache_path(model_name, seed, fold_no, model_hash)
        cached = load_fold_cache(cache_path, va, len(X) - n_train, len(importance_sum))
        if cached is not None:
            vi, vp, tp, best_iter, auc, imp = cached
            oof[vi] = vp
            test_pred_logit_fold.append(logit(np.clip(tp, 1e-6, 1 - 1e-6)))
            fold_scores.append(auc)
            best_iters.append(best_iter)
            pseudo_counts.append(None if args.pseudo_label else 0)
            importance_sum += imp
            log(f"  fold {fold_no + 1}/{len(folds)} cache hit: AUC={auc:.6f}, iter={best_iter}")
            continue

        t_fold = time.time()
        Xtr = X.iloc[tr].copy()
        Xva = X.iloc[va].copy()
        Xte = X_test_base.copy()
        ktr = key_frame.iloc[tr]
        kva = key_frame.iloc[va]
        kte = key_frame.iloc[n_train:]
        if native_keys:
            for key, feature_name in zip(key_frame.columns, key_feature_names):
                for frame, keys in ((Xtr, ktr), (Xva, kva), (Xte, kte)):
                    frame[feature_name] = normalize_key(keys[key]).to_numpy()
        else:
            Xtr, Xva, Xte = add_crossfit_te(Xtr, Xva, Xte, ktr, kva, kte, y[tr], seed + fold_no * 7919,
                                            args.inner_te_folds, args.te_alpha)
        if use_uidy:
            # Partner outcomes from the outer-training fold only; validation/test labels are never visible.
            lab = np.full(n_total, np.nan, dtype="float64")
            lab[tr] = y[tr]
            arrs = uid_label_features(uid_meta["gid"], uid_meta["time"], lab, float(y[tr].mean()))
            for name in UID_LABEL_COLS:
                a = arrs[name]
                Xtr[name] = a[tr]
                Xva[name] = a[va]
                Xte[name] = a[n_train:]
            if auto_uid_prefixes and "auto_gids" in uid_meta:
                for j, prefix in enumerate(auto_uid_prefixes):
                    auto_arrs = uid_label_features(uid_meta["auto_gids"][:, j], uid_meta["time"], lab, float(y[tr].mean()))
                    for name in UID_LABEL_COLS:
                        feature_name = f"{prefix}_{name}"
                        a = auto_arrs[name]
                        Xtr[feature_name] = a[tr]
                        Xva[feature_name] = a[va]
                        Xte[feature_name] = a[n_train:]

        ytr = y[tr]
        if args.es_holdout > 0:
            # Early stopping on a slice of the training fold, so the validation fold stays untouched
            # and the OOF AUC is not tuned on its own labels (costs es_holdout of the training rows).
            sss = StratifiedShuffleSplit(n_splits=1, test_size=args.es_holdout, random_state=seed + fold_no)
            fit_i, es_i = next(sss.split(np.zeros(len(ytr)), ytr))
            Xfit, yfit, Xes, yes = Xtr.iloc[fit_i], ytr[fit_i], Xtr.iloc[es_i], ytr[es_i]
        else:
            Xfit, yfit, Xes, yes = Xtr, ytr, Xva, y[va]
        if args.geometry != "off":
            Xtr, Xva, Xte, Xfit, Xes = add_fold_geometry(
                Xtr, Xva, Xte, Xfit, Xes, geometry_source_cols, yfit, seed + fold_no * 104729
            )
        if args.knn_features:
            Xfit, Xes, Xva, Xte = add_fold_knn_features(
                Xfit, Xes, Xva, Xte, geometry_source_cols, yfit, seed + fold_no * 130363
            )
        es_rounds = args.lgbm_es_rounds or (80 if args.quick else 250)

        if model_name in ("lgbm", "lgbm_et"):
            fold_params = params.copy()
            if model_name == "lgbm" and (args.lgbm_monotone == "ext" or
                                        (args.lgbm_monotone == "auto" and args.lgbm_profile == "autolab")):
                mono_cols = {"EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "HC_EXT_MEAN", "HC_EXT_MIN", "HC_EXT_MAX"}
                fold_params["monotone_constraints"] = [-1 if c in mono_cols else 0 for c in Xfit.columns]
            model = LGBMClassifier(**fold_params)
            callbacks = [lgb.early_stopping(es_rounds, verbose=False),
                         lgb.log_evaluation(250 if not args.quick else 100)]
            model.fit(Xfit, yfit, eval_set=[(Xes, yes)], eval_metric="auc", callbacks=callbacks,
                      categorical_feature="auto")
            best_iter = int(model.best_iteration_ or params["n_estimators"])
            if args.refit_outer:
                selected_iter = best_iter
                best_iter = max(1, int(round(selected_iter * args.refit_iteration_scale)))
                refit_params = fold_params.copy()
                refit_params["n_estimators"] = best_iter
                del model
                model = LGBMClassifier(**refit_params)
                model.fit(Xtr, ytr, categorical_feature="auto")
                log(f"  outer refit: {len(Xfit):,} -> {len(Xtr):,} rows; selected={selected_iter}, trees={best_iter}")
            if args.pseudo_label:
                # Use only this fold's model-fit partition to label competition test rows.
                # Validation rows never contribute labels, predictions, or early stopping.
                base_test_p = model.predict_proba(Xte, num_iteration=best_iter)[:, 1]
                cutoff = args.pseudo_label_threshold
                pseudo_mask = (base_test_p <= 1.0 - cutoff) | (base_test_p >= cutoff)
                n_pseudo = int(pseudo_mask.sum())
                pseudo_counts.append(n_pseudo)
                if n_pseudo:
                    pseudo_y = (base_test_p[pseudo_mask] >= cutoff).astype("int8")
                    X_aug = pd.concat([Xfit, Xte.loc[pseudo_mask]], ignore_index=True)
                    y_aug = np.concatenate([yfit, pseudo_y])
                    sample_weight = np.concatenate([
                        np.ones(len(yfit), dtype="float32"),
                        np.full(n_pseudo, args.pseudo_label_weight, dtype="float32"),
                    ])
                    pseudo_params = fold_params.copy()
                    pseudo_params["n_estimators"] = best_iter
                    pseudo_model = LGBMClassifier(**pseudo_params)
                    pseudo_model.fit(X_aug, y_aug, sample_weight=sample_weight,
                                     categorical_feature="auto")
                    vp = pseudo_model.predict_proba(Xva, num_iteration=best_iter)[:, 1]
                    tp = pseudo_model.predict_proba(Xte, num_iteration=best_iter)[:, 1]
                    imp = pseudo_model.booster_.feature_importance(importance_type="gain").astype("float64")
                    del X_aug, y_aug, sample_weight, pseudo_model
                else:
                    vp = model.predict_proba(Xva, num_iteration=best_iter)[:, 1]
                    tp = base_test_p
                    imp = model.booster_.feature_importance(importance_type="gain").astype("float64")
                log(f"  pseudo-label rows={n_pseudo:,}/{len(Xte):,} at confidence >= {cutoff:.3f}; weight={args.pseudo_label_weight:g}")
            else:
                pseudo_counts.append(0)
                vp = model.predict_proba(Xva, num_iteration=best_iter)[:, 1]
                tp = model.predict_proba(Xte, num_iteration=best_iter)[:, 1]
                imp = model.booster_.feature_importance(importance_type="gain").astype("float64")
        elif model_name == "catboost":
            from catboost import CatBoostClassifier
            Xfitc = as_catboost_frame(Xfit, cat_cols)
            Xesc = as_catboost_frame(Xes, cat_cols)
            Xvac = as_catboost_frame(Xva, cat_cols)
            Xtec = as_catboost_frame(Xte, cat_cols)
            model = CatBoostClassifier(**params)
            model.fit(Xfitc, yfit, eval_set=(Xesc, yes), cat_features=cat_cols,
                      early_stopping_rounds=300 if not args.quick else 80, use_best_model=True)
            best_iter = int(model.get_best_iteration() + 1)
            if args.refit_outer:
                selected_iter = best_iter
                best_iter = max(1, int(round(selected_iter * args.refit_iteration_scale)))
                refit_params = params.copy()
                refit_params["iterations"] = best_iter
                refit_params.pop("od_type", None)
                refit_params.pop("od_wait", None)
                del model
                model = CatBoostClassifier(**refit_params)
                Xtrc = as_catboost_frame(Xtr, cat_cols)
                model.fit(Xtrc, ytr, cat_features=cat_cols, verbose=False)
                del Xtrc
                log(f"  outer refit: {len(Xfit):,} -> {len(Xtr):,} rows; selected={selected_iter}, trees={best_iter}")
            vp = model.predict_proba(Xvac)[:, 1]
            tp = model.predict_proba(Xtec)[:, 1]
            imp = model.get_feature_importance(type="FeatureImportance").astype("float64")
            del Xfitc, Xesc, Xvac, Xtec
        elif model_name == "extratrees":
            Xfit_et = as_extra_trees_matrix(Xfit)
            Xva_et = as_extra_trees_matrix(Xva)
            Xte_et = as_extra_trees_matrix(Xte)
            # Median imputation is fit on this fold's model-training partition only.
            medians = np.nanmedian(Xfit_et, axis=0)
            medians[~np.isfinite(medians)] = 0.0
            for values in (Xfit_et, Xva_et, Xte_et):
                missing = ~np.isfinite(values)
                if missing.any():
                    values[missing] = np.broadcast_to(medians, values.shape)[missing]
            model = ExtraTreesClassifier(**params)
            model.fit(Xfit_et, yfit)
            best_iter = int(params["n_estimators"])
            vp = model.predict_proba(Xva_et)[:, 1]
            tp = model.predict_proba(Xte_et)[:, 1]
            imp = model.feature_importances_.astype("float64")
            del Xfit_et, Xva_et, Xte_et, medians
        else:
            from xgboost import XGBClassifier, XGBRanker
            from sklearn.linear_model import LogisticRegression
            # Modern XGBoost supports pandas categorical dtypes with enable_categorical=True.
            estimator = XGBRanker if model_name == "xgb_rank" else XGBClassifier
            model = estimator(**params)
            if model_name == "xgb_rank":
                model.fit(Xfit, yfit, group=[len(yfit)], eval_set=[(Xes, yes)],
                          eval_group=[[len(yes)]], verbose=False)
            else:
                model.fit(Xfit, yfit, eval_set=[(Xes, yes)], verbose=False)
            best_iteration_zero_based = getattr(model, "best_iteration", None)
            best_iter = int(best_iteration_zero_based + 1) if best_iteration_zero_based is not None else int(params["n_estimators"])
            if model_name == "xgb_rank":
                # Calibrate only on inner ES rows; outer validation labels are
                # never used. This aligns rank-score scales across OOF folds.
                es_score = model.predict(Xes, iteration_range=(0, best_iter))
                rank_calibrator = LogisticRegression(C=1000.0, max_iter=200)
                rank_calibrator.fit(es_score.reshape(-1, 1), yes)
            if args.refit_outer:
                selected_iter = best_iter
                best_iter = max(1, int(round(selected_iter * args.refit_iteration_scale)))
                refit_params = params.copy()
                refit_params["n_estimators"] = best_iter
                refit_params.pop("early_stopping_rounds", None)
                del model
                model = estimator(**refit_params)
                if model_name == "xgb_rank":
                    model.fit(Xtr, ytr, group=[len(ytr)], verbose=False)
                else:
                    model.fit(Xtr, ytr, verbose=False)
                log(f"  outer refit: {len(Xfit):,} -> {len(Xtr):,} rows; selected={selected_iter}, trees={best_iter}")
            if model_name == "xgb_rank":
                vp = rank_calibrator.predict_proba(model.predict(Xva, iteration_range=(0, best_iter)).reshape(-1, 1))[:, 1]
                tp = rank_calibrator.predict_proba(model.predict(Xte, iteration_range=(0, best_iter)).reshape(-1, 1))[:, 1]
            else:
                vp = model.predict_proba(Xva, iteration_range=(0, best_iter))[:, 1]
                tp = model.predict_proba(Xte, iteration_range=(0, best_iter))[:, 1]
            imp = model.feature_importances_.astype("float64")

        if model_name not in ("lgbm", "lgbm_et"):
            pseudo_counts.append(0)
        auc = float(roc_auc_score(y[va], vp))
        oof[va] = vp.astype("float32")
        test_pred_logit_fold.append(logit(np.clip(tp, 1e-6, 1 - 1e-6)))
        fold_scores.append(auc)
        best_iters.append(best_iter)
        if len(imp) == len(importance_sum):
            importance_sum += imp
        else:
            log(f"  Warning: importance length changed ({len(imp)} vs {len(importance_sum)}); importance table may be incomplete")
        save_fold_cache(cache_path, va, vp, tp, best_iter, auc, imp)
        log(f"  fold {fold_no + 1}/{len(folds)} AUC={auc:.6f}, best_iter={best_iter}, {time.time()-t_fold:.1f}s")
        del Xtr, Xva, Xte, Xfit, Xes, model
        gc.collect()

    # Average fold-level test predictions in logit space; each fold has its own fitted model.
    test_logit = np.mean(np.vstack(test_pred_logit_fold), axis=0)
    test_pred = expit(test_logit).astype("float32")
    summary = {
        "model": model_name, "seed": seed, "fold_auc": [float(x) for x in fold_scores],
        "mean_fold_auc": float(np.mean(fold_scores)), "pooled_oof_auc": float(roc_auc_score(y, oof)),
        "best_iterations": [int(x) for x in best_iters], "model_hash": model_hash,
        "pseudo_label_rows_by_fold": [None if x is None else int(x) for x in pseudo_counts],
    }
    seed_path = CACHE_DIR / f"seed_{model_name}_s{seed}_{DATA_TAG}_{model_hash}.npz"
    np.savez(seed_path, oof=oof, test=test_pred)
    # Importance names follow the column order built in each fold: base features, TE columns, UID label columns.
    if len(importance_sum) == X.shape[1] + len(key_frame.columns) + len(extra_names) + len(geometry_names) + len(knn_names):
        fi_names = list(X.columns) + key_feature_names + extra_names + geometry_names + knn_names
        fi = pd.DataFrame({"feature": fi_names, "gain_or_importance": importance_sum / max(1, len(folds))})
        fi.sort_values("gain_or_importance", ascending=False).to_csv(
            OUT_DIR / f"feature_importance_{model_name}_s{seed}{('_' + tag) if tag else ''}.csv", index=False)
    log(f"{model_name} seed={seed}: pooled OOF AUC={summary['pooled_oof_auc']:.6f}; mean fold AUC={summary['mean_fold_auc']:.6f}")
    return oof, test_pred, summary


# ---------------------------------------------------------------------------
# Data loading

def load_application_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Load exactly train.csv and test.csv. No other CSV is opened."""
    train_path = locate_file("train.csv")
    test_path = locate_file("test.csv")
    if train_path is None or test_path is None:
        raise FileNotFoundError(
            f"Input directory {DATA_DIR} must contain exactly the required files train.csv and test.csv. "
            "The pipeline intentionally does not load auxiliary/history tables or sample_submission.csv."
        )
    log(f"Reading only train={train_path.name}, test={test_path.name}")
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    if TARGET not in train.columns:
        raise ValueError(f"Training file {train_path.name} has no {TARGET} column")
    if ID not in train.columns or ID not in test.columns:
        raise ValueError(f"Both train.csv and test.csv need {ID}")

    y = train[TARGET].astype("int8").to_numpy()
    train_x = train.drop(columns=[TARGET])
    # Keep only feature columns present in both files and retain train.csv column order.
    common = [c for c in train_x.columns if c in test.columns]
    if ID not in common:
        raise ValueError("ID column disappeared during train/test alignment")
    missing_test = [c for c in train_x.columns if c not in test.columns and c != TARGET]
    if missing_test:
        log(f"WARNING: {len(missing_test)} train columns are absent from test and will be excluded; examples={missing_test[:8]}")
    train_x = train_x[common]
    test_x = test[common]
    app_all = pd.concat([train_x, test_x], axis=0, ignore_index=True)
    if app_all[ID].duplicated().any():
        raise ValueError("Duplicate SK_ID_CURR found in combined train/test; investigate input files")
    log(f"Rows: train={len(train_x):,}, test={len(test_x):,}, positives={y.mean():.4%}; only two CSV inputs used")
    del train, test, train_x, test_x
    gc.collect()
    return app_all, y


def make_folds(seed: int, y: np.ndarray, n_train: int, gid: np.ndarray):
    if args.group_cv:
        # All applications of one person land in the same fold. Validation rows then have no labeled
        # partners, so this is a deliberately pessimistic estimate for the UID label features.
        sgk = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        return list(sgk.split(np.zeros(n_train), y, groups=gid[:n_train]))
    return list(StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed).split(np.zeros(n_train), y))


def main():
    global FIXED_LGBM_OVERRIDES, FIXED_CB_OVERRIDES, FIXED_XGB_OVERRIDES
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory does not exist: {DATA_DIR.resolve()}")
    seeds = parse_seeds(args.seeds)
    if args.lgbm_params_json:
        with open(args.lgbm_params_json, "r", encoding="utf-8") as f:
            FIXED_LGBM_OVERRIDES = json.load(f)
        if not isinstance(FIXED_LGBM_OVERRIDES, dict):
            raise ValueError("--lgbm-params-json must contain a JSON object")
    for source, destination in ((args.catboost_params_json, FIXED_CB_OVERRIDES),
                                (args.xgb_params_json, FIXED_XGB_OVERRIDES)):
        if source:
            overrides = json.loads(Path(source).read_text(encoding="utf-8"))
            if not isinstance(overrides, dict):
                raise ValueError("Model parameter overrides must be JSON objects")
            destination.update(overrides)
    if args.folds < 3:
        raise ValueError("Use at least 3 folds; 5 is recommended")
    if args.geometry != "off":
        if args.umap_fit_sample < 3 or args.umap_neighbors < 2 or args.umap_fit_sample <= args.umap_neighbors:
            raise ValueError("UMAP requires --umap-fit-sample > --umap-neighbors >= 2")
        if args.umap_components < 2:
            raise ValueError("UMAP geometry needs at least 2 components")
        if args.geometry == "umap_klein" and args.umap_components != 2:
            raise ValueError("Klein mapping requires --umap-components 2")
        if not 0.0 <= args.umap_target_weight <= 1.0:
            raise ValueError("--umap-target-weight must be between 0 and 1")
        if not 0.0 <= args.umap_min_dist <= 1.0:
            raise ValueError("--umap-min-dist must be between 0 and 1")
        if importlib.util.find_spec("umap") is None:
            raise ImportError("--geometry requires umap-learn; install it with `python -m pip install umap-learn`")
    if args.knn_features:
        if args.knn_fit_sample < 300 or args.knn_components < 2 or args.knn_inner_folds < 2:
            raise ValueError("kNN features require --knn-fit-sample >= 300, --knn-components >= 2, and --knn-inner-folds >= 2")
        if importlib.util.find_spec("pynndescent") is None:
            raise ImportError("--knn-features requires pynndescent; install it with `python -m pip install pynndescent`")
    if args.lgbm_es_rounds is not None and args.lgbm_es_rounds < 1:
        raise ValueError("--lgbm-es-rounds must be positive")
    model_names = [x.strip().lower() for x in args.models.split(",") if x.strip()]
    allowed = {"lgbm", "lgbm_et", "catboost", "xgboost", "xgb_rank", "extratrees"}
    unknown = set(model_names) - allowed
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}; allowed={sorted(allowed)}")
    if not model_names:
        raise ValueError("No models selected")
    acquire_research_gpu_slot(model_names)
    if args.optuna_trials:
        if args.optuna_trials < 1:
            raise ValueError("--optuna-trials must be positive")
        if importlib.util.find_spec("optuna") is None:
            raise ImportError("Optuna is not installed; install it in your Python environment with: pip install optuna")
        if model_names != ["lgbm"] or args.lgbm_profile != "autolab":
            raise ValueError("Optuna mode requires --models lgbm --lgbm-profile autolab")
        if len(seeds) != 1 or args.group_cv or args.es_holdout <= 0:
            raise ValueError("Optuna screening requires one seed, ordinary stratified CV, and --es-holdout > 0")
        if FIXED_LGBM_OVERRIDES:
            raise ValueError("Do not combine Optuna search with --lgbm-params-json")

    app_all, y = load_application_data()
    n_train = len(y)
    n_total = len(app_all)
    ids_all = app_all[ID].to_numpy()
    features_cache = CACHE_DIR / f"features_{DATA_TAG}_{FLAG_TAG}.pkl"
    if args.application_cache:
        features_cache=Path(args.application_cache)
        if not features_cache.exists() or features_cache.name!=f"features_{DATA_TAG}_{FLAG_TAG}.pkl":
            raise ValueError('Explicit application cache must exist and match data fingerprint and feature flags')
    features = te_keys = uid_df = None
    if features_cache.exists() and not args.no_feature_cache:
        log(f"Loading feature cache {features_cache.name}; data fingerprint={DATA_TAG}")
        try:
            features, te_keys, uid_df = pd.read_pickle(features_cache)
            if len(features) != n_total or len(te_keys) != n_total or len(uid_df) != n_total:
                log("Feature cache row count mismatch; rebuilding")
                features = None
        except Exception as exc:
            if args.application_cache:raise ValueError('Explicit application cache is unreadable; refusing to overwrite it') from exc
            log(f"Feature cache unreadable ({exc}); rebuilding")
            features = None
    if features is None and args.application_cache:
        raise ValueError('Explicit application cache does not match input row counts; refusing to overwrite it')
    if features is None:
        log(f"Building train/test-only application features; data fingerprint={DATA_TAG}; flags={FLAG_TAG}")
        features, te_keys, uid_df = build_features(app_all, ids_all, n_train)
        log(f"Writing resumable feature cache to {features_cache} (can be large)")
        pd.to_pickle((features, te_keys, uid_df), features_cache)
    if len(features) != n_total or len(te_keys) != n_total:
        raise RuntimeError(f"Feature row-count mismatch: expected {n_total}, got features={len(features)}, keys={len(te_keys)}")
    if not np.array_equal(features.index.to_numpy(), ids_all):
        raise RuntimeError("Feature join changed application row order; refusing to create misaligned OOF predictions/submission")
    if DOMAIN_CONFIG:
        definitions={};parts=[]
        if args.domain_russian_age!='off':
            from scripts.russian_law_features import build as ru_build
            extra,desc=ru_build(app_all,args.domain_russian_age);parts.append(extra);definitions.update(desc)
        if args.domain_behavior!='off':
            from scripts.application_behavior_features import build as behavior_build
            extra,desc=behavior_build(app_all,args.domain_behavior);parts.append(extra);definitions.update(desc)
        extra=pd.concat(parts,axis=1)
        if len(extra)!=len(features) or not extra.columns.is_unique or set(extra).intersection(features):raise ValueError('Invalid domain feature alignment/columns')
        extra.index=features.index
        features=pd.concat([features,extra],axis=1)
        (OUT_DIR/'domain_feature_manifest.json').write_text(json.dumps(dict(configuration=DOMAIN_CONFIG,definitions=definitions,source='Only suppliedapplicationtrain/test'),indent=2),encoding='utf-8')
        log(f'Added {len(extra.columns)} post-cache input-only domain features: {DOMAIN_CONFIG}')
    if SUBSPACE_CONFIG is not None:
        from scripts.feature_subspaces import apply_subspace
        features,te_keys=apply_subspace(features,te_keys,SUBSPACE_CONFIG)
        (OUT_DIR/'feature_subspace_resolved.json').write_text(json.dumps(dict(spec=SUBSPACE_CONFIG,
            common_uid_label_features=bool(not args.no_uid),base_columns=len(features.columns),key_columns=len(te_keys.columns)),indent=2),encoding='utf-8')
        log(f'Subspace {SUBSPACE_CONFIG.get("name","unnamed")}: {len(features.columns)} base features, {len(te_keys.columns)} keys; fold-safe UID shared')
    del app_all
    gc.collect()
    y = np.asarray(y, dtype=np.int8)
    gid = uid_df["gid"].to_numpy(dtype="int64")
    uid_meta = {"gid": gid, "time": uid_df["time"].to_numpy(dtype="float64")}
    auto_uid_cols = [f"{name}_GID" for name in ("HC_AUID1", "HC_AUID2", "HC_AUID3") if f"{name}_GID" in uid_df.columns]
    if args.autolab_uid and len(auto_uid_cols) == 3:
        uid_meta["auto_gids"] = uid_df[auto_uid_cols].to_numpy(dtype="int64")
    n_multi = int((np.bincount(gid)[gid] >= 2).sum())
    log(f"UID groups: {len(np.unique(gid)):,}; rows in a multi-application group={n_multi:,} ({n_multi / n_total:.2%})")

    if args.optuna_trials:
        import optuna
        seed = seeds[0]
        folds = make_folds(seed, y, n_train, gid)
        log(f"Optuna LGBM search: trials={args.optuna_trials}, seed={seed}, folds={len(folds)}, features={features.shape[1]}")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))

        def objective(trial):
            overrides = {
                "num_leaves": trial.suggest_int("num_leaves", 6, 16),
                "min_child_samples": trial.suggest_int("min_child_samples", 150, 500, step=25),
                "learning_rate": trial.suggest_float("learning_rate", 0.012, 0.035, log=True),
                "subsample": trial.suggest_float("subsample", 0.60, 0.90),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.20, 0.55),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.5, 12.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 8.0),
                "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
            }
            _, _, summary = train_one_model(
                "lgbm", seed, features, te_keys, y, n_train, folds, uid_meta,
                tag=f"optuna_t{trial.number:03d}", lgb_overrides=overrides,
            )
            score = float(summary["pooled_oof_auc"])
            trial.set_user_attr("mean_fold_auc", summary["mean_fold_auc"])
            log(f"Optuna trial {trial.number + 1}/{args.optuna_trials}: pooled OOF AUC={score:.6f}; params={overrides}")
            return score

        study.optimize(objective, n_trials=args.optuna_trials, n_jobs=1)
        best_params_path = OUT_DIR / "optuna_lgbm_best_params.json"
        best_params_path.write_text(json.dumps(study.best_params, indent=2), encoding="utf-8")
        trials_path = OUT_DIR / "optuna_lgbm_trials.csv"
        study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs")).to_csv(trials_path, index=False)
        log(f"Best seed {seed} pooled OOF AUC={study.best_value:.6f}; params={study.best_params}")
        log(f"Saved parameters: {best_params_path}")
        log(f"Saved trial log: {trials_path}")
        return

    if args.leak_check:
        # Permuted labels destroy any real signal; a working pipeline must score ~0.5. Anything clearly
        # above 0.5 means a feature (TE / UID label block) is leaking information about the row's own label.
        y_perm = np.random.default_rng(12345).permutation(y).astype(np.int8)
        folds = list(StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seeds[0]).split(np.zeros(n_train), y_perm))
        model = "lgbm" if "lgbm" in model_names else model_names[0]
        _, _, s = train_one_model(model, seeds[0], features, te_keys, y_perm, n_train, folds, uid_meta, tag="leakcheck")
        auc = s["pooled_oof_auc"]
        # Null standard error of AUC (Hanley-McNeil, AUC=0.5): sqrt((n+1) / (12 * n_pos * n_neg)).
        n_pos = int(y_perm.sum()); n_neg = int(len(y_perm) - n_pos)
        se = float(np.sqrt((n_pos + n_neg + 1) / (12.0 * n_pos * n_neg)))
        z = (auc - 0.5) / se
        verdict = "OK (consistent with 0.5)" if abs(z) < 3.0 else "SUSPICIOUS: leakage likely"
        log(f"LEAK CHECK ({model}, permuted TARGET): pooled OOF AUC={auc:.6f}, null SE={se:.4f}, z={z:+.2f} -> {verdict}")
        (OUT_DIR / "leak_check.json").write_text(json.dumps({"model": model, "auc": auc, "null_se": se, "z": z, "verdict": verdict}, indent=2))
        return

    all_results = []
    model_oofs: dict[str, list[np.ndarray]] = {m: [] for m in model_names}
    model_tests: dict[str, list[np.ndarray]] = {m: [] for m in model_names}
    coverage = None

    for seed in seeds:
        folds = make_folds(seed, y, n_train, gid)
        if coverage is None:
            coverage = uid_coverage(gid, n_train, folds)
            log("UID coverage (share of rows with >=1 labeled partner): " + ", ".join(f"{k}={v:.4f}" for k, v in coverage.items()))
        for model_name in model_names:
            log("=" * 72)
            log(f"Starting {model_name} | seed={seed} | folds={args.folds} | features={features.shape[1]}")
            oof, test_pred, summary = train_one_model(model_name, seed, features, te_keys, y, n_train, folds, uid_meta)
            model_oofs[model_name].append(oof)
            model_tests[model_name].append(test_pred)
            all_results.append(summary)
            np.save(OUT_DIR / f"oof_{model_name}_seed{seed}.npy", oof)
            np.save(OUT_DIR / f"test_{model_name}_seed{seed}.npy", test_pred)

    model_final_oof = {}
    model_final_test = {}
    for model_name in model_names:
        poof = np.mean(np.vstack(model_oofs[model_name]), axis=0).astype("float32")
        ptest = np.mean(np.vstack(model_tests[model_name]), axis=0).astype("float32")
        model_final_oof[model_name] = poof
        model_final_test[model_name] = ptest
        log(f"MODEL SUMMARY {model_name}: OOF AUC={roc_auc_score(y, poof):.6f}; per-seed={[round(roc_auc_score(y, p), 6) for p in model_oofs[model_name]]}")

    # Conservative equal-weight logit blend: no blend weights are fitted on the OOF labels.
    blend_logits = np.column_stack([logit(np.clip(model_final_oof[m], 1e-6, 1 - 1e-6)) for m in model_names])
    test_logits = np.column_stack([logit(np.clip(model_final_test[m], 1e-6, 1 - 1e-6)) for m in model_names])
    oof_blend = expit(blend_logits.mean(axis=1)).astype("float32")
    test_blend = expit(test_logits.mean(axis=1)).astype("float32")
    blend_auc = float(roc_auc_score(y, oof_blend))
    log("=" * 72)
    log(f"EQUAL-WEIGHT LOGIT BLEND pooled OOF AUC={blend_auc:.6f}")
    log("Model AUCs: " + ", ".join(f"{m}={roc_auc_score(y, model_final_oof[m]):.6f}" for m in model_names))
    blend_diag = {}
    if len(model_names) >= 2:
        corr = np.corrcoef(blend_logits.T)
        log("OOF logit correlation:\n" + pd.DataFrame(corr, index=model_names, columns=model_names).round(4).to_string())
        rank_blend = np.mean([rankdata(blend_logits[:, j]) for j in range(blend_logits.shape[1])], axis=0)
        blend_diag["rank_mean_auc"] = float(roc_auc_score(y, rank_blend))
        log(f"Rank-mean blend OOF AUC={blend_diag['rank_mean_auc']:.6f} (same ordering information, different averaging)")
        if len(model_names) >= 3:
            loo = {}
            for j, m in enumerate(model_names):
                keep = [k for k in range(len(model_names)) if k != j]
                loo[m] = float(roc_auc_score(y, blend_logits[:, keep].mean(axis=1)))
            blend_diag["blend_auc_without_model"] = loo
            log("Blend AUC without each model: " + ", ".join(f"-{m}={v:.6f}" for m, v in loo.items()))
    if args.es_holdout <= 0:
        log("NOTE: early stopping used the validation folds, so OOF AUC is slightly optimistic. "
            "Use --es-holdout 0.1 for an unbiased CV estimate (costs ~10% of each training fold).")

    pred_oof = pd.DataFrame({ID: features.index[:n_train].to_numpy(), TARGET: y})
    for m in model_names:
        pred_oof[f"P_{m.upper()}"] = model_final_oof[m]
    pred_oof["P_LOGIT_BLEND"] = oof_blend
    pred_oof.to_csv(OUT_DIR / "oof_predictions.csv", index=False)

    test_ids = features.index[n_train:].to_numpy()
    sub = pd.DataFrame({ID: test_ids, TARGET: test_blend})
    sub.to_csv(OUT_DIR / args.submission_name, index=False)
    log(f"Wrote submission: {OUT_DIR / args.submission_name}")

    result = {
        "code_version": CODE_VERSION,
        "data_fingerprint": DATA_TAG,
        "feature_flags": {
            **({'domain_features':DOMAIN_CONFIG} if DOMAIN_CONFIG else {}),
            "uid": not args.no_uid,
            "ext_impute": not args.no_ext_impute,
            "product_x": not args.no_product_x,
            "autolab_age_end": args.autolab_age_end,
            "autolab_days_diffs": args.autolab_days_diffs,
            "client_dates": args.client_dates,
            "id_features": args.id_features,
            "id_te_only": args.id_te_only,
            "autolab_rank_extra": args.autolab_rank_extra,
            "autolab_tariff": args.autolab_tariff,
            "autolab_value_counts": args.autolab_value_counts,
            "autolab_uid": args.autolab_uid,
            "ratio_income_stress": args.ratio_income_stress,
            "peer_context": args.peer_context,
            "finance_extra_keys": args.finance_extra_keys,
            "equation_disposable_income": (args.equation_disposable_component or
                                           ("all" if args.equation_disposable_income else False)),
            "client_id_renewal": args.client_id_renewal,
            "ext_copula": args.ext_copula,
        },
        "cv": {"folds": int(args.folds), "group_cv": bool(args.group_cv), "es_holdout": float(args.es_holdout),
               "refit_outer": args.refit_outer, "refit_iteration_scale": args.refit_iteration_scale},
        "parameter_overrides": {"lgbm": FIXED_LGBM_OVERRIDES, "catboost": FIXED_CB_OVERRIDES,
                                "xgboost": FIXED_XGB_OVERRIDES},
        "rows_train": int(n_train),
        "rows_test": int(len(test_blend)),
        "feature_count": int(features.shape[1]),
        "categorical_feature_count": int(len(features.select_dtypes(include=['category']).columns)),
        "models": model_names,
        "lgbm_profile": args.lgbm_profile,
        "geometry": {"mode": args.geometry, "umap_fit_sample": int(args.umap_fit_sample),
                     "components": int(args.umap_components),
                     "neighbors": int(args.umap_neighbors), "min_dist": float(args.umap_min_dist),
                     "metric": args.umap_metric, "target_weight": float(args.umap_target_weight)},
        "knn_features": {"enabled": bool(args.knn_features), "reference_sample": int(args.knn_fit_sample),
                         "pca_components": int(args.knn_components), "inner_folds": int(args.knn_inner_folds),
                         "columns": KNN_FEATURE_NAMES if args.knn_features else []},
        "seeds": seeds,
        "uid_rows_in_multi_application_group": n_multi,
        "uid_coverage": coverage,
        "target_encoding": {"keys": list(te_keys.columns), "inner_folds": args.inner_te_folds, "smoothing_alpha": args.te_alpha},
        "catboost_native_keys": args.catboost_native_keys,
        "model_runs": all_results,
        "model_pooled_oof_auc_after_seed_average": {m: float(roc_auc_score(y, model_final_oof[m])) for m in model_names},
        "equal_weight_logit_blend_pooled_oof_auc": blend_auc,
        "blend_diagnostics": blend_diag,
        "elapsed_seconds": float(time.time() - T0),
        "input_policy": "Only train.csv and test.csv are read; all other CSV files are ignored.",
        "note": "OOF AUC is a CV estimate, not a guarantee of public/private leaderboard score.",
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"Saved run summary: {OUT_DIR / 'run_summary.json'}")
    log(f"Done in {(time.time() - T0)/60:.1f} minutes")


if __name__ == "__main__":
    main()

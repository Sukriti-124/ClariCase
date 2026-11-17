"""Trace one complaint through every stage of the ClariCase prediction pipeline.

Read-only: loads the served artifacts, runs the complaint in-process and
prints what each stage computed and how the team and issue were chosen.

Run from the repo root:
    python scripts/trace_prediction.py "my card was charged twice"
    python -m scripts.trace_prediction --json < complaint.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap: make `src` (and `data.preparation`, referenced by pickled
# vectorizers) importable before any project import.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Any, Callable  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from src.ensemble.probs import TEAMS, reorder  # noqa: E402
from src.explain.adapters import record_from_trace  # noqa: E402
from src.explain.template import summary_facts  # noqa: E402
from src.utils.config import ROOT, TEAM_MAPPING  # noqa: E402
from src.utils.text import clean_text  # noqa: E402

IM1_PATH = ROOT / "docs" / "metrics" / "improved_model_1" / "model.joblib"


class TraceFatal(RuntimeError):
    """A required stage (Improved Model 1) failed; the trace cannot continue."""


# ---- stage runner -----------------------------------------------------------

def run_stage(name: str, fn: Callable[..., dict], *args,
              required: bool = False, **kw) -> dict:
    """Call one stage function and normalise its outcome into a Stage_Result.

    ImportError / FileNotFoundError -> skipped; any other exception -> error.
    A required stage that does not end `ok` raises TraceFatal.
    """
    try:
        result = fn(*args, **kw)
        if not isinstance(result, dict):
            result = {"value": result}
        result.setdefault("status", "ok")
        result.setdefault("reason", None)
    except (ImportError, FileNotFoundError) as e:
        result = {"status": "skipped", "reason": f"{type(e).__name__}: {e}"}
    except Exception as e:  # noqa: BLE001 - every stage failure is reported
        result = {"status": "error", "reason": f"{type(e).__name__}: {e}"}
    result["stage"] = name
    if required and result["status"] != "ok":
        raise TraceFatal(result["reason"] or f"stage {name} did not complete")
    return result


# ---- stage 1: input ---------------------------------------------------------

def trace_input(text: str) -> dict:
    return {"status": "ok", "reason": None, "raw": text,
            "n_words": len(text.split()), "cleaned": clean_text(text)}


def team_names() -> dict[str, str]:
    """team_id -> team_name from TEAM_MAPPING; {} when the file is missing."""
    if not Path(TEAM_MAPPING).exists():
        return {}
    import pandas as pd
    df = pd.read_csv(TEAM_MAPPING)
    return {str(k): str(v) for k, v in zip(df["team_id"], df["team_name"])}


# ---- stage 2: Improved Model 1 ----------------------------------------------

def top_ngram_contributions(vectorizer, coef_row, X_row, top_n: int) -> list[dict]:
    """N-grams of X_row with the largest tfidf * coef for one class.

    Works for a TfidfVectorizer and a FeatureUnion (names then carry the
    transformer prefix, e.g. "word__" / "char__"). coef_row may be sparse or
    dense.
    """
    names = np.asarray(vectorizer.get_feature_names_out())
    if sp.issparse(coef_row):
        coef = np.asarray(coef_row.toarray()).ravel()
    else:
        coef = np.asarray(coef_row).ravel()
    row = sp.csr_matrix(X_row)
    if row.shape[0] != 1:
        row = row[0]
    row.sum_duplicates()
    nz = row.data != 0
    idx = row.indices[nz]
    vals = row.data[nz].astype(np.float64)
    contrib = vals * coef[idx].astype(np.float64)
    order = np.argsort(-contrib, kind="stable")[:max(int(top_n), 0)]
    return [{"ngram": str(names[idx[o]]), "tfidf": float(vals[o]),
             "coef": float(coef[idx[o]]), "contribution": float(contrib[o])}
            for o in order]


def _coef_row_for(clf, cls) -> Any:
    """Coefficient row that scores `cls` (binary LinearSVC has one row)."""
    classes = [str(c) for c in clf.classes_]
    coef = clf.coef_
    if coef.shape[0] == 1 and len(classes) == 2:
        return coef[0] if str(cls) == classes[1] else -coef[0]
    return coef[classes.index(str(cls))]


def trace_improved_model_1(text: str, bundle: dict, names: dict, top_k: int,
                           top_ngrams: int) -> dict:
    vec = bundle["vectorizer"]
    hsvm = bundle["model"]
    team_classes = [str(c) for c in bundle["team_classes"]]
    global_thr = bundle.get("global_threshold")

    X = vec.transform([text])
    raw_classes = [str(c) for c in hsvm.team_clf.classes_]
    raw = np.atleast_1d(np.asarray(hsvm.team_clf.decision_function(X))[0])
    if raw.shape[0] == 1 and len(raw_classes) == 2:  # binary: one margin
        raw = np.array([-raw[0], raw[0]])
    raw_by_team = {c: float(s) for c, s in zip(raw_classes, raw)}

    proba = np.asarray(hsvm.team_proba(X)[0], dtype=np.float64)
    order = np.argsort(proba)[::-1]
    chosen = team_classes[order[0]]
    conf = float(proba[order[0]])

    teams = [{"team_id": t, "team_name": names.get(t, ""),
              "raw_score": raw_by_team.get(t), "prob": float(proba[j])}
             for j, t in ((j, team_classes[j]) for j in order)]

    ngrams = top_ngram_contributions(
        vec, _coef_row_for(hsvm.team_clf, chosen), X, top_ngrams)

    route = ("auto" if global_thr is not None and conf >= float(global_thr)
             else "review")
    team_probs_vec = reorder(proba[None, :], team_classes, TEAMS)[0]

    return {
        "status": "ok", "reason": None,
        "model_name": bundle.get("model_name", "improved_model_1"),
        "nnz": int(X.nnz),
        "teams": teams,
        "top_teams": teams[:max(int(top_k), 1)],
        "top_ngrams": ngrams,
        "team": chosen,
        "team_name": names.get(chosen, ""),
        "confidence": conf,
        "global_threshold": None if global_thr is None else float(global_thr),
        "route": route,
        "team_probs": {t: float(p) for t, p in zip(TEAMS, team_probs_vec)},
        "X": X,  # kept for later stages, dropped from JSON
    }


# ---- stage 3: issue head ----------------------------------------------------

def trace_issue_head(X, bundle: dict, team: str, top_k: int) -> dict:
    from src.improved_model_1.model import _ConstantIssue

    hsvm = bundle["model"]
    issue_labels = bundle.get("issue_labels", {}) or {}
    clf = hsvm.issue_clfs_.get(team)
    if clf is None:
        return {"status": "ok", "reason": None, "team": team,
                "kind": "none", "issues": []}
    if isinstance(clf, _ConstantIssue):
        kind = "constant"
    elif team in (hsvm.calibrated_issue_ or {}):
        kind = "calibrated"
    else:
        kind = "softmax(decision_function)"
    top = hsvm.predict_issue_topk(X, teams=np.array([team]), k=top_k)[0]
    issues = [{"issue_id": str(iid), "issue_label": issue_labels.get(iid, ""),
               "prob": float(p)} for iid, p in top]
    return {"status": "ok", "reason": None, "team": team, "kind": kind,
            "issues": issues}


# ---- stage 4: Baseline Model 2 ----------------------------------------------

BM2_PATH = ROOT / "docs" / "metrics" / "baseline_model_2" / "model.joblib"
IM2_PATH = ROOT / "docs" / "metrics" / "improved_model_2" / "model.joblib"
KNN_INDEX_PATH = ROOT / "data" / "cache" / "knn_embed" / "index.npz"
STACK_PATH = ROOT / "docs" / "metrics" / "ensemble_stack" / "stack.joblib"


def _probs_dict(P, teams=TEAMS) -> dict[str, float]:
    return {str(t): float(p) for t, p in zip(teams, np.asarray(P).ravel())}


def _top_of(P, teams=TEAMS) -> tuple[str, float]:
    P = np.asarray(P, dtype=np.float64).ravel()
    j = int(P.argmax())
    return str(teams[j]), float(P[j])


def trace_baseline_model_2(text: str, path: Path = BM2_PATH) -> dict:
    """Team probabilities of the baseline text pipeline, in TEAMS order."""
    import joblib

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    est = joblib.load(path)
    P = reorder(est.predict_proba([text]), est.classes_, TEAMS)[0]
    team, conf = _top_of(P)
    return {"status": "ok", "reason": None, "model_name": "baseline_model_2",
            "team": team, "confidence": conf, "team_probs": _probs_dict(P)}


# ---- stage 5: embedding models (Improved Model 2 + kNN vote) ----------------

def trace_improved_model_2(text: str, path: Path = IM2_PATH,
                           skip_embed: bool = False) -> dict:
    """Team probabilities from the sentence-encoder model. Also returns the
    query embedding (kept out of JSON) so the kNN stage can reuse it."""
    if skip_embed:
        return {"status": "skipped", "reason": "--skip-embed set"}
    import os

    import joblib

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    # In-process mode makes no network calls: only a locally cached encoder
    # is used; a missing cache surfaces as an error for this stage.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from src.improved_model_2.predict import encoder_for  # ImportError -> skipped

    raw = joblib.load(path)
    enc = encoder_for(raw, None)
    q = np.asarray(enc.encode([text]), dtype=np.float32)
    P = reorder(raw["classifier"].predict_proba(q), raw["team_classes"],
                TEAMS)[0]
    team, conf = _top_of(P)
    return {"status": "ok", "reason": None,
            "model_name": raw.get("model_name", "improved_model_2"),
            "encoder_name": str(raw["encoder_name"]),
            "max_seq_length": int(raw["max_seq_length"]),
            "embedding_dim": int(q.shape[1]),
            "team": team, "confidence": conf, "team_probs": _probs_dict(P),
            "embedding": q[0]}


def trace_knn(embedding, index_path: Path = KNN_INDEX_PATH,
              encoder_name: str | None = None,
              max_seq_length: int | None = None) -> dict:
    """Similarity-weighted team vote among the nearest training complaints,
    computed exactly as embed-api's KnnIndex.team_vote."""
    index_path = Path(index_path)
    if embedding is None:
        return {"status": "skipped",
                "reason": "needs improved_model_2 embedding"}
    if not index_path.exists():
        return {"status": "skipped", "reason": f"{index_path} not found"}
    from src.ensemble.knn import topk_neighbours, vote

    z = np.load(index_path, allow_pickle=False)
    teams = [str(t) for t in z["teams"]]
    if teams != list(TEAMS):
        raise RuntimeError(f"index team order {teams} != {list(TEAMS)}")
    idx_encoder = str(z["encoder_name"])
    idx_len = int(z["max_seq_length"])
    if encoder_name is not None and max_seq_length is not None and \
            (str(encoder_name), int(max_seq_length)) != (idx_encoder, idx_len):
        raise RuntimeError(
            f"knn index was built with {idx_encoder} (len {idx_len}) but the "
            f"model uses {encoder_name} (len {max_seq_length})")
    X = np.ascontiguousarray(z["embeddings"], dtype=np.float32)
    team_idx = z["team_idx"].astype(np.int64)
    k = int(z["k"])
    q = np.asarray(embedding, dtype=np.float32).ravel()
    sims, idx = topk_neighbours(X, q[None, :], k, backend="numpy")
    P = vote(idx, sims, team_idx, len(TEAMS))[0]
    team, conf = _top_of(P)
    neighbours = [{"rank": r + 1, "similarity": float(sims[0, r]),
                   "team_id": TEAMS[int(team_idx[idx[0, r]])]}
                  for r in range(min(5, idx.shape[1]))]
    return {"status": "ok", "reason": None, "model_name": "knn_embed",
            "k": k, "n_index": int(len(X)), "team": team, "confidence": conf,
            "team_probs": _probs_dict(P), "nearest": neighbours}


def trace_embed(text: str, im2_path: Path = IM2_PATH,
                knn_path: Path = KNN_INDEX_PATH,
                skip_embed: bool = False) -> dict:
    """Stage 5: Improved Model 2, then the kNN vote that reuses its embedding."""
    im2 = run_stage("improved_model_2", trace_improved_model_2, text,
                    im2_path, skip_embed=skip_embed)
    if skip_embed:
        knn = {"status": "skipped", "reason": "--skip-embed set",
               "stage": "knn_embed"}
    elif im2["status"] != "ok":
        knn = {"status": "skipped",
               "reason": "needs improved_model_2 embedding",
               "stage": "knn_embed"}
    else:
        knn = run_stage("knn_embed", trace_knn, im2.get("embedding"),
                        knn_path, im2.get("encoder_name"),
                        im2.get("max_seq_length"))
    return {"improved_model_2": im2, "knn_embed": knn}


# ---- stage 6: ensemble (mirrors StackBundle.decide / fallback) --------------

def model_prob_vectors(stages: dict[str, dict], teams) -> dict[str, np.ndarray]:
    """Probability vectors in `teams` order for every stage that ended ok and
    carries `team_probs`."""
    out = {}
    for name, res in stages.items():
        if res and res.get("status") == "ok" and res.get("team_probs"):
            tp = res["team_probs"]
            out[name] = np.array([float(tp.get(str(t), 0.0)) for t in teams],
                                 dtype=np.float64)
    return out


def _routed(score: float, team: str, base, team_thresholds) -> bool:
    from src.ensemble.thresholds import route_mask
    return bool(route_mask(np.array([score]), np.array([team]), base,
                           team_thresholds or None)[0])


def _decide(raw: dict, model_probs: dict[str, np.ndarray],
            n_words: int) -> dict:
    from src.ensemble import gate as G
    from src.ensemble.stack import stack_features, stacker_proba

    teams = [str(t) for t in raw["teams"]]
    models = list(raw["models"])
    scenario_kind = str(raw["scenario"])
    uses_stacker = scenario_kind.startswith("stack")
    if uses_stacker and raw.get("stacker") is None:
        raise RuntimeError("bundle scenario needs a stacker but has none")
    mats = [np.asarray(model_probs[m], dtype=np.float64)[None, :]
            for m in models]
    model_teams = {m: teams[int(P.argmax())] for m, P in zip(models, mats)}

    if uses_stacker:
        P = stacker_proba(raw["stacker"], stack_features(mats), len(teams))[0]
        gate_mats = [P[None, :]] + mats
        stack_probs = _probs_dict(P, teams)
    else:
        P = mats[0][0]
        gate_mats = mats
        stack_probs = None

    # Label shift: a single in-process request has seen one row, below
    # RollingPrior.min_rows, so `adjust` would return P unchanged.
    source_prior = raw.get("source_prior")
    prior = (_probs_dict(source_prior, teams) if source_prior is not None
             else None)

    j = int(P.argmax())
    team = teams[j]
    Xg = G.prob_features([j], gate_mats, [n_words], len(teams))
    names = raw.get("gate_feature_names") or G.prob_feature_names(
        (["stack"] if uses_stacker else []) + models, teams)
    names = [str(n) for n in names]
    gate_features = [{"name": n, "value": float(v)}
                     for n, v in zip(names, Xg[0])]
    score = float(raw["gate"].predict_proba(Xg)[0, 1])

    base = raw["base_threshold"]
    team_thr = raw.get("team_thresholds") or {}
    auto = _routed(score, team, base, team_thr)
    info = {str(k): v for k, v in team_thr.items()}.get(team)
    threshold = base if info is None else info["threshold"]

    if auto:
        scenario = "auto"
    elif all(t == team for t in model_teams.values()):
        scenario = "assisted"
    else:
        scenario = "contested"
    order = np.argsort(-P, kind="stable")[:3]
    return {
        "path": "decide",
        "stacker_used": uses_stacker,
        "stack_probs": stack_probs,
        "prior_active": False,
        "source_prior": prior,
        "label_shift_note": ("label-shift correction inactive: a single "
                             "in-process request is below the rolling prior's "
                             "minimum row count, so P is used unchanged"),
        "gate_features": gate_features,
        "gate_feature_names_match": len(names) == Xg.shape[1],
        "n_gate_features": int(Xg.shape[1]),
        "team": team,
        "score": score,
        "threshold": None if threshold is None else float(threshold),
        "base_threshold": None if base is None else float(base),
        "auto_route": auto,
        "scenario": scenario,
        "model_teams": model_teams,
        "candidates": [teams[int(k)] for k in order],
    }


def _fallback(raw: dict, primary_vec, missing: list[str]) -> dict:
    teams = [str(t) for t in raw["teams"]]
    primary = raw.get("primary_name", raw["models"][0])
    policy = raw.get("primary_fallback_policy") or {
        "base": raw.get("primary_fallback_threshold"), "teams": None}
    P = np.asarray(primary_vec, dtype=np.float64).ravel()
    j = int(P.argmax())
    team, score = teams[j], float(P[j])
    base = policy["base"]
    auto = base is not None and _routed(score, team, base, policy.get("teams"))
    team_info = {str(k): v for k, v in (policy.get("teams") or {}).items()}
    team_thr = team_info[team]["threshold"] if team in team_info else base
    return {
        "path": "fallback",
        "stacker_used": False,
        "stack_probs": None,
        "prior_active": False,
        "team": team,
        "score": score,
        "threshold": None if base is None else float(base),
        "team_threshold": None if team_thr is None else float(team_thr),
        "auto_route": bool(auto),
        "scenario": "degraded",
        "model_teams": {primary: team},
        "reason_degraded": f"missing models: {missing}",
    }


def trace_ensemble(stack_raw: dict, model_probs: dict[str, np.ndarray],
                   n_words: int) -> dict:
    """Ensemble decision for one complaint. `model_probs` maps model name to
    a probability vector in stack_raw["teams"] order, only for models that
    ran."""
    required = list(stack_raw["models"])
    primary = stack_raw.get("primary_name", required[0])
    available = [m for m in required if m in model_probs]
    missing = [m for m in required if m not in model_probs]
    head = {"model_name": stack_raw.get("model_name"),
            "bundle_scenario": stack_raw.get("scenario"),
            "required_models": required, "available_models": available,
            "missing": missing, "primary": primary}
    if missing:
        if primary not in model_probs:
            return {"status": "error",
                    "reason": f"primary model {primary} has no probabilities",
                    **head}
        body = _fallback(stack_raw, model_probs[primary], missing)
    else:
        body = _decide(stack_raw, model_probs, n_words)
    return {"status": "ok", "reason": None, **head, **body}


def trace_stack(path: Path, stages: dict[str, dict], n_words: int) -> dict:
    """Load the stack bundle and run trace_ensemble on the model stages that
    ended ok (keyed by model name, each with `team_probs`). A missing file
    raises FileNotFoundError, which run_stage records as skipped."""
    import joblib

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    raw = joblib.load(path)
    return trace_ensemble(raw, model_prob_vectors(stages, raw["teams"]),
                          n_words)


# ---- stage 7: issue resolution ----------------------------------------------

def resolve_issues(X, bundle: dict, primary: dict, ensemble: dict | None,
                   top_k: int, issue_head: dict | None = None) -> dict:
    """Issues for the final team: reuse the primary issue ranking when the
    ensemble kept the primary team, else re-rank with the primary issue head
    for the ensemble team (as ensemble-api asks svm-api /issues).
    `issue_head` is the stage-3 result; when absent the primary ranking is
    recomputed."""
    primary_team = primary["team"]
    final_team = primary_team
    if ensemble and ensemble.get("status") == "ok" and ensemble.get("team"):
        final_team = str(ensemble["team"])
    if final_team == primary_team and issue_head and \
            issue_head.get("status") == "ok":
        return {"status": "ok", "reason": None, "team": final_team,
                "source": "improved_model_1", "kind": issue_head.get("kind"),
                "issues": issue_head.get("issues", [])}
    head = trace_issue_head(X, bundle, final_team, top_k)
    source = ("re-ranked for ensemble team" if final_team != primary_team
              else "improved_model_1")
    return {"status": "ok", "reason": None, "team": final_team,
            "source": source, "kind": head.get("kind"),
            "issues": head.get("issues", [])}


# ---- stage 8: summary -------------------------------------------------------

def _fmt_p(x) -> str:
    return "n/a" if x is None else f"{float(x):.3f}"


def summarize(results: dict, names: dict) -> dict:
    st = results["stages"]
    im1 = st["improved_model_1"]
    ens = st.get("ensemble") or {}
    issues = st.get("issues") or {}
    ens_ok = ens.get("status") == "ok"

    # Team, issues, route and scenario come from the summary logic shared
    # with the Template_Explainer (Req 14.3).
    facts = summary_facts(record_from_trace(results, names))
    team = facts.team
    auto = facts.auto_route
    route = facts.route
    scenario = facts.scenario
    confidence = ens.get("score") if ens_ok else im1["confidence"]
    trace_issues = issues.get("issues") or []
    by_id = {str(i["issue_id"]): i for i in trace_issues}
    top_issues = [by_id[i["issue_id"]] for i in facts.issues]
    # Outcome holds at most three issues; any extra (--top-k > 3) follow.
    top_issues += trace_issues[len(top_issues):]

    why = [f"Improved Model 1 picked {im1['team']} "
           f"({im1.get('team_name') or 'unnamed'}) with calibrated "
           f"confidence {_fmt_p(im1['confidence'])}"]
    pos = [g["ngram"] for g in im1.get("top_ngrams", [])
           if g["contribution"] > 0][:3]
    if pos:
        why.append("driven mostly by the n-grams " +
                   ", ".join(f"'{g}'" for g in pos))
    others = []
    for m in ("baseline_model_2", "improved_model_2", "knn_embed"):
        r = st.get(m) or {}
        if r.get("status") == "ok":
            others.append(f"{m} {'agreed' if r['team'] == im1['team'] else 'chose ' + r['team']}")
    text = "; ".join(why) + "."
    if others:
        text += " Other models: " + ", ".join(others) + "."
    if ens_ok and ens.get("path") == "decide":
        cmp = ">=" if ens.get("auto_route") else "<"
        text += (f" The ensemble chose {team} with gate score "
                 f"{_fmt_p(ens.get('score'))} {cmp} threshold "
                 f"{_fmt_p(ens.get('threshold'))}, so scenario is "
                 f"'{scenario}'.")
    elif ens_ok and ens.get("path") == "fallback":
        text += (f" The ensemble fell back to the primary alone "
                 f"({ens.get('reason_degraded')}); primary confidence "
                 f"{_fmt_p(ens.get('score'))} vs fallback threshold "
                 f"{_fmt_p(ens.get('threshold'))} gives route '{route}'.")
    else:
        text += (f" The ensemble stage was {ens.get('status', 'not run')} "
                 f"({ens.get('reason')}), so the primary decision is used: "
                 f"confidence {_fmt_p(im1['confidence'])} vs global threshold "
                 f"{_fmt_p(im1.get('global_threshold'))} gives route "
                 f"'{route}'.")
    if issues.get("source") == "re-ranked for ensemble team":
        text += (f" Issues were re-ranked for {team} because the ensemble "
                 f"changed the team.")
    top = (issues.get("issues") or [])[:1]
    if top:
        lab = top[0]["issue_label"] or top[0]["issue_id"]
        text += f" Top issue: {lab} ({_fmt_p(top[0]['prob'])})."

    return {"status": "ok", "reason": None, "team": team,
            "team_name": names.get(team, ""), "confidence": confidence,
            "issues": top_issues, "route": route,
            "auto_route": auto, "scenario": scenario, "why": text}


# ---- pipeline ---------------------------------------------------------------

def _load_im1(path: Path | None = None) -> dict:
    import joblib

    path = Path(IM1_PATH if path is None else path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return joblib.load(path)


def run_trace(text: str, args) -> dict:
    """Stages 1-8 in order. Raises TraceFatal when Improved Model 1 fails."""
    top_k = int(getattr(args, "top_k", 3))
    top_ngrams = int(getattr(args, "top_ngrams", 10))
    skip_embed = bool(getattr(args, "skip_embed", False))
    names = team_names()
    stages: dict[str, dict] = {}

    stages["input"] = run_stage("input", trace_input, text)
    holder: dict = {}

    def _im1() -> dict:
        holder["bundle"] = _load_im1()
        return trace_improved_model_1(text, holder["bundle"], names, top_k,
                                      top_ngrams)

    im1 = run_stage("improved_model_1", _im1, required=True)
    stages["improved_model_1"] = im1
    bundle = holder["bundle"]
    X = im1["X"]

    stages["issue_head"] = run_stage("issue_head", trace_issue_head, X,
                                     bundle, im1["team"], top_k)
    stages["baseline_model_2"] = run_stage(
        "baseline_model_2", trace_baseline_model_2, text, BM2_PATH)
    stages.update(trace_embed(text, IM2_PATH, KNN_INDEX_PATH,
                              skip_embed=skip_embed))
    model_stages = {"improved_model_1": im1,
                    "baseline_model_2": stages["baseline_model_2"],
                    "improved_model_2": stages["improved_model_2"],
                    "knn_embed": stages["knn_embed"]}
    stages["ensemble"] = run_stage("ensemble", trace_stack, STACK_PATH,
                                   model_stages, stages["input"]["n_words"])
    stages["issues"] = run_stage("issues", resolve_issues, X, bundle, im1,
                                 stages["ensemble"], top_k,
                                 issue_head=stages["issue_head"])
    results = {"mode": "in-process", "text": text, "stages": stages}
    stages["summary"] = run_stage("summary", summarize, results, names)
    return results


# ---- output -----------------------------------------------------------------

_DROP_KEYS = {"X", "embedding"}


def to_jsonable(obj) -> Any:
    """Plain-Python copy of obj: numpy scalars/arrays converted, sparse
    matrices and the X / embedding keys dropped."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()
                if k not in _DROP_KEYS and not sp.issparse(v)}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if sp.issparse(obj):
        return None
    return str(obj)


_STAGE_TITLES = [
    ("input", "Input"),
    ("improved_model_1", "Improved Model 1"),
    ("issue_head", "Issue head"),
    ("baseline_model_2", "Baseline Model 2"),
    ("improved_model_2", "Embedding models: Improved Model 2"),
    ("knn_embed", "Embedding models: kNN vote"),
    ("ensemble", "Ensemble"),
    ("issues", "Issue resolution"),
    ("summary", "Summary"),
]
_STAGE_NUM = {"input": "1", "improved_model_1": "2", "issue_head": "3",
              "baseline_model_2": "4", "improved_model_2": "5a",
              "knn_embed": "5b", "ensemble": "6", "issues": "7",
              "summary": "8"}


def _row(name: str, value, out) -> None:
    print(f"  {name:<22} {value}", file=out)


def _print_probs(probs: dict | None, names: dict, out, indent="    ") -> None:
    if not probs:
        print(f"{indent}(none)", file=out)
        return
    for t, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        print(f"{indent}{t:<5} {names.get(t, ''):<38} {p:.4f}", file=out)


def _print_issues(issues: list, out) -> None:
    if not issues:
        print("    (no issues)", file=out)
    for i in issues:
        print(f"    {i['issue_id']:<8} {i['prob']:.4f}  {i['issue_label']}",
              file=out)


def print_text(results: dict, out=None) -> None:
    out = out or sys.stdout
    st = results["stages"]
    names = team_names()
    for key, title in _STAGE_TITLES:
        r = st.get(key)
        if r is None:
            continue
        print(f"\n== [{_STAGE_NUM[key]}] {title} ==", file=out)
        if r.get("status") == "skipped":
            print(f"  SKIPPED: {r.get('reason')}", file=out)
            continue
        if r.get("status") == "error":
            print(f"  ERROR: {r.get('reason')}", file=out)
            continue
        if key == "input":
            _row("raw", repr(r["raw"]), out)
            _row("words", r["n_words"], out)
            _row("cleaned", repr(r["cleaned"]), out)
        elif key == "improved_model_1":
            _row("non-zero features", r["nnz"], out)
            print("  teams (team_id  team_name  raw_score  prob):", file=out)
            for t in r["teams"]:
                raw = "n/a" if t["raw_score"] is None else f"{t['raw_score']:+.4f}"
                print(f"    {t['team_id']:<5} {t['team_name']:<38} "
                      f"{raw:>9}  {t['prob']:.4f}", file=out)
            print(f"  top n-grams for {r['team']} (tfidf x coef = contribution):",
                  file=out)
            for g in r["top_ngrams"]:
                print(f"    {g['ngram']!r:<32} {g['tfidf']:.4f} x "
                      f"{g['coef']:+.4f} = {g['contribution']:+.4f}", file=out)
            _row("chosen team", f"{r['team']} {r['team_name']}", out)
            _row("confidence", f"{r['confidence']:.4f}", out)
            _row("global_threshold", _fmt_p(r["global_threshold"]), out)
            _row("route", r["route"], out)
        elif key == "issue_head":
            _row("team", r["team"], out)
            _row("head kind", r["kind"], out)
            _print_issues(r["issues"], out)
        elif key in ("baseline_model_2", "improved_model_2", "knn_embed"):
            if key == "improved_model_2":
                _row("encoder", f"{r['encoder_name']} (len {r['max_seq_length']}, "
                     f"dim {r['embedding_dim']})", out)
            if key == "knn_embed":
                _row("k / index size", f"{r['k']} / {r['n_index']}", out)
            _row("top team", f"{r['team']} ({r['confidence']:.4f})", out)
            print("  team probs:", file=out)
            _print_probs(r["team_probs"], names, out)
            for nb in r.get("nearest", []):
                print(f"    neighbour {nb['rank']}: {nb['team_id']} "
                      f"sim {nb['similarity']:.4f}", file=out)
        elif key == "ensemble":
            _row("bundle", f"{r.get('model_name')} "
                 f"(scenario {r.get('bundle_scenario')})", out)
            _row("required models", ", ".join(r["required_models"]), out)
            _row("with probabilities", ", ".join(r["available_models"]), out)
            _row("path", r["path"], out)
            if r["path"] == "fallback":
                _row("reason", r["reason_degraded"], out)
            if r.get("stack_probs"):
                print("  stacker probs:", file=out)
                _print_probs(r["stack_probs"], names, out)
            if r["path"] == "decide":
                print(f"  {r['label_shift_note']}", file=out)
                if r.get("source_prior"):
                    print("  source_prior:", file=out)
                    _print_probs(r["source_prior"], names, out)
                print("  gate features:", file=out)
                for f in r["gate_features"]:
                    print(f"    {f['name']:<36} {f['value']:.4f}", file=out)
            _row("team", f"{r['team']} {names.get(r['team'], '')}", out)
            _row("score", _fmt_p(r["score"]), out)
            _row("threshold", _fmt_p(r["threshold"]), out)
            _row("auto_route", r["auto_route"], out)
            _row("scenario", r["scenario"], out)
            _row("model top teams", ", ".join(
                f"{m}={t}" for m, t in r["model_teams"].items()), out)
        elif key == "issues":
            _row("team", r["team"], out)
            _row("source", r["source"], out)
            _print_issues(r["issues"], out)
        elif key == "summary":
            _row("team", f"{r['team']} {r['team_name']}", out)
            _row("route", r["route"], out)
            _row("scenario", r["scenario"], out)
            print("  top issues:", file=out)
            _print_issues(r["issues"], out)
            print(f"\n  Why: {r['why']}", file=out)


# ---- HTTP mode --------------------------------------------------------------

def trace_http(url: str, text: str, top_k: int, timeout: float = 30) -> dict:
    """POST the complaint to a running /predict endpoint and return the body
    plus the routing headers. Raises ValueError / urllib errors on failure."""
    import json
    import urllib.request

    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(f"URL must start with http:// or https://: {url!r}")
    req = urllib.request.Request(
        url, data=json.dumps({"text": text, "top_k": top_k}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        body = json.loads(resp.read().decode("utf-8"))
        headers = {h: resp.headers.get(h)
                   for h in ("x-routing-scenario", "x-missing-models")}
    if not isinstance(body, dict):
        raise ValueError("response body is not a JSON object")
    return {"status": "ok", "reason": None, "url": url, "body": body,
            "headers": headers}


_HTTP_FIELDS = ["team", "team_confidence", "auto_route", "scenario",
                "team_probs", "extra_team_probs", "stack_team_probs",
                "top_issues", "agentic"]


def print_http(result: dict, out=None) -> None:
    out = out or sys.stdout
    body = result["body"]
    names = team_names()
    print(f"== HTTP trace: {result['url']} ==", file=out)
    for f in _HTTP_FIELDS:
        v = body.get(f)
        if f in ("team_probs", "stack_team_probs") and isinstance(v, dict):
            print(f"  {f}:", file=out)
            _print_probs(v, names, out)
        elif f == "extra_team_probs" and isinstance(v, dict):
            print(f"  {f}:", file=out)
            for m, probs in v.items():
                print(f"    {m}:", file=out)
                _print_probs(probs, names, out, indent="      ")
        elif f == "top_issues" and isinstance(v, list):
            print(f"  {f}:", file=out)
            for i in v:
                print(f"    {i.get('issue')}  {i.get('prob')}", file=out)
        else:
            _row(f, v, out)
    for h, v in result["headers"].items():
        _row(h, v, out)


# ---- CLI --------------------------------------------------------------------

def _parser():
    import argparse

    p = argparse.ArgumentParser(
        description="Trace one complaint through the ClariCase prediction "
                    "pipeline. Reads stdin when no text is given.")
    p.add_argument("text", nargs="?", help="complaint text (default: stdin)")
    p.add_argument("--top-k", type=int, default=3,
                   help="issues to show (default 3)")
    p.add_argument("--top-ngrams", type=int, default=10,
                   help="n-gram contributions to show (default 10)")
    p.add_argument("--skip-embed", action="store_true",
                   help="skip Improved Model 2 and the kNN vote (no torch)")
    p.add_argument("--json", action="store_true",
                   help="print one JSON object instead of text")
    p.add_argument("--http", metavar="URL",
                   help="POST to a running /predict endpoint instead")
    return p


def main(argv: list[str] | None = None) -> int:
    import contextlib
    import json

    args = _parser().parse_args(argv)
    text = args.text if args.text is not None else sys.stdin.read()
    if not text or not text.strip():
        print("ERROR: complaint text is empty", file=sys.stderr)
        return 1

    if args.http:
        try:
            result = trace_http(args.http, text, args.top_k)
        except Exception as e:  # noqa: BLE001 - URLError, HTTPError, bad JSON
            print(f"ERROR: HTTP request failed: {type(e).__name__}: {e}",
                  file=sys.stderr)
            return 3
        if args.json:
            print(json.dumps(to_jsonable({"mode": "http", "text": text,
                                          "stages": {"http": result}}),
                             indent=2))
        else:
            print_http(result)
        return 0

    try:
        # Keep stdout clean (for --json): library progress output goes to stderr.
        with contextlib.redirect_stdout(sys.stderr):
            results = run_trace(text, args)
    except TraceFatal as e:
        print(f"ERROR: Improved Model 1 failed: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(to_jsonable(results), indent=2))
    else:
        print_text(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())

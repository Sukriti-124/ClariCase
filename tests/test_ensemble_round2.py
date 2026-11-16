"""Round-2 routing and issue pipeline, on synthetic data."""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.ensemble import probs as PR
from src.ensemble import thresholds as TH
from src.ensemble.issue_head import conditional_issue_probs, fit_issue_heads
from src.ensemble.knn import topk_neighbours, vote
from src.ensemble.stack import fold_splits
from src.ensemble.train_issue import (IssueData, issue_metrics, pooled,
                                      prior_log, topk_in_team, tune)
from src.ensemble.train_stack import (CANDIDATES, POLICIES, Aligned, align,
                                      build_scenarios, evaluate, pick_final)
from src.utils.config import ID_COL, ISSUE_COL, TEAM_COL

TEAMS3 = ["T01", "T02", "T03"]


# ---- probability files ------------------------------------------------------

def test_reorder_and_block_normalize():
    P = np.array([[0.2, 0.8], [0.6, 0.4]])
    out = PR.reorder(P, ["T02", "T01"], ["T01", "T02", "T03"])
    np.testing.assert_allclose(out, [[0.8, 0.2, 0.0], [0.4, 0.6, 0.0]])
    issue_team = np.array(["A", "A", "B", "B", "B"])
    Q = np.array([[2.0, 0.0, 0.0, 0.0, 0.0]])
    N = PR.block_normalize(Q, issue_team, eps=1e-3)
    np.testing.assert_allclose(N[:, :2].sum(1), 1.0)
    np.testing.assert_allclose(N[:, 2:], 1 / 3)  # empty block -> uniform
    assert (N > 0).all()


def test_issue_file_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("src.ensemble.data.PREDICTIONS_ROOT", tmp_path)
    df = pd.DataFrame({ID_COL: [1, 2], TEAM_COL: ["T01", "T02"],
                       ISSUE_COL: [PR.ISSUES[0], PR.ISSUES[-1]]})
    Q = np.random.default_rng(0).random((2, len(PR.ISSUES)))
    PR.write_issue_probs("m", "val", df, Q)
    back = PR.read_issue_probs("m", "val")
    assert back[ID_COL].tolist() == ["1", "2"]
    np.testing.assert_allclose(PR.issue_matrix(back), Q, rtol=1e-6)
    with pytest.raises(KeyError):
        PR.team_matrix(back)


# ---- thresholds --------------------------------------------------------------

def test_routed_counts_match_brute_force():
    rng = np.random.default_rng(1)
    s = np.round(rng.random(500), 2)
    c = rng.random(500) < s
    grid = np.array([0.0, 0.3, 0.5, 0.505, 0.9, 1.0])
    n, k = TH.routed_counts(s, c, grid)
    for t, nn, kk in zip(grid, n, k):
        assert nn == (s >= t).sum() and kk == c[s >= t].sum()


def test_wilson_lower_is_below_point_estimate_and_tightens():
    assert TH.wilson_lower(90, 100) < 0.9
    assert TH.wilson_lower(9000, 10000) > TH.wilson_lower(90, 100)
    assert TH.wilson_lower(0, 0) == 0.0


def test_choose_threshold_margin_and_wilson_are_stricter():
    rng = np.random.default_rng(2)
    s = rng.random(20000)
    c = rng.random(20000) < (0.5 + 0.5 * s)
    plain = TH.choose_threshold(s, c, 0.9)
    strict = TH.choose_threshold(s, c, 0.9, margin=0.02, wilson=True)
    assert plain is not None and strict is not None and strict > plain
    assert c[s >= plain].mean() >= 0.9
    assert TH.choose_threshold(s, c, 0.9999, wilson=True) is None


def test_team_floors_only_raise_and_flag_hopeless_teams():
    rng = np.random.default_rng(3)
    n = 6000
    chosen = rng.choice(["good", "weak", "bad", "tiny"], n,
                        p=[0.45, 0.3, 0.249, 0.001])
    s = rng.random(n)
    p_right = np.select([chosen == "good", chosen == "weak", chosen == "bad"],
                        [0.97, 0.55 + 0.45 * s, 0.4], 0.9)
    right = rng.random(n) < p_right
    true = np.where(right, chosen, "other")
    out = TH.team_floor_thresholds(s, chosen, true, base=0.1, floor=0.85,
                                   min_routed=30)
    assert out["good"]["status"] == "ok" and out["good"]["threshold"] == 0.1
    assert out["weak"]["status"] == "raised" and out["weak"]["threshold"] > 0.1
    assert out["bad"]["status"] == "never" and out["bad"]["threshold"] is None
    assert out["tiny"]["status"] == "unverified"
    mask = TH.route_mask(s, chosen, 0.1, out)
    assert not mask[chosen == "bad"].any()
    rep = TH.routing_report(mask, chosen, true, floor=0.85)
    assert "weak" not in rep["teams_below_floor"]
    assert "good" not in rep["teams_below_floor"]


def test_temporal_margin_detects_drift_and_is_zero_without_it():
    rng = np.random.default_rng(4)
    n = 30000
    dates = pd.date_range("2025-07-01", periods=n, freq="5min")
    s = rng.random(n)
    late = np.arange(n) >= 2 * n // 3
    base = 0.6 + 0.4 * s
    c_drift = rng.random(n) < np.where(late, base - 0.06, base)
    m = TH.temporal_margin(s, c_drift, dates, 0.9)
    assert m["margin"] >= 0.02  # true drift is 6 points; capped at 0.02
    c_flat = rng.random(n) < base
    assert TH.temporal_margin(s, c_flat, dates, 0.9)["margin"] < 0.01
    assert TH.temporal_margin(s, c_flat, [None] * n, 0.9)["margin"] == 0.0


# ---- nearest neighbours and embedding issue head ------------------------------

def test_knn_numpy_matches_brute_force_and_votes_sum_to_one():
    rng = np.random.default_rng(5)
    T = rng.normal(size=(300, 16)).astype(np.float32)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    Q = T[:40] + 0.01 * rng.normal(size=(40, 16)).astype(np.float32)
    sims, idx = topk_neighbours(T, Q, k=5, chunk=7, backend="numpy")
    brute = np.argsort(-(Q @ T.T), axis=1)[:, :5]
    np.testing.assert_array_equal(idx, brute)
    assert (np.diff(sims, axis=1) <= 1e-6).all()
    assert (idx[:, 0] == np.arange(40)).all()
    labels = rng.integers(0, 3, 300)
    V = vote(idx, sims, labels, 3)
    np.testing.assert_allclose(V.sum(1), 1.0)
    assert (V.argmax(1)[np.all(labels[idx] == labels[idx][:, :1], axis=1)]
            == labels[idx][:, 0][np.all(labels[idx] == labels[idx][:, :1],
                                        axis=1)]).all()


def test_embedding_issue_heads_give_block_distributions():
    rng = np.random.default_rng(6)
    issues = np.array(["A1", "A2", "B1", "B2", "C1"])
    issue_team = np.array(["A", "A", "B", "B", "C"])
    centres = rng.normal(size=(5, 8))
    lab = rng.integers(0, 5, 600)
    X = centres[lab] + 0.3 * rng.normal(size=(600, 8))
    heads = fit_issue_heads(X, issue_team[lab], issues[lab])
    assert heads["C"] == "C1"
    Q = conditional_issue_probs(heads, X, issues, issue_team)
    for t in "ABC":
        np.testing.assert_allclose(Q[:, issue_team == t].sum(1), 1.0)
    own = Q[np.arange(600), lab]
    assert own.mean() > 0.8


def test_im1_batched_issue_probs_match_row_by_row():
    from src.ensemble.export_predictions import im1_issue_conditional
    from src.improved_model_1.model import HierarchicalSVM
    rng = np.random.default_rng(7)
    teams = ["T01", "T02", "T07"]
    tissues = {"T01": PR.ISSUES[:3], "T02": [i for i in PR.ISSUES
                                             if i.startswith("T02")][:2],
               "T07": [i for i in PR.ISSUES if i.startswith("T07")][:1]}
    rows_t, rows_i, feats = [], [], []
    for _ in range(900):
        t = teams[rng.integers(0, 3)]
        i = tissues[t][rng.integers(0, len(tissues[t]))]
        v = np.zeros(40)
        v[hash(t) % 10] += 3
        v[10 + (hash(i) % 30)] += 2
        feats.append(v + rng.random(40))
        rows_t.append(t)
        rows_i.append(i)
    X = sparse.csr_matrix(np.array(feats))
    m = HierarchicalSVM(C_team=1.0, C_issue=1.0).fit(X, rows_t, rows_i)
    m.fit_team_calibrator(X, rows_t)
    m.fit_issue_calibrators(X, rows_t, rows_i, min_rows=100)
    m.calibrated_issue_.pop("T02", None)  # exercise the uncalibrated path
    Q = im1_issue_conditional(m, X[:25])
    ref = m.predict_issue_topk(X[:25], teams=np.array(["T01"] * 25), k=3)
    col = {i: j for j, i in enumerate(PR.ISSUES)}
    for r in range(25):
        for iss, p in ref[r]:
            assert Q[r, col[iss]] == pytest.approx(p, abs=1e-9)
    np.testing.assert_allclose(Q[:, [col[i] for i in tissues["T07"]]], 1.0)
    for t in ("T01", "T02"):
        np.testing.assert_allclose(
            Q[:, [col[i] for i in tissues[t]]].sum(1), 1.0, atol=1e-9)


# ---- team stack ---------------------------------------------------------------

def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _synthetic_aligned(n, seed, start):
    """Three models with independent errors; the second is weaker but makes
    different mistakes, so the stack should beat the first alone."""
    rng = np.random.default_rng(seed)
    y = rng.choice(TEAMS3, n, p=[0.6, 0.3, 0.1])
    yi = np.array([TEAMS3.index(t) for t in y])
    probs = {}
    for name, acc, sharp in (("m1", 0.82, 2.5), ("m2", 0.78, 2.0),
                             ("m3", 0.70, 1.5)):
        hit = rng.random(n) < acc
        noisy = np.where(hit, yi, rng.integers(0, 3, n))
        z = sharp * np.eye(3)[noisy] + rng.normal(0, 0.8, (n, 3))
        probs[name] = _softmax(z)
    dates = pd.date_range("2025-07-01", periods=n, freq="10min").to_numpy()
    return Aligned(ids=np.arange(start, start + n).astype(str), y=y,
                   n_words=rng.integers(20, 400, n),
                   groups=(np.arange(n) // 3).astype(str), dates=dates,
                   probs=probs)


@pytest.fixture(scope="module")
def stacked():
    val = _synthetic_aligned(9000, 0, 0)
    test = _synthetic_aligned(9000, 1, 100_000)
    sc, info = build_scenarios(val, {"test": test}, ["m1", "m2", "m3"],
                               teams=TEAMS3, kind="hgb", C_grid=(0.1, 1.0),
                               n_splits=3)
    res = evaluate(sc, val, {"test": test}, teams=TEAMS3, target=0.9,
                   margin_mode="auto", floor=0.85)
    return sc, info, res, test


def test_stack_reports_every_scenario_policy_and_split(stacked):
    sc, info, res, _ = stacked
    assert set(sc) == {"primary_alone", "gate_on_primary", "stack_alone",
                       "stack_gate"}
    plain = [r for r in res["rows"] if not r["corrected"]]
    assert len(plain) == 4 * len(POLICIES) * 2
    assert len(res["rows"]) == len(plain) + 2 * len(POLICIES)  # corrected
    assert info["stacker_C"] in (0.1, 1.0)
    assert pick_final(res) in CANDIDATES


def test_stack_beats_primary_on_accuracy_and_coverage(stacked):
    sc, _, res, test = stacked
    df = pd.DataFrame(res["rows"])
    t = df[(df["split"] == "test") & (df["policy"] == "global")
           & ~df["corrected"]].set_index("scenario")
    assert t.loc["stack_alone", "accuracy"] > t.loc["primary_alone",
                                                    "accuracy"] + 0.02
    assert t.loc["stack_gate", "coverage"] > t.loc["primary_alone", "coverage"]
    assert t.loc["gate_on_primary", "accuracy"] == t.loc["primary_alone",
                                                         "accuracy"]


def test_margin_floors_hold_on_validation(stacked):
    _, _, res, _ = stacked
    df = pd.DataFrame(res["rows"])
    df = df[~df["corrected"]]
    v = df[(df["split"] == "val") & (df["policy"] == "margin_floors")]
    assert (v["teams_below_floor"] == "").all()
    assert (v["precision"].dropna() >= 0.9).all()
    g = df[(df["split"] == "val")].pivot(index="scenario", columns="policy",
                                         values="coverage")
    assert (g["margin_floors"] <= g["global"] + 1e-9).all()


def test_align_inner_joins_and_requires_probabilities():
    def frame(ids, p):
        df = pd.DataFrame({ID_COL: ids, "actual_team_id": "T01",
                           "predicted_team_id": "T01", "confidence": p,
                           "n_words": 10})
        for t in PR.TEAMS:
            df[f"p_{t}"] = p if t == "T01" else (1 - p) / 10
        return df
    a = align({"a": frame(["1", "2", "3"], 0.9), "b": frame(["2", "3"], 0.5)})
    assert list(a.ids) == ["2", "3"]
    assert a.probs["b"].shape == (2, len(PR.TEAMS))
    with pytest.raises(KeyError):
        align({"a": frame(["1"], 0.9).drop(columns="p_T05")})


def test_fold_splits_keep_groups_together():
    y = np.arange(100) % 3
    groups = (np.arange(100) // 4).astype(str)
    for fit, hold in fold_splits(y, groups, n_splits=5):
        assert set(groups[fit]).isdisjoint(groups[hold])


# ---- issue pooling --------------------------------------------------------------

def _issue_data(n, seed):
    rng = np.random.default_rng(seed)
    issues = np.array(PR.ISSUES)
    lab = rng.integers(0, len(issues), n)
    y_issue = issues[lab]
    y_team = PR.ISSUE_TEAM[lab]

    def component(acc):
        Q = rng.random((n, len(issues)))
        hit = rng.random(n) < acc
        Q[np.arange(n)[hit], lab[hit]] += 3.0
        return np.log(PR.block_normalize(Q))
    logQ = {"good": component(0.7), "helper": component(0.5),
            "noise": component(0.0)}
    return IssueData(ids=np.arange(n).astype(str), y_team=y_team,
                     y_issue=y_issue, logQ=logQ,
                     teams={"src": y_team})


def test_topk_in_team_stays_inside_the_team():
    L = np.random.default_rng(8).random((50, len(PR.ISSUES)))
    teams = np.array(["T07"] * 25 + ["T03"] * 25)
    top = topk_in_team(L, teams, k=5)
    for r in range(50):
        valid = top[r][top[r] >= 0]
        assert (PR.ISSUE_TEAM[valid] == teams[r]).all()
    n_t07 = int((PR.ISSUE_TEAM == "T07").sum())
    assert (top[:25] >= 0).sum(1).max() == min(5, n_t07)


def test_pooling_tunes_away_noise_and_beats_the_best_single_model():
    val, test = _issue_data(6000, 9), _issue_data(6000, 10)
    prior = prior_log(val.y_issue)
    final, table = tune(val, ["good", "helper", "noise"], prior)
    assert final["weights"]["good"] == 1.0
    assert final["weights"]["noise"] <= 0.25
    assert final["weights"]["helper"] > 0
    single = issue_metrics(topk_in_team(pooled(test.logQ, {"good": 1.0}),
                                        test.y_team), test.y_issue)
    both = issue_metrics(topk_in_team(pooled(test.logQ, final["weights"],
                                             prior, final["beta"]),
                                      test.y_team), test.y_issue)
    assert both["accuracy"] > single["accuracy"]
    assert both["top3_accuracy"] >= both["accuracy"]
    assert len(table) == 4 * 4 * 4


# ---- label shift ---------------------------------------------------------------

def test_label_shift_em_recovers_a_shifted_prior():
    from src.ensemble.label_shift import (RollingPrior, adjust,
                                          estimate_prior, prior_from_labels)
    rng = np.random.default_rng(11)
    source = np.array([0.6, 0.3, 0.1])
    target = np.array([0.05, 0.6, 0.35])
    n = 20000
    y = rng.choice(3, n, p=target)
    # A calibrated-to-source model: posterior = likelihood * source prior.
    like = np.where(rng.random(n)[:, None] < 0.75, np.eye(3)[y],
                    rng.dirichlet(np.ones(3), n))
    like = like * 0.8 + 0.2 * rng.dirichlet(np.ones(3), n)
    P = like * source
    P /= P.sum(1, keepdims=True)
    est, iters = estimate_prior(P, source)
    assert iters < 200
    assert np.abs(est - target).max() < 0.05
    acc_before = (P.argmax(1) == y).mean()
    acc_after = (adjust(P, source, est).argmax(1) == y).mean()
    assert acc_after > acc_before + 0.05
    # Online version converges to the same place and is bounded.
    rp = RollingPrior(source, window=5000, min_rows=500, refresh=100)
    assert not rp.active
    for row in P[:3000]:
        rp.observe(row)
    assert rp.active and rp.n_estimates >= 1
    assert np.abs(rp.prior - target).max() < 0.08
    assert np.allclose(rp.adjust(P[0]).sum(), 1.0)
    rp.reset()
    assert not rp.active and np.allclose(rp.prior, source)
    assert np.allclose(prior_from_labels(["a", "a", "b"], ["a", "b", "c"]),
                       [2 / 4, 1 / 4, 1 / 4])


def test_evaluate_reports_corrected_rows_for_stack_scenarios(stacked):
    sc, _, _, test = stacked
    val = _synthetic_aligned(9000, 0, 0)
    res = evaluate(sc, val, {"test": test}, teams=TEAMS3, target=0.9,
                   margin_mode="0.0", floor=0.85, prior_correction=True)
    df = pd.DataFrame(res["rows"])
    corrected = df[df["corrected"]]
    assert set(corrected["scenario"]) == {"stack_alone", "stack_gate"}
    assert set(corrected["split"]) == {"test"}
    assert "test" in res["estimated_priors"]["stack_gate"]
    assert pick_final(res) in CANDIDATES


def test_knn_serving_index_roundtrip(tmp_path):
    from src.ensemble.export_predictions import write_knn_index
    rng = np.random.default_rng(12)
    X = rng.normal(size=(50, 8)).astype(np.float32)
    path = tmp_path / "index.npz"
    write_knn_index(X, rng.integers(0, 11, 50), rng.integers(0, 71, 50),
                    "enc", 256, 25, path=path)
    z = np.load(path)
    assert z["embeddings"].dtype == np.float16 and z["embeddings"].shape == (50, 8)
    assert list(z["teams"]) == list(PR.TEAMS) and int(z["k"]) == 25
    assert str(z["encoder_name"]) == "enc"

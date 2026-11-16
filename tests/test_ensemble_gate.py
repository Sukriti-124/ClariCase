import numpy as np
import pandas as pd
import pytest

from src.ensemble import gate as G
from src.ensemble.data import assemble
from src.ensemble.train_gate import comparison, fit_and_evaluate
from src.utils.config import GROUP_COL, ID_COL, NORTH_STAR_KEY

TEAMS = ["T01", "T02", "T03"]


def _synthetic(n: int, seed: int) -> pd.DataFrame:
    """Primary model whose own confidence is only weakly informative, plus a
    second model whose agreement says a lot about whether the primary is
    right. This is the situation the gate is meant to exploit."""
    rng = np.random.default_rng(seed)
    truth = rng.choice(TEAMS, size=n)
    primary_right = rng.random(n) < 0.80
    wrong = np.array([rng.choice([t for t in TEAMS if t != y]) for y in truth])
    primary = np.where(primary_right, truth, wrong)
    primary_conf = np.clip(
        np.where(primary_right, 0.72, 0.62) + rng.normal(0, 0.15, n), 0, 1)
    second_right = rng.random(n) < 0.80
    wrong2 = np.array([rng.choice([t for t in TEAMS if t != y]) for y in truth])
    second = np.where(second_right, truth, wrong2)
    return pd.DataFrame({
        ID_COL: [str(i + seed * 1_000_000) for i in range(n)],
        "actual_team_id": truth,
        "n_words": rng.integers(20, 400, n),
        "primary_team": primary, "primary_conf": primary_conf,
        "second__team": second,
        "second__conf": np.clip(rng.normal(0.7, 0.1, n), 0, 1),
        "group": rng.integers(0, n // 2, n),
    })


def test_feature_matrix_shape_and_names():
    X = G.build_features(["T01", "T02"], [0.9, 0.4], [["T01", "T03"]],
                         [[0.8, 0.7]], [10, 20], TEAMS)
    names = G.feature_names(TEAMS, ["second"])
    assert X.shape == (2, len(names))
    row = dict(zip(names, X[0]))
    assert row["primary_conf"] == 0.9
    assert row["second_agree"] == 1.0 and row["second_agree_x_conf"] == 0.8
    assert row["primary_is_T01"] == 1.0 and row["primary_is_T02"] == 0.0
    assert dict(zip(names, X[1]))["second_agree"] == 0.0


def test_build_features_rejects_misaligned_secondaries():
    with pytest.raises(ValueError):
        G.build_features(["T01"], [0.9], [["T01", "T02"]], [[0.8, 0.7]], [5],
                         TEAMS)


@pytest.mark.parametrize("kind", G.GATE_KINDS)
def test_cross_fit_scores_cover_every_row(kind):
    df = _synthetic(1500, 0)
    X = G.build_features(df.primary_team, df.primary_conf, [df.second__team],
                         [df.second__conf], df.n_words, TEAMS)
    y = (df.primary_team == df.actual_team_id).to_numpy()
    oof = G.cross_fit_scores(X, y, groups=df.group.to_numpy(), kind=kind)
    assert not np.isnan(oof).any()
    assert ((oof >= 0) & (oof <= 1)).all()


def test_gate_routes_more_than_primary_alone_at_both_targets():
    fit_df, eval_df = _synthetic(6000, 1), _synthetic(6000, 2)
    res = fit_and_evaluate(fit_df, eval_df, "primary", ["second"],
                           kind="logreg")
    cmp = comparison(res)
    for key, floor in (("at_95p", 0.93), ("at_90p", 0.88)):
        assert cmp[key]["gate"] is not None
        base_cov = 0.0 if cmp[key]["primary_alone"] is None \
            else cmp[key]["primary_alone"]["coverage"]
        assert cmp[key]["gate"]["coverage"] > base_cov + 0.10
        # frozen threshold, held-out rows
        assert cmp[key]["gate"]["precision"] >= floor
    # The served threshold is the north-star one.
    assert res["bundle"]["threshold"] == cmp[NORTH_STAR_KEY]["gate"]["threshold"]
    assert res["gate"]["value"] == cmp[NORTH_STAR_KEY]["gate"]["coverage"]


def test_score_one_matches_batch_and_decide_uses_threshold():
    df = _synthetic(3000, 3)
    res = fit_and_evaluate(df, _synthetic(1000, 4), "primary", ["second"])
    bundle = res["bundle"]
    text = "word " * 50
    one = G.score_one(bundle, "T01", 0.9, [("T01", 0.8)], text)
    X = G.build_features(["T01"], [0.9], [["T01"]], [[0.8]], [50],
                         bundle["team_classes"])
    assert one == pytest.approx(float(G.score(bundle, X)[0]))
    agree = G.score_one(bundle, "T01", 0.7, [("T01", 0.7)], text)
    disagree = G.score_one(bundle, "T01", 0.7, [("T02", 0.7)], text)
    assert agree > disagree
    assert G.decide({**bundle, "threshold": 0.5}, 0.6) is True
    assert G.decide({**bundle, "threshold": 0.5}, 0.4) is False
    assert G.decide({**bundle, "threshold": None}, 0.99) is False
    with pytest.raises(ValueError):
        G.score_one(bundle, "T01", 0.9, [], text)


def test_assemble_keeps_only_rows_every_model_scored():
    def preds(ids, team):
        return pd.DataFrame({ID_COL: ids, "actual_team_id": "T01",
                             "predicted_team_id": team, "confidence": 0.5,
                             "n_words": 10})
    groups = pd.DataFrame({ID_COL: ["1", "2", "3"], GROUP_COL: ["a", "b", "c"]})
    out = assemble(preds(["1", "2", "3"], "T01"),
                   {"m2": preds(["2", "3", "4"], "T02")}, groups=groups)
    assert out[ID_COL].tolist() == ["2", "3"]
    assert out["m2__team"].tolist() == ["T02", "T02"]
    assert out["group"].tolist() == ["b", "c"]

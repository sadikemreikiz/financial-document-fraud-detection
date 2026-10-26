import numpy as np
import pytest

from docfraud.fusion.model import (
    FusionModel,
    Platt,
    calibration_slope,
    out_of_fold,
    probabilities,
)

NAMES = ("a", "b", "constant")


def data(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.normal(2, 3, n), rng.normal(0, 1, n), np.ones(n)])
    y = rng.random(n) < 1 / (1 + np.exp(-(0.8 * (x[:, 0] - 2) - 0.5 * x[:, 1])))
    return x, y


def test_the_model_standardises_on_its_own_receipts_and_splits_its_log_odds() -> None:
    x, y = data()
    model = FusionModel.fit(x, y, NAMES)
    assert model.scaler.mean_ == pytest.approx(x.mean(axis=0))
    assert model.scaler.scale_[2] == 1.0  # a constant input keeps a spread of 1
    parts = model.contributions(x)
    assert parts.sum(axis=1) + model.intercept == pytest.approx(model.log_odds(x))
    assert parts[:, 2] == pytest.approx(0.0)
    weights = model.coefficients
    assert list(weights) == list(NAMES)
    assert weights["a"] > 0 > weights["b"]


def test_fitting_refuses_mismatched_names_and_a_single_class() -> None:
    x, y = data()
    with pytest.raises(ValueError, match="names"):
        FusionModel.fit(x, y, NAMES[:2])
    with pytest.raises(ValueError, match="forged and authentic"):
        FusionModel.fit(x, np.ones(len(y), dtype=bool), NAMES)


def test_out_of_fold_scores_never_see_their_own_fold() -> None:
    x, y = data()
    fold_of = np.arange(len(y)) % 5
    log_odds, models = out_of_fold(x, y, fold_of, NAMES)
    assert sorted(models) == [0, 1, 2, 3, 4]
    held = fold_of == 0
    assert log_odds[held] == pytest.approx(models[0].log_odds(x[held]))
    flipped = y.copy()
    flipped[held] = ~flipped[held]  # labels of fold 0 cannot reach fold 0's scores
    again, _ = out_of_fold(x, flipped, fold_of, NAMES)
    assert again[held] == pytest.approx(log_odds[held])
    assert not np.allclose(again[~held], log_odds[~held])


def test_platt_scaling_keeps_the_order_and_refuses_to_reverse_it() -> None:
    x, y = data()
    log_odds = FusionModel.fit(x, y, NAMES).log_odds(x)
    platt = Platt.fit(log_odds, y)
    p = platt.probabilities(log_odds)
    assert platt.slope > 0
    assert np.all((p > 0) & (p < 1))
    order = np.argsort(log_odds)
    assert np.all(np.diff(p[order]) >= 0)
    with pytest.raises(ValueError, match="reverse"):
        Platt.fit(-log_odds, y)


def test_the_calibration_slope_shows_overconfidence() -> None:
    rng = np.random.default_rng(1)
    z = rng.normal(0, 1.5, 20_000)
    y = rng.random(z.size) < probabilities(z)
    assert calibration_slope(y, probabilities(z)) == pytest.approx(1.0, abs=0.05)
    assert calibration_slope(y, probabilities(3 * z)) == pytest.approx(1 / 3, abs=0.03)

"""How to compare the fusion inputs of the two splits (ADR-0029, section 7 (d)).

Writes reports/fusion/input-shift-method.md, the evidence behind the method of
:mod:`docfraud.fusion.shift`. It uses the training inputs only, out of fold, and no label: the 562
training receipts are split at random into 372 and 190, the size of the validation split, so
that there is no shift between the two parts, and the tests are asked whether there is one.
Then one input of the smaller part is moved by a known amount, and the tests are asked again.

Three tests are compared: the Kolmogorov-Smirnov distance with the asymptotic p-value of SciPy,
the same distance with a permutation p-value, and a classifier that tries to tell the two parts
apart (a logistic regression on the seven inputs of track T2, cross-validated ROC-AUC).

Usage: ``uv run python experiments/input_shift_method.py`` (needs
artifacts/fusion/training-inputs.json, made by ``docfraud fusion training-inputs``)
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from docfraud.fusion.inputs import FUSION, LINK_1, PRACTICAL, SYSTEM_2_INPUT
from docfraud.fusion.shift import permutation_test
from docfraud.fusion.training_inputs import load_table
from docfraud.markdown import Cell, Table, render_table

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "artifacts" / "fusion" / "training-inputs.json"
OUTPUT = ROOT / "reports" / "fusion" / "input-shift-method.md"
VALIDATION = 190  # receipts in the validation split
SPLITS = 300  # random splits without a shift
CLASSIFIER_SPLITS = 200
SHIFTED_SPLITS = 100  # random splits per shift
PERMUTATIONS = 1_000
LEVEL = 0.05

type Matrix = npt.NDArray[np.float64]


def split(rng: np.random.Generator, n: int) -> npt.NDArray[np.bool_]:
    """A random part of ``VALIDATION`` receipts out of ``n``."""
    second = np.zeros(n, dtype=bool)
    second[rng.permutation(n)[:VALIDATION]] = True
    return second


def classifier_auc(x: Matrix, second: npt.NDArray[np.bool_]) -> float:
    """How well a logistic regression tells the two parts apart, out of fold."""
    model = make_pipeline(StandardScaler(), LogisticRegression())
    folds = StratifiedKFold(5, shuffle=True, random_state=0)
    p = cross_val_predict(model, x, second, cv=folds, method="predict_proba")[:, 1]
    return float(roc_auc_score(second, p))


def false_alarms(x: Matrix, rng: np.random.Generator) -> Table:
    """How often each test finds a shift between two random parts, where there is none."""
    rows: list[list[Cell]] = []
    for j, name in enumerate(PRACTICAL):
        asymptotic = permutation = 0
        for _ in range(SPLITS):
            second = split(rng, x.shape[0])
            a, b = x[~second, j], x[second, j]
            asymptotic += stats.ks_2samp(a, b).pvalue < LEVEL
            permutation += (
                permutation_test(a, b, PERMUTATIONS, int(rng.integers(2**31))).p_value < LEVEL
            )
        rows.append([name, len(np.unique(x[:, j])), asymptotic / SPLITS, permutation / SPLITS])
    se = (LEVEL * (1 - LEVEL) / SPLITS) ** 0.5
    return Table(
        "Without a shift: how often each test finds one",
        ("Input", "Distinct values", "KS, asymptotic p", "KS, permutation p"),
        rows,
        note=f"{SPLITS} random splits of the 562 training receipts into 372 and {VALIDATION}; "
        f"share of splits with p < {LEVEL}, which should be about {LEVEL} (standard error of "
        f"each share about {se:.3f}). Permutation p from {PERMUTATIONS:,} shuffles; track T2, "
        "system 2 seed 0.",
    )


def classifier_null(x: Matrix, rng: np.random.Generator) -> tuple[Table, float]:
    """The classifier's AUC between random parts, and the AUC that counts as a shift found."""
    aucs = np.array([classifier_auc(x, split(rng, x.shape[0])) for _ in range(CLASSIFIER_SPLITS)])
    critical = float(np.quantile(aucs, 1 - LEVEL))
    table = Table(
        "Without a shift: the classifier's ROC-AUC",
        ("Mean", "Median", "95th percentile"),
        [[float(aucs.mean()), float(np.median(aucs)), critical]],
        note=f"{CLASSIFIER_SPLITS} random splits; a logistic regression with default settings on "
        "the seven inputs of track T2, standardised, 5-fold cross-validated. An AUC above the "
        "95th percentile is taken as a shift found.",
    )
    return table, critical


def moved(
    x: Matrix, second: npt.NDArray[np.bool_], name: str, by: float, rng: np.random.Generator
) -> Matrix:
    """The inputs with one input of the second part moved: a score raised, a share of 1 grown."""
    out = x.copy()
    j = PRACTICAL.index(name)
    if name == LINK_1:
        zeros = np.flatnonzero(second & (out[:, j] == 0))
        out[rng.choice(zeros, round(by * second.sum()), replace=False), j] = 1.0
    else:
        out[second, j] = np.clip(out[second, j] + by, 0.0, 1.0)
    return out


def power(x: Matrix, rng: np.random.Generator, critical: float) -> Table:
    """How often each test finds a known shift of one input."""
    rows: list[list[Cell]] = []
    columns = [PRACTICAL.index(n) for n in FUSION]
    for name, by in (
        (SYSTEM_2_INPUT, 0.04),
        (SYSTEM_2_INPUT, 0.08),
        (LINK_1, 0.05),
        (LINK_1, 0.10),
    ):
        j = PRACTICAL.index(name)
        found = np.zeros(3)  # asymptotic KS, permutation KS, classifier
        for _ in range(SHIFTED_SPLITS):
            second = split(rng, x.shape[0])
            shifted = moved(x, second, name, by, rng)
            a, b = shifted[~second, j], shifted[second, j]
            test = permutation_test(a, b, PERMUTATIONS, int(rng.integers(2**31)))
            found += (
                stats.ks_2samp(a, b).pvalue < LEVEL,
                test.p_value < LEVEL,
                classifier_auc(shifted[:, columns], second) > critical,
            )
        change = f"raised by {by:.2f}" if name != LINK_1 else f"share of 1 grown by {by:.2f}"
        rows.append([name, change, *(found / SHIFTED_SPLITS)])
    return Table(
        "With one input moved: how often each test finds it",
        ("Input", "Moved", "KS, asymptotic p", "KS, permutation p", "Classifier"),
        rows,
        note=f"{SHIFTED_SPLITS} random splits per row; the input of the {VALIDATION} receipts is "
        "moved (scores kept within 0 and 1). KS: p < 0.05 for that input alone; classifier: AUC "
        "above the 95th percentile without a shift.",
    )


def main() -> None:
    """Write the report."""
    rows = load_table(INPUTS).rows("ocr", 0)
    x = np.array([[r.inputs[name] for name in PRACTICAL] for r in rows], dtype=float)
    rng = np.random.default_rng(0)
    null_table, critical = classifier_null(x[:, [PRACTICAL.index(n) for n in FUSION]], rng)
    tables = [false_alarms(x, rng), null_table, power(x, rng, critical)]
    header = (
        "# How to compare the fusion inputs of the two splits\n\n"
        "> Generated by `experiments/input_shift_method.py`. Do not edit by hand.\n\n"
        "The evidence behind the method of the input comparison of "
        "[ADR-0029](../../docs/decisions/0029-fusion-preregistration.md), section 7 (d), "
        "chosen before any validation input was computed. Only the out-of-fold inputs of the "
        "training split are used, without labels: random splits into 372 and 190 receipts have "
        "no shift by construction, and moving one input of the smaller part makes a known one. "
        "The KS distance with a permutation p-value was chosen: where values tie, the "
        "asymptotic p-value finds a shift less often than the 5 % it should when there is none, "
        "and so misses real ones too; and the classifier finds a shift of one input less often "
        "than the KS distance does.\n\n"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(header + "\n".join(render_table(t) for t in tables), encoding="utf-8")
    print(f"written: {OUTPUT.relative_to(ROOT)}")  # noqa: T201


if __name__ == "__main__":
    main()

import math

import pytest

from a2 import evaluate


def test_metrics_known_values():
    y, p = [1, 2, 3], [1, 4, 6]
    assert evaluate.mse(y, p) == pytest.approx(13 / 3)
    assert evaluate.rmse(y, p) == pytest.approx(math.sqrt(13 / 3))
    assert evaluate.mae(y, p) == pytest.approx(5 / 3)
    assert evaluate.accuracy([1, 0, 1], [1, 1, 1]) == pytest.approx(2 / 3)


def test_perfect_predictions_score_zero():
    assert evaluate.mse([2, 3], [2, 3]) == 0


@pytest.mark.parametrize("y,p", [([], []), ([1, 2], [1])])
def test_bad_inputs_raise(y, p):
    with pytest.raises(ValueError):
        evaluate.mse(y, p)


def test_report_and_table_sorted_by_first_metric():
    rows = [evaluate.report("bad", [1, 2], [3, 4]), evaluate.report("good", [1, 2], [1, 2])]
    assert set(rows[0]) == {"model", "MSE", "RMSE", "MAE"}
    table = evaluate.results_table(rows)
    assert list(table["model"]) == ["good", "bad"]

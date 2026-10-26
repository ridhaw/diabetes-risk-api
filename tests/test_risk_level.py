import pytest

from diabetes_api.main import risk_level


@pytest.mark.parametrize(
    "probability, expected",
    [(0.0, "faible"), (0.29, "faible"), (0.3, "modéré"), (0.59, "modéré"), (0.6, "élevé"), (1.0, "élevé")],
)
def test_risk_level_thresholds(probability, expected):
    assert risk_level(probability) == expected

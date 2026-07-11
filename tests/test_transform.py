import json
from pathlib import Path

import pytest

from bondviz.data.transform import TransformError, normalize

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_normalize_normal_payload():
    payload = _load("raw_payload_normal.json")
    curve = normalize(payload)

    assert curve.curve_date.isoformat() == "2024-01-02"
    assert curve.curve_type == "par_yield"
    assert curve.source == "ust_par_yield"

    tenors = [p.tenor_months for p in curve.points]
    assert tenors == sorted(tenors)
    assert set(tenors) == {1, 2, 3, 6, 12, 24, 36, 60, 84, 120, 240, 360}

    ten_year = next(p for p in curve.points if p.tenor_months == 120)
    assert ten_year.yield_pct == pytest.approx(3.95)


def test_normalize_malformed_payload_raises():
    payload = _load("raw_payload_malformed.json")
    # a valid date but no numeric tenor columns survive -> nothing to store
    with pytest.raises(TransformError):
        normalize(payload)


def test_normalize_empty_payload_raises():
    payload = _load("raw_payload_empty.json")
    with pytest.raises(TransformError):
        normalize(payload)


def test_normalize_missing_date_raises():
    with pytest.raises(TransformError):
        normalize({"record": {"BC_10YEAR": "4.00"}})


def test_normalize_is_pure_and_deterministic():
    payload = _load("raw_payload_normal.json")
    assert normalize(payload) == normalize(payload)

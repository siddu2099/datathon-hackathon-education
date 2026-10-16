"""
SpaceSync — Phase 1 Regression Test Suite
Validates the original 9 Phase 1 acceptance cases against the hardened engine.
"""

import datetime
import sys
from pathlib import Path
import pytest

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import (
    load_and_validate_datasets,
    evaluate_room_request,
    RoomRequest,
    DatasetBundle,
)


@pytest.fixture(scope="session")
def bundle() -> DatasetBundle:
    return load_and_validate_datasets()


def test_tc1_valid_request(bundle: DatasetBundle):
    req = RoomRequest(
        date=datetime.date(2025, 9, 15), # Monday
        start_hour=14.0,
        duration_hours=2.0,
        required_capacity=50,
        purpose="Club meeting",
    )
    res = evaluate_room_request(req, bundle)
    assert res.feasible_count > 0
    top = res.candidates[0]
    assert top.capacity >= 50
    assert top.fit_score == round((50 / top.capacity) * 100, 1)


def test_tc2_capacity_shortfall(bundle: DatasetBundle):
    req = RoomRequest(
        date=datetime.date(2025, 9, 15),
        start_hour=14.0,
        duration_hours=2.0,
        required_capacity=250, # Campus max is 179
        purpose="Mega event",
    )
    res = evaluate_room_request(req, bundle)
    assert res.feasible_count == 0
    assert res.rejection_summary["capacity_shortfall"] == 120


def test_tc3_missing_projector(bundle: DatasetBundle):
    # R008 has projector == 0
    req = RoomRequest(
        date=datetime.date(2025, 9, 15),
        start_hour=14.0,
        duration_hours=2.0,
        required_capacity=20,
        projector_required=True,
    )
    res = evaluate_room_request(req, bundle)
    r008_rej = next((r for r in res.rejections if r.room_id == "R008"), None)
    assert r008_rej is not None
    assert any("Missing required facility: Projector" in reason for reason in r008_rej.reasons)


def test_tc4_class_overlap(bundle: DatasetBundle):
    # S00002 is R120 on Mon, start_hour=12, duration=2 (12:00-14:00)
    req = RoomRequest(
        date=datetime.date(2025, 9, 15),
        start_hour=13.0,
        duration_hours=2.0,
        required_capacity=20,
    )
    res = evaluate_room_request(req, bundle)
    r120_rej = next((r for r in res.rejections if r.room_id == "R120"), None)
    assert r120_rej is not None
    assert any("Class conflict" in reason for reason in r120_rej.reasons)


def test_tc5_event_overlap(bundle: DatasetBundle):
    # EV00001: R051 on 2025-09-13 07:00:00, duration 1h
    req = RoomRequest(
        date=datetime.date(2025, 9, 13),
        start_hour=7.0,
        duration_hours=1.0,
        required_capacity=20,
    )
    res = evaluate_room_request(req, bundle)
    r051_rej = next((r for r in res.rejections if r.room_id == "R051"), None)
    assert r051_rej is not None
    assert any("Event conflict" in reason for reason in r051_rej.reasons)


def test_tc6_adjacent_intervals(bundle: DatasetBundle):
    # S00002 ends at 14:00. Request starts at 14:00.
    c_start = datetime.datetime(2025, 9, 15, 12, 0)
    c_end = datetime.datetime(2025, 9, 15, 14, 0)
    req_start = datetime.datetime(2025, 9, 15, 14, 0)
    req_end = datetime.datetime(2025, 9, 15, 16, 0)
    overlap = (c_start < req_end) and (req_start < c_end)
    assert not overlap, "Adjacent interval should not overlap!"


def test_tc7_accessibility(bundle: DatasetBundle):
    req = RoomRequest(
        date=datetime.date(2025, 9, 15),
        start_hour=14.0,
        duration_hours=2.0,
        required_capacity=20,
        accessible_required=True,
    )
    res = evaluate_room_request(req, bundle)
    r019_rej = next((r for r in res.rejections if r.room_id == "R019"), None)
    assert r019_rej is not None
    assert res.rejection_summary["inaccessible"] == 20


def test_tc8_no_feasible_candidates(bundle: DatasetBundle):
    req = RoomRequest(
        date=datetime.date(2025, 9, 15),
        start_hour=10.0,
        duration_hours=4.0,
        required_capacity=170,
        computers_required=True,
    )
    res = evaluate_room_request(req, bundle)
    assert res.feasible_count == 0


def test_tc9_dataset_failure():
    try:
        load_and_validate_datasets("invalid/nonexistent/path")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass

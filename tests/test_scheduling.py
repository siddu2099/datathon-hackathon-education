"""
SpaceSync — Scheduling Correctness & Temporal Engine Test Suite (Phase 2)
DATATHON 2026 | VIT-AP University | Education Track
"""

import copy
import datetime
import sys
from pathlib import Path
import numpy as np
import pytest
import pandas as pd

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import (
    load_and_validate_datasets,
    evaluate_room_request,
    RoomRequest,
    DatasetBundle,
    normalize_weekday,
)


@pytest.fixture(scope="session")
def real_bundle() -> DatasetBundle:
    """Load the authentic challenge datasets."""
    return load_and_validate_datasets()


# -----------------------------------------------------------------------------
# 1. TEMPORAL BOUNDARY TESTS (Adjacent, Nested, Partial)
# -----------------------------------------------------------------------------

def test_adjacent_intervals_no_conflict(real_bundle: DatasetBundle):
    """
    Boundary condition 1:
    A class from 10:00 to 11:00 and a request from 11:00 to 12:00 do not conflict.
    Formula: start_a < end_b and start_b < end_a.
    10:00 < 12:00 (True), but 11:00 < 11:00 (False) -> No conflict.
    """
    # Create isolated bundle with Room R999 and a 10:00-11:00 class
    custom_rooms = pd.DataFrame([{
        "room_id": "R999", "building": "North", "floor": 1, "room_type": "Classroom",
        "capacity": 50, "projector": 1, "computers": 1, "accessible": 1, "power_outlets": 10,
        "is_corrupted": False
    }])
    custom_classes = pd.DataFrame([{
        "schedule_id": "S_ADJ", "room_id": "R999", "course_code": "CS101",
        "weekday": "Mon", "start_hour": 10.0, "duration_hours": 1.0, "end_hour": 11.0,
        "start_hour_num": 10.0, "duration_hours_num": 1.0, "weekday_norm": "Mon",
        "expected_attendance": 30, "department": "CSE", "is_corrupted": False
    }])
    custom_events = pd.DataFrame(columns=[
        "event_id", "room_id", "event_type", "start_time", "duration_hours", "end_time",
        "expected_attendance", "actual_attendance", "is_corrupted"
    ])
    bundle = DatasetBundle(
        rooms=custom_rooms,
        class_schedule=custom_classes,
        room_requests=pd.DataFrame(),
        events=custom_events,
        audit_summary={"room_weekday_collisions": set()},
    )

    # Request starts exactly at 11:00 (11:00 to 12:00) on a Monday
    req = RoomRequest(
        date=datetime.date(2025, 9, 15), # Monday
        start_hour=11.0,
        duration_hours=1.0,
        required_capacity=30,
    )
    result = evaluate_room_request(req, bundle)
    assert result.feasible_count == 1, "Adjacent request starting at class end_hour should be feasible"
    assert result.candidates[0].room_id == "R999"


def test_nested_interval_conflict(real_bundle: DatasetBundle):
    """
    Boundary condition 2:
    A class from 10:00 to 12:00 and a request from 10:30 to 11:30 conflict.
    """
    custom_rooms = pd.DataFrame([{
        "room_id": "R999", "building": "North", "floor": 1, "room_type": "Classroom",
        "capacity": 50, "projector": 1, "computers": 1, "accessible": 1, "power_outlets": 10,
        "is_corrupted": False
    }])
    custom_classes = pd.DataFrame([{
        "schedule_id": "S_NEST", "room_id": "R999", "course_code": "CS102",
        "weekday": "Mon", "start_hour": 10.0, "duration_hours": 2.0, "end_hour": 12.0,
        "start_hour_num": 10.0, "duration_hours_num": 2.0, "weekday_norm": "Mon",
        "expected_attendance": 30, "department": "CSE", "is_corrupted": False
    }])
    bundle = DatasetBundle(
        rooms=custom_rooms,
        class_schedule=custom_classes,
        room_requests=pd.DataFrame(),
        events=pd.DataFrame(columns=["event_id", "room_id", "event_type", "start_time", "duration_hours", "end_time", "is_corrupted"]),
        audit_summary={"room_weekday_collisions": set()},
    )

    # Fractional hour: 10:30 (10.5) to 11:30 (duration 1.0)
    req = RoomRequest(
        date=datetime.date(2025, 9, 15), # Monday
        start_hour=10.5,
        duration_hours=1.0,
        required_capacity=30,
    )
    result = evaluate_room_request(req, bundle)
    assert result.feasible_count == 0, "Nested interval must conflict"
    assert result.rejection_summary["class_conflict"] == 1
    assert "CS102" in result.rejections[0].reasons[0]


def test_partial_interval_conflict(real_bundle: DatasetBundle):
    """
    Boundary condition 3:
    A class from 10:00 to 11:30 and a request from 11:00 to 12:00 conflict.
    """
    custom_rooms = pd.DataFrame([{
        "room_id": "R999", "building": "North", "floor": 1, "room_type": "Classroom",
        "capacity": 50, "projector": 1, "computers": 1, "accessible": 1, "power_outlets": 10,
        "is_corrupted": False
    }])
    custom_classes = pd.DataFrame([{
        "schedule_id": "S_PART", "room_id": "R999", "course_code": "CS103",
        "weekday": "Mon", "start_hour": 10.0, "duration_hours": 1.5, "end_hour": 11.5,
        "start_hour_num": 10.0, "duration_hours_num": 1.5, "weekday_norm": "Mon",
        "expected_attendance": 30, "department": "CSE", "is_corrupted": False
    }])
    bundle = DatasetBundle(
        rooms=custom_rooms,
        class_schedule=custom_classes,
        room_requests=pd.DataFrame(),
        events=pd.DataFrame(columns=["event_id", "room_id", "event_type", "start_time", "duration_hours", "end_time", "is_corrupted"]),
        audit_summary={"room_weekday_collisions": set()},
    )

    req = RoomRequest(
        date=datetime.date(2025, 9, 15), # Monday
        start_hour=11.0,
        duration_hours=1.0, # 11:00 to 12:00
        required_capacity=30,
    )
    result = evaluate_room_request(req, bundle)
    assert result.feasible_count == 0, "Partial overlap (10:00-11:30 vs 11:00-12:00) must conflict"
    assert result.rejection_summary["class_conflict"] == 1


# -----------------------------------------------------------------------------
# 2. CROSS-MIDNIGHT CLASSES & EVENTS
# -----------------------------------------------------------------------------

def test_previous_day_cross_midnight_class():
    """
    Boundary condition 4:
    A previous-day class starting at 23:00 on Sunday with duration 3 hours (ends Monday 02:00)
    must block a request on Monday from 00:30 to 01:30.
    """
    custom_rooms = pd.DataFrame([{
        "room_id": "R888", "building": "East", "floor": 2, "room_type": "Lab",
        "capacity": 60, "projector": 1, "computers": 1, "accessible": 1, "power_outlets": 15,
        "is_corrupted": False
    }])
    custom_classes = pd.DataFrame([{
        "schedule_id": "S_NIGHT", "room_id": "R888", "course_code": "NIGHT_LAB",
        "weekday": "Sun", "start_hour": 23.0, "duration_hours": 3.0, "end_hour": 26.0,
        "start_hour_num": 23.0, "duration_hours_num": 3.0, "weekday_norm": "Sun",
        "expected_attendance": 20, "department": "ECE", "is_corrupted": False
    }])
    bundle = DatasetBundle(
        rooms=custom_rooms,
        class_schedule=custom_classes,
        room_requests=pd.DataFrame(),
        events=pd.DataFrame(columns=["event_id", "room_id", "event_type", "start_time", "duration_hours", "end_time", "is_corrupted"]),
        audit_summary={"room_weekday_collisions": set()},
    )

    # Request on Monday (2025-09-15) from 00:30 to 01:30
    req = RoomRequest(
        date=datetime.date(2025, 9, 15), # Monday
        start_hour=0.5, # 00:30
        duration_hours=1.0, # until 01:30
        required_capacity=20,
    )
    result = evaluate_room_request(req, bundle)
    assert result.feasible_count == 0, "Sunday night class extending past midnight into Monday must block early Monday request"
    assert result.rejection_summary["class_conflict"] == 1
    assert "NIGHT_LAB" in result.rejections[0].reasons[0]


def test_dated_event_crossing_midnight():
    """
    Boundary condition 5:
    A dated event on 2025-08-24 starting at 23:00 with duration 3h (ends 2025-08-25 02:00).
    Real event EV00006 in room R104.
    Must block request on 2025-08-25 from 01:00 to 02:00.
    """
    bundle = load_and_validate_datasets()
    
    # Request on next day (2025-08-25) early morning
    req = RoomRequest(
        date=datetime.date(2025, 8, 25),
        start_hour=1.0, # 01:00
        duration_hours=1.0, # 01:00 to 02:00
        required_capacity=20,
    )
    result = evaluate_room_request(req, bundle)
    r104_rej = next((r for r in result.rejections if r.room_id == "R104"), None)
    assert r104_rej is not None, "R104 must be blocked on 2025-08-25 01:00 by midnight-crossing event EV00006"
    assert any("Event conflict" in r and "Exam" in r for r in r104_rej.reasons)


def test_event_on_other_date_does_not_block(real_bundle: DatasetBundle):
    """
    Boundary condition 6:
    An event on another date does not block merely because its clock time matches.
    EV00001 is on room R051 on Saturday 2025-09-13 07:00-08:00.
    A request on Saturday 2025-09-20 07:00-08:00 must NOT be blocked by EV00001.
    """
    req = RoomRequest(
        date=datetime.date(2025, 9, 20), # Next Saturday
        start_hour=7.0,
        duration_hours=1.0,
        required_capacity=20,
    )
    result = evaluate_room_request(req, real_bundle)
    r051_rej = next((r for r in result.rejections if r.room_id == "R051"), None)
    if r051_rej:
        reasons_str = " ".join(r051_rej.reasons)
        assert "EV00001" not in reasons_str, "EV00001 falsely leaked into a different calendar date"


# -----------------------------------------------------------------------------
# 3. RECURRING WEEKDAY NORMALIZATION & INTEGRITY
# -----------------------------------------------------------------------------

def test_weekday_normalization():
    assert normalize_weekday("Monday") == "Mon"
    assert normalize_weekday("mon") == "Mon"
    assert normalize_weekday("Fri") == "Fri"
    assert normalize_weekday("FRIDAY") == "Fri"
    assert normalize_weekday("wednesday") == "Wed"
    assert normalize_weekday("InvalidDay") is None
    assert normalize_weekday(None) is None


def test_recurring_classes_apply_across_dates(real_bundle: DatasetBundle):
    """
    Class S00002 is on Monday 12:00-14:00 in R120.
    It must consistently block all Mondays (Sep 8, Sep 15, Sep 22) but NOT Tuesday.
    """
    for mon in [datetime.date(2025, 9, 8), datetime.date(2025, 9, 15), datetime.date(2025, 9, 22)]:
        req = RoomRequest(date=mon, start_hour=12.0, duration_hours=2.0, required_capacity=20)
        res = evaluate_room_request(req, real_bundle)
        r120_rej = next((r for r in res.rejections if r.room_id == "R120"), None)
        assert r120_rej is not None, f"R120 must be blocked on {mon}"
        assert any("CS030" in r for r in r120_rej.reasons)

    # Tuesday 12:00-14:00: S00002 must not conflict
    req_tue = RoomRequest(date=datetime.date(2025, 9, 16), start_hour=12.0, duration_hours=2.0, required_capacity=20)
    res_tue = evaluate_room_request(req_tue, real_bundle)
    r120_rej_tue = next((r for r in res_tue.rejections if r.room_id == "R120"), None)
    if r120_rej_tue:
        assert "CS030" not in " ".join(r120_rej_tue.reasons)


# -----------------------------------------------------------------------------
# 4. FAIL-SAFE VERIFICATION ON CORRUPTED RECORDS
# -----------------------------------------------------------------------------

def test_corrupted_schedule_record_fails_safely():
    """
    Boundary condition:
    If a room has an unparseable or corrupted scheduling record on the requested day,
    the engine must NOT claim availability. It must fail safe with a verification failure.
    """
    custom_rooms = pd.DataFrame([{
        "room_id": "R777", "building": "South", "floor": 1, "room_type": "Classroom",
        "capacity": 80, "projector": 1, "computers": 1, "accessible": 1, "power_outlets": 20,
        "is_corrupted": False
    }])
    # Corrupted class record: start_hour is NaN, duration is -1
    custom_classes = pd.DataFrame([{
        "schedule_id": "S_CORRUPT", "room_id": "R777", "course_code": "CS_BAD",
        "weekday": "Mon", "start_hour": np.nan, "duration_hours": -1.0, "end_hour": np.nan,
        "start_hour_num": np.nan, "duration_hours_num": -1.0, "weekday_norm": "Mon",
        "expected_attendance": 20, "department": "CSE", "is_corrupted": True
    }])
    bundle = DatasetBundle(
        rooms=custom_rooms,
        class_schedule=custom_classes,
        room_requests=pd.DataFrame(),
        events=pd.DataFrame(columns=["event_id", "room_id", "event_type", "start_time", "duration_hours", "end_time", "is_corrupted"]),
        audit_summary={"room_weekday_collisions": set()},
    )

    req = RoomRequest(
        date=datetime.date(2025, 9, 15), # Monday
        start_hour=14.0,
        duration_hours=2.0,
        required_capacity=40,
    )
    result = evaluate_room_request(req, bundle)
    assert result.feasible_count == 0, "Corrupted record must fail safe; room cannot be reported available"
    assert result.rejection_summary["verification_failure"] == 1
    assert "Verification failure" in result.rejections[0].reasons[0]


# -----------------------------------------------------------------------------
# 5. INTEGRATION TEST WITH AUTHENTIC CHALLENGE DATASETS
# -----------------------------------------------------------------------------

def test_authentic_request_produces_ranked_candidates(real_bundle: DatasetBundle):
    """Verify that a legitimate request on the real dataset generates ranked candidates with valid fit scores."""
    req = RoomRequest(
        date=datetime.date(2025, 9, 15),
        start_hour=14.0,
        duration_hours=2.0,
        required_capacity=50,
        purpose="Hackathon Team Meeting",
    )
    result = evaluate_room_request(req, real_bundle)
    assert result.feasible_count > 0
    top = result.candidates[0]
    assert top.capacity >= 50
    assert 0 < top.fit_score <= 100
    assert len(top.qualification_reasons) >= 3
    # Capacity fit ranking descending:
    for i in range(len(result.candidates) - 1):
        assert result.candidates[i].fit_score >= result.candidates[i + 1].fit_score

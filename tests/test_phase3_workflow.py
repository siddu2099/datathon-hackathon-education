"""
SpaceSync — Phase 3 Test Suite: Alternatives & Persistent Reservation Workflow
DATATHON 2026 | VIT-AP University | Education Track
"""

import copy
import datetime
import os
from pathlib import Path
import sqlite3
import sys
import pytest

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import (
    load_and_validate_datasets,
    evaluate_room_request,
    find_alternative_slots,
    init_db,
    create_room_request_record,
    confirm_reservation,
    cancel_reservation,
    get_active_reservations,
    get_request_history,
    get_audit_logs,
    RoomRequest,
    DatasetBundle,
)


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    """Provide an isolated SQLite database path for testing."""
    db_file = tmp_path / "test_spacesync.db"
    init_db(db_file)
    return db_file


@pytest.fixture(scope="session")
def bundle() -> DatasetBundle:
    return load_and_validate_datasets()


# -----------------------------------------------------------------------------
# 1. ALTERNATIVE-SLOT TESTS
# -----------------------------------------------------------------------------

def test_find_valid_alternatives_when_slot_occupied(bundle: DatasetBundle, test_db_path: Path):
    """
    When a request has 0 feasible rooms in the requested slot,
    the engine should discover valid nearby time openings.
    """
    # Mon 2025-09-15 10:00-14:00 (4h) with 140 attendance, projector=True
    # In this slot, large rooms R003, R007, R013, R014 etc. have class commitments
    req = RoomRequest(
        date=datetime.date(2025, 9, 15),
        start_hour=10.0,
        duration_hours=4.0,
        required_capacity=140,
        projector_required=True,
    )
    initial_res = evaluate_room_request(req, bundle, db_path=test_db_path)
    
    # Run alternative search
    alternatives = find_alternative_slots(req, bundle, db_path=test_db_path, max_alternatives=3)
    
    if initial_res.feasible_count == 0:
        assert len(alternatives) > 0, "Should find valid alternatives nearby"
    
    # Verify that every alternative returned is indeed 100% feasible in its slot
    for alt in alternatives:
        assert alt.candidate.capacity >= req.required_capacity
        assert alt.candidate.projector is True
        # Verify alternative request directly against engine
        alt_req = copy.deepcopy(req)
        alt_req.date = alt.date
        alt_req.start_hour = alt.start_hour
        eval_check = evaluate_room_request(alt_req, bundle, db_path=test_db_path)
        matching = [c for c in eval_check.candidates if c.room_id == alt.candidate.room_id]
        assert len(matching) > 0, f"Alternative room {alt.candidate.room_id} must be verified feasible"


def test_no_valid_alternatives_when_capacity_impossible(bundle: DatasetBundle, test_db_path: Path):
    """
    When attendance exceeds the maximum room capacity on campus (179 seats),
    no alternatives can exist, and the engine must return an empty list without crash.
    """
    req = RoomRequest(
        date=datetime.date(2025, 9, 15),
        start_hour=14.0,
        duration_hours=2.0,
        required_capacity=250, # impossible
    )
    alternatives = find_alternative_slots(req, bundle, db_path=test_db_path)
    assert len(alternatives) == 0, "No alternatives should exist when capacity is impossible"


def test_preserves_mandatory_constraints_in_alternatives(bundle: DatasetBundle, test_db_path: Path):
    """
    Ensure all physical constraints (computers, projector, accessibility)
    are strictly preserved during alternative slot exploration.
    """
    req = RoomRequest(
        date=datetime.date(2025, 9, 15),
        start_hour=10.0,
        duration_hours=2.0,
        required_capacity=40,
        projector_required=True,
        computers_required=True,
        accessible_required=True,
    )
    alternatives = find_alternative_slots(req, bundle, db_path=test_db_path, max_alternatives=5)
    
    for alt in alternatives:
        cand = alt.candidate
        assert cand.capacity >= 40, "Alternative capacity constraint violated"
        assert cand.projector is True, "Alternative projector constraint violated"
        assert cand.computers is True, "Alternative computers constraint violated"
        assert cand.accessible is True, "Alternative accessibility constraint violated"


# -----------------------------------------------------------------------------
# 2. PERSISTENT RESERVATION & CONCURRENCY TESTS
# -----------------------------------------------------------------------------

def test_confirm_valid_reservation(bundle: DatasetBundle, test_db_path: Path):
    """Verify creating a request and confirming a valid reservation."""
    req = RoomRequest(
        date=datetime.date(2025, 9, 15),
        start_hour=14.0,
        duration_hours=2.0,
        required_capacity=50,
        purpose="Datathon Workshop",
    )
    req_id = create_room_request_record(req, db_path=test_db_path)
    assert req_id.startswith("REQ-")
    
    # Confirm room R103 (capacity 51)
    success, msg, res_id = confirm_reservation(req_id, "R103", bundle=bundle, db_path=test_db_path)
    assert success is True
    assert res_id is not None
    assert res_id.startswith("RES-")
    
    # Verify in DB
    active = get_active_reservations(db_path=test_db_path)
    assert len(active) == 1
    assert active[0]["reservation_id"] == res_id
    assert active[0]["room_id"] == "R103"


def test_reject_overlapping_confirmed_reservation(bundle: DatasetBundle, test_db_path: Path):
    """
    Once a room is confirmed, any overlapping confirmation attempt for that room
    must be rejected atomically.
    """
    # 1. First booking: Room R103 from 14:00 to 16:00
    req1 = RoomRequest(
        date=datetime.date(2025, 9, 15),
        start_hour=14.0,
        duration_hours=2.0,
        required_capacity=40,
        purpose="First Booking",
    )
    req1_id = create_room_request_record(req1, db_path=test_db_path)
    succ1, _, res1_id = confirm_reservation(req1_id, "R103", bundle=bundle, db_path=test_db_path)
    assert succ1 is True

    # 2. Overlapping booking attempt: Room R103 from 15:00 to 17:00 (overlaps 15:00-16:00)
    req2 = RoomRequest(
        date=datetime.date(2025, 9, 15),
        start_hour=15.0,
        duration_hours=2.0,
        required_capacity=40,
        purpose="Conflicting Second Booking",
    )
    req2_id = create_room_request_record(req2, db_path=test_db_path)
    succ2, msg2, res2_id = confirm_reservation(req2_id, "R103", bundle=bundle, db_path=test_db_path)
    
    assert succ2 is False
    assert res2_id is None
    assert "booked by a concurrent reservation" in msg2 or "Conflict" in msg2


def test_confirmed_reservation_blocks_search(bundle: DatasetBundle, test_db_path: Path):
    """
    Confirmed reservations must immediately block subsequent search evaluations.
    """
    req = RoomRequest(
        date=datetime.date(2025, 9, 15),
        start_hour=14.0,
        duration_hours=2.0,
        required_capacity=50,
        purpose="Block Test",
    )
    # Confirm R103
    req_id = create_room_request_record(req, db_path=test_db_path)
    confirm_reservation(req_id, "R103", bundle=bundle, db_path=test_db_path)

    # Evaluate exact same slot
    res = evaluate_room_request(req, bundle, db_path=test_db_path)
    r103_cand = next((c for c in res.candidates if c.room_id == "R103"), None)
    assert r103_cand is None, "Confirmed room R103 should NOT be in feasible candidates"
    
    r103_rej = next((r for r in res.rejections if r.room_id == "R103"), None)
    assert r103_rej is not None
    assert any("Reservation conflict" in reason for reason in r103_rej.reasons)
    assert res.rejection_summary["reservation_conflict"] >= 1


def test_persistence_across_restarts(bundle: DatasetBundle, test_db_path: Path):
    """Verify that reservations persist across separate database connections."""
    req = RoomRequest(
        date=datetime.date(2025, 9, 15),
        start_hour=14.0,
        duration_hours=2.0,
        required_capacity=50,
        purpose="Persistent Booking",
    )
    # Choose a candidate proven feasible by the engine
    eval_res = evaluate_room_request(req, bundle, db_path=test_db_path)
    assert eval_res.feasible_count > 0
    chosen_room = eval_res.candidates[0].room_id

    req_id = create_room_request_record(req, db_path=test_db_path)
    succ, _, res_id = confirm_reservation(req_id, chosen_room, bundle=bundle, db_path=test_db_path)
    assert succ is True

    # Re-read directly using a fresh connection
    with sqlite3.connect(test_db_path) as conn:
        cur = conn.execute("SELECT reservation_id, status FROM reservations WHERE reservation_id = ?", (res_id,))
        row = cur.fetchone()
        assert row is not None
        assert row[0] == res_id
        assert row[1] == "ACTIVE"


def test_idempotent_duplicate_confirmation(bundle: DatasetBundle, test_db_path: Path):
    """
    Repeated confirmation calls on the same request_id (e.g. Streamlit rerun or double-click)
    must succeed idempotently without creating duplicate bookings.
    """
    req = RoomRequest(
        date=datetime.date(2025, 9, 15),
        start_hour=14.0,
        duration_hours=2.0,
        required_capacity=50,
        purpose="Idempotent Booking",
    )
    eval_res = evaluate_room_request(req, bundle, db_path=test_db_path)
    assert eval_res.feasible_count > 0
    chosen_room = eval_res.candidates[0].room_id

    req_id = create_room_request_record(req, db_path=test_db_path)
    
    # First confirmation
    succ1, _, res1_id = confirm_reservation(req_id, chosen_room, bundle=bundle, db_path=test_db_path)
    assert succ1 is True
    
    # Second confirmation with same request_id
    succ2, msg2, res2_id = confirm_reservation(req_id, chosen_room, bundle=bundle, db_path=test_db_path)
    assert succ2 is True
    assert res2_id == res1_id
    assert "already confirmed" in msg2

    # Check total rows in reservations table
    with sqlite3.connect(test_db_path) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM reservations WHERE request_id = ?", (req_id,))
        assert cur.fetchone()[0] == 1, "Must not create duplicate reservation rows"


def test_cancel_reservation_releases_room(bundle: DatasetBundle, test_db_path: Path):
    """
    Cancelling a reservation must release the room and allow subsequent bookings.
    """
    req = RoomRequest(
        date=datetime.date(2025, 9, 15),
        start_hour=14.0,
        duration_hours=2.0,
        required_capacity=50,
        purpose="Cancellation Test",
    )
    eval_res = evaluate_room_request(req, bundle, db_path=test_db_path)
    assert eval_res.feasible_count > 0
    chosen_room = eval_res.candidates[0].room_id

    req_id = create_room_request_record(req, db_path=test_db_path)
    succ, _, res_id = confirm_reservation(req_id, chosen_room, bundle=bundle, db_path=test_db_path)
    assert succ is True

    # Cancel reservation
    c_succ, c_msg = cancel_reservation(res_id, reason="Changed plans", db_path=test_db_path)
    assert c_succ is True

    # Verify chosen room is now feasible again
    eval_after = evaluate_room_request(req, bundle, db_path=test_db_path)
    released_cand = next((c for c in eval_after.candidates if c.room_id == chosen_room), None)
    assert released_cand is not None, f"Room {chosen_room} should be available after cancellation"

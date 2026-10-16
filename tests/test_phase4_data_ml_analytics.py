"""
SpaceSync — Phase 4 Automated Test Suite
DATATHON 2026 | VIT-AP University | Education Track

Validates:
1. Repeatable data loading, schema normalization, and in-memory indexing.
2. 100% agreement between optimized indexed evaluation and unindexed feasibility evaluation.
3. Batch evaluation of historical room requests with accurate metrics (latency, throughput, feasible-candidate coverage).
4. Time-slot aggregation and chronological train-test split without future leakage.
5. Reproducible ML demand forecasting evaluation against historical baselines.
6. Reconciled room utilization calculation using interval unions (preventing double-counting).
7. Explicit detection and non-dropping preservation of timetable anomalies.
"""

import datetime
from pathlib import Path
import sys
import pytest
import numpy as np
import pandas as pd

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import (
    AllocationCandidate,
    AllocationResult,
    DatasetBundle,
    RoomRequest,
    calculate_reconciled_utilization,
    evaluate_batch_historical_requests,
    evaluate_demand_forecasting,
    evaluate_room_request,
    get_admin_analytics_summary,
    load_and_validate_datasets,
    merge_intervals,
)


@pytest.fixture(scope="module")
def bundle():
    """Load and validate dataset bundle once for the test module."""
    return load_and_validate_datasets()


# -----------------------------------------------------------------------------
# 1. DATA LOADING & IN-MEMORY INDEXING
# -----------------------------------------------------------------------------

def test_data_loading_and_index_structure(bundle: DatasetBundle):
    """Verify that all 4 CSVs are loaded with expected schemas and in-memory indexes."""
    assert len(bundle.rooms) == 120
    assert len(bundle.class_schedule) == 5000
    assert len(bundle.room_requests) == 5000
    assert len(bundle.events) == 1800

    # In-memory indexes verified
    assert len(bundle.rooms_records) == 120
    assert len(bundle.classes_by_room_weekday) > 0
    assert len(bundle.events_by_room_date) > 0

    # Check index key types
    for key, class_list in bundle.classes_by_room_weekday.items():
        assert isinstance(key, tuple)
        assert len(key) == 2  # (room_id, weekday)
        assert isinstance(key[0], str)
        assert isinstance(key[1], str)
        assert isinstance(class_list, list)

    for key, event_list in bundle.events_by_room_date.items():
        assert isinstance(key, tuple)
        assert len(key) == 2  # (room_id, date)
        assert isinstance(key[0], str)
        assert isinstance(key[1], datetime.date)
        assert isinstance(event_list, list)


def test_timetable_anomalies_preserved(bundle: DatasetBundle):
    """Verify that known timetable collisions and overcapacities are detected, not silently cleaned."""
    audit = bundle.audit_summary
    assert audit["total_rooms"] == 120
    assert audit["class_overlap_pairs"] == 3426
    assert audit["classes_in_overlap"] == 3630
    assert audit["classes_in_overlap_pct"] == 72.6
    assert audit["classes_over_capacity"] == 2009
    assert audit["events_over_capacity"] == 722
    assert audit["events_cross_midnight"] == 190


# -----------------------------------------------------------------------------
# 2. OPTIMIZATION EQUIVALENCE & SCHEDULING CORRECTNESS
# -----------------------------------------------------------------------------

def test_indexed_and_unindexed_evaluation_agreement(bundle: DatasetBundle):
    """Verify that indexed evaluation produces identical results to unindexed evaluation across sample requests."""
    test_cases = [
        # Normal meeting
        RoomRequest(date=datetime.date(2025, 9, 15), start_hour=14.0, duration_hours=2.0, required_capacity=50, purpose="Meeting"),
        # Small workshop with projector
        RoomRequest(date=datetime.date(2025, 9, 16), start_hour=10.0, duration_hours=1.5, required_capacity=25, projector_required=True, purpose="Workshop"),
        # Large lecture with computers
        RoomRequest(date=datetime.date(2025, 9, 17), start_hour=9.0, duration_hours=3.0, required_capacity=100, computers_required=True, purpose="Class"),
        # Impossible capacity (zero feasible candidates)
        RoomRequest(date=datetime.date(2025, 9, 18), start_hour=11.0, duration_hours=1.0, required_capacity=250, purpose="All Hands"),
        # Sunday request (no classes)
        RoomRequest(date=datetime.date(2025, 9, 21), start_hour=15.0, duration_hours=2.0, required_capacity=30, purpose="Club"),
    ]

    for req in test_cases:
        # Standard evaluate_room_request uses indexed bundle
        indexed_res = evaluate_room_request(req, bundle)

        # Create mock unindexed bundle without in-memory indexes to force fallback
        unindexed_bundle = DatasetBundle(
            rooms=bundle.rooms,
            class_schedule=bundle.class_schedule,
            room_requests=bundle.room_requests,
            events=bundle.events,
            audit_summary=bundle.audit_summary,
            rooms_records=[],
            classes_by_room_weekday={},
            events_by_room_date={},
        )
        fallback_res = evaluate_room_request(req, unindexed_bundle)

        assert indexed_res.feasible_count == fallback_res.feasible_count
        assert [c.room_id for c in indexed_res.candidates] == [c.room_id for c in fallback_res.candidates]
        assert indexed_res.rejection_summary == fallback_res.rejection_summary
        if indexed_res.candidates:
            assert indexed_res.candidates[0].fit_score == fallback_res.candidates[0].fit_score


# -----------------------------------------------------------------------------
# 3. BATCH HISTORICAL REQUEST EVALUATION
# -----------------------------------------------------------------------------

def test_batch_historical_requests_evaluation(bundle: DatasetBundle):
    """Verify that batch evaluation of historical requests processes correctly and measures performance."""
    # Test batch evaluation with limit=200 for fast CI testing
    result = evaluate_batch_historical_requests(bundle, limit=200)

    assert result["total_requests"] == 200
    assert result["processed_requests"] == 200
    assert result["throughput_req_per_sec"] > 50  # Must be fast (>50 req/s, typically >200 req/s)
    assert result["median_latency_ms"] < 20.0     # Median latency well under 20ms
    assert result["feasible_count"] + result["zero_feasible_count"] == 200
    assert 0.0 <= result["feasible_pct"] <= 100.0
    assert "capacity_shortfall" in result["rejection_reason_distribution"]
    assert "class_conflict" in result["rejection_reason_distribution"]
    assert result["mean_fit_score"] > 50.0


# -----------------------------------------------------------------------------
# 4. INTERVAL MERGING & RECONCILED UTILIZATION
# -----------------------------------------------------------------------------

def test_merge_intervals_arithmetic():
    """Verify interval union logic correctly merges overlapping, contiguous, and disjoint intervals."""
    # Overlapping intervals: [9, 11] and [10, 12] -> [9, 12] (3 hours, not 4)
    res = merge_intervals([(9.0, 11.0), (10.0, 12.0)])
    assert res == [(9.0, 12.0)]

    # Contiguous intervals: [9, 10] and [10, 11] -> [9, 11]
    res = merge_intervals([(9.0, 10.0), (10.0, 11.0)])
    assert res == [(9.0, 11.0)]

    # Disjoint intervals: [9, 10] and [14, 16] -> unchanged
    res = merge_intervals([(9.0, 10.0), (14.0, 16.0)])
    assert res == [(9.0, 10.0), (14.0, 16.0)]

    # Multiple overlapping intervals
    res = merge_intervals([(8.0, 10.0), (9.0, 12.0), (11.0, 13.0), (15.0, 17.0)])
    assert res == [(8.0, 13.0), (15.0, 17.0)]


def test_reconciled_utilization_safeguard(bundle: DatasetBundle):
    """Verify that utilization calculation exposes the double-counting inflation from timetable collisions."""
    util = calculate_reconciled_utilization(bundle)

    assert util["total_campus_rooms"] == 120
    assert util["weekly_available_hours"] == 8640.0
    assert util["raw_class_hours"] == 7488.0
    assert util["raw_utilization_pct"] == 86.7
    assert util["reconciled_occupied_hours"] == 4682.0
    assert util["reconciled_utilization_pct"] == 54.2
    assert util["inflation_hours"] == 2806.0
    assert util["inflation_pct"] == 32.5

    # Check breakdown tables
    assert len(util["by_building"]) == 5
    assert len(util["by_room_type"]) == 5
    for bldg in util["by_building"]:
        assert bldg["Raw Util %"] > bldg["Reconciled Util %"]
        assert bldg["Inflation %"] > 0.0


# -----------------------------------------------------------------------------
# 5. ML DEMAND FORECASTING EVALUATION & NO LEAKAGE
# -----------------------------------------------------------------------------

def test_demand_forecasting_no_future_leakage_and_metrics(bundle: DatasetBundle):
    """
    Verify chronological train/test split, absence of future leakage,
    and reproduction of empirical evaluation metrics.
    """
    ml_res = evaluate_demand_forecasting(bundle)

    assert "Hourly room request count" in ml_res["target"]
    assert ml_res["split_method"] == "Chronological 80/20 train-test split (Zero future leakage)"

    # Check train/test periods
    assert "2025-08-01" in ml_res["train_period"]
    assert "2025-11-04" in ml_res["train_period"]
    assert "2025-11-05" in ml_res["test_period"]
    assert "2025-11-28" in ml_res["test_period"]

    # Verify metrics reproduction
    # Baseline MAE rounds to 1.408
    assert abs(ml_res["baseline_mae"] - 1.4078) < 0.001
    # Random Forest MAE rounds to 1.484 (worse than baseline)
    assert abs(ml_res["rf_mae"] - 1.4839) < 0.001
    assert ml_res["rf_mae"] > ml_res["baseline_mae"]
    # Poisson MAE shows modest improvement
    assert abs(ml_res["poisson_mae"] - 1.3913) < 0.001
    assert ml_res["poisson_mae"] < ml_res["baseline_mae"]

    # Verify metrics table structure
    models_evaluated = [row["Model"] for row in ml_res["metrics_table"]]
    assert any("Baseline" in m for m in models_evaluated)
    assert any("Random Forest" in m for m in models_evaluated)
    assert any("Poisson" in m for m in models_evaluated)

    # Predictions dataframe checks
    pred_df = ml_res["test_predictions"]
    assert len(pred_df) == 312  # 24 test days * 13 hours
    assert "request_count" in pred_df.columns
    assert "pred_baseline" in pred_df.columns
    assert "pred_rf" in pred_df.columns
    assert "pred_poisson" in pred_df.columns


# -----------------------------------------------------------------------------
# 6. ADMIN ANALYTICS CONSOLIDATION
# -----------------------------------------------------------------------------

def test_admin_analytics_summary_consolidation(bundle: DatasetBundle):
    """Verify that get_admin_analytics_summary aggregates all sections cleanly without error."""
    summary = get_admin_analytics_summary(bundle)

    assert "inventory" in summary
    assert "demand_patterns" in summary
    assert "timetable_integrity" in summary
    assert "utilization" in summary
    assert "batch_benchmark" in summary
    assert "ml_experiment" in summary

    inv = summary["inventory"]
    assert inv["total_rooms"] == 120
    assert inv["projector_pct"] == 78.3
    assert inv["computers_pct"] == 37.5
    assert inv["accessible_pct"] == 83.3

    demand = summary["demand_patterns"]
    assert demand["total_requests"] == 5000
    assert demand["status_distribution"]["Approved"] == 2723
    assert demand["status_distribution"]["Waitlisted"] == 939
    assert demand["status_distribution"]["Cancelled"] == 718
    assert demand["status_distribution"]["Rejected"] == 620

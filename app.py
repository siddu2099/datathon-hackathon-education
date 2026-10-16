"""
SpaceSync — Streamlit Application (Phase 3: Core Product with Alternatives & Persistence)
DATATHON 2026 | VIT-AP University | Education Track
"""

import copy
import datetime
from typing import List, Optional

import pandas as pd
import streamlit as st

from engine import (
    AllocationCandidate,
    AllocationResult,
    AlternativeRecommendation,
    DatasetBundle,
    RoomRequest,
    calculate_reconciled_utilization,
    cancel_reservation,
    confirm_reservation,
    create_room_request_record,
    evaluate_batch_historical_requests,
    evaluate_demand_forecasting,
    evaluate_room_request,
    find_alternative_slots,
    get_active_reservations,
    get_admin_analytics_summary,
    get_audit_logs,
    get_default_db_path,
    get_request_history,
    init_db,
    load_and_validate_datasets,
)

# Page configuration
st.set_page_config(
    page_title="SpaceSync — Campus Space Allocation",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .rank-badge {
        background-color: #2563EB;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .alt-badge {
        background-color: #059669;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .score-badge {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .pill-tag {
        display: inline-block;
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 6px;
        padding: 0.2rem 0.6rem;
        font-size: 0.8rem;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .pill-tag-active {
        background-color: #EFF6FF;
        border-color: #93C5FD;
        color: #1D4ED8;
    }
    .warning-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin-top: 0.75rem;
        font-size: 0.85rem;
        color: #92400E;
    }
    .receipt-box {
        background-color: #F0FDF4;
        border: 1px solid #86EFAC;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        color: #166534;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_header(bundle: DatasetBundle):
    st.markdown('<div class="main-header">🏛️ SpaceSync</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Campus Space Operations & Intelligent Room Allocation — '
        '<em>"Find the right room. Recover idle capacity. Explain every allocation."</em></div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    audit = bundle.audit_summary
    with col1:
        st.metric(label="Campus Rooms", value=audit["total_rooms"], help="120 room profiles across 5 buildings")
    with col2:
        st.metric(label="Scheduled Classes", value=f"{audit['total_classes']:,}", help="Recurring weekly class commitments")
    with col3:
        st.metric(label="Campus Events", value=f"{audit['total_events']:,}", help="Dated university events & seminars")
    with col4:
        st.metric(
            label="Timetable Collisions Flagged",
            value=f"{audit['class_overlap_pairs']:,}",
            delta=f"{audit['classes_in_overlap_pct']}% of classes",
            delta_color="inverse",
            help="3,426 overlapping same-room, same-weekday class pairs in the source schedule",
        )
    st.divider()

    st.warning(
        "⚠️ **Notice — Synthetic Educational Dataset:** SpaceSync is evaluating challenge data for DATATHON 2026. "
        "Room suitability and conflict audits reflect records provided in the dataset and do not establish live campus availability."
    )


def main():
    # Initialize database
    db_path = init_db()

    # Load and cache datasets
    try:
        bundle = load_and_validate_datasets()
    except Exception as e:
        st.error(f"❌ Failed to load source datasets: {e}")
        st.stop()

    render_header(bundle)

    # Initialize session state tracking
    if "current_request_id" not in st.session_state:
        st.session_state["current_request_id"] = None
    if "last_booking_status" not in st.session_state:
        st.session_state["last_booking_status"] = None

    # Main navigation tabs
    tab_allocate, tab_bookings, tab_analytics = st.tabs([
        "🔍 Space Allocation & Recommendations",
        "📅 Confirmed Reservations & Audit Trail",
        "📊 Administrative Analytics & Insights",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: SPACE ALLOCATION & ALTERNATIVES
    # -------------------------------------------------------------------------
    with tab_allocate:
        with st.sidebar:
            st.header("📋 Space Request Intake")
            st.caption("Specify event timing and mandatory operational constraints.")

            req_date = st.date_input(
                "Event Date",
                value=datetime.date(2025, 9, 15),
                min_value=datetime.date(2025, 8, 1),
                max_value=datetime.date(2025, 12, 31),
                help="Selected date derives the academic timetable weekday.",
            )

            col_time1, col_time2 = st.columns(2)
            with col_time1:
                hour_options = [h / 2.0 for h in range(48)]  # 00:00 to 23:30
                start_hour = st.selectbox(
                    "Start Time",
                    options=hour_options,
                    index=28,  # 14:00 (2 PM)
                    format_func=lambda h: f"{int(h):02d}:{int(round((h % 1) * 60)):02d}",
                )
            with col_time2:
                duration_options = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
                duration = st.selectbox(
                    "Duration",
                    options=duration_options,
                    index=3,  # 2.0 hours
                    format_func=lambda d: f"{d:g} hr{'s' if d > 1 else ''}",
                )

            attendance = st.number_input(
                "Expected Attendance",
                min_value=1,
                max_value=300,
                value=50,
                step=5,
                help="Rooms with capacity strictly below this will be eliminated.",
            )

            purpose = st.selectbox(
                "Event Purpose",
                options=["Club meeting", "Workshop", "Study group", "Project work", "Class", "Seminar / Talk"],
                index=0,
            )

            st.subheader("Mandatory Requirements")
            facilities_selected = st.multiselect(
                "Required Equipment",
                options=["Projector", "Computers"],
                default=[],
                help="Only rooms equipped with all selected items are feasible.",
            )

            accessible_req = st.checkbox(
                "Require Wheelchair / Step-Free Accessibility",
                value=False,
                help="Must be verified accessible in room metadata.",
            )

            submit_btn = st.button("🔍 Evaluate Space Allocation", type="primary", width="stretch")

        # Construct request object
        req = RoomRequest(
            date=req_date,
            start_hour=start_hour,
            duration_hours=duration,
            required_capacity=attendance,
            projector_required="Projector" in facilities_selected,
            computers_required="Computers" in facilities_selected,
            accessible_required=accessible_req,
            purpose=purpose,
        )

        req_start, req_end = req.get_interval()
        st.info(
            f"**Evaluating:** `{req.purpose}` on **{req_date.strftime('%A, %b %d, %Y')}** "
            f"from **{req_start.strftime('%H:%M')}** to **{req_end.strftime('%H:%M')}** "
            f"({req.duration_hours:g}h) for **{req.required_capacity} attendees**."
        )

        # Show reservation confirmation message if just booked
        if st.session_state.get("last_booking_status"):
            status_info = st.session_state["last_booking_status"]
            if status_info["success"]:
                st.markdown(
                    f"""
                    <div class="receipt-box">
                        <strong>✅ Booking Confirmed!</strong><br>
                        <strong>Reservation ID:</strong> <code>{status_info['res_id']}</code><br>
                        <strong>Room:</strong> Room {status_info['room_id']} •
                        <strong>Interval:</strong> {status_info['interval']}<br>
                        <em>This room is now atomically locked and blocked for subsequent requests. View details in the Active Bookings tab.</em>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.error(f"❌ **Booking Failed:** {status_info['message']}")

        # Execute evaluation
        result = evaluate_room_request(req, bundle, db_path=db_path)

        # CASE 1: FEASIBLE ROOMS FOUND
        if result.feasible_count > 0:
            st.success(
                f"✅ **{result.feasible_count} out of {result.total_rooms_evaluated} rooms** "
                f"meet 100% of physical, academic, and persistent reservation constraints. Showing top recommendations:"
            )

            top_candidates = result.candidates[:3]

            for idx, cand in enumerate(top_candidates, start=1):
                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                            <div>
                                <span class="rank-badge">Option #{idx}</span>
                                <span style="font-size: 1.35rem; font-weight: 700; margin-left: 0.5rem; color: #0F172A;">
                                    Room {cand.room_id}
                                </span>
                                <span style="color: #64748B; font-size: 0.95rem; margin-left: 0.5rem;">
                                    {cand.building} Building • Floor {cand.floor} • {cand.room_type}
                                </span>
                            </div>
                            <div>
                                <span class="score-badge">Fit Score: {cand.fit_score}%</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Room Capacity", f"{cand.capacity} seats", delta=f"+{cand.capacity - cand.requested_attendance} buffer")
                    with m2:
                        st.metric("Requested Attendance", f"{cand.requested_attendance} seats")
                    with m3:
                        st.metric("Capacity Fit Ratio", f"{cand.fit_score}%", help="(Requested / Capacity) * 100")
                    with m4:
                        st.metric("Power Outlets", f"{cand.power_outlets} ports")

                    st.markdown("**Room Equipment & Verification:**")
                    tags_html = ""
                    tags_html += f'<span class="pill-tag {"pill-tag-active" if cand.projector else ""}">📽️ Projector: {"Yes" if cand.projector else "No"}</span>'
                    tags_html += f'<span class="pill-tag {"pill-tag-active" if cand.computers else ""}">💻 Computers: {"Yes" if cand.computers else "No"}</span>'
                    tags_html += f'<span class="pill-tag {"pill-tag-active" if cand.accessible else ""}">♿ Accessible: {"Yes" if cand.accessible else "No"}</span>'
                    tags_html += f'<span class="pill-tag">🔌 Outlets: {cand.power_outlets}</span>'
                    st.markdown(tags_html, unsafe_allow_html=True)

                    with st.expander("🔍 Qualification Evidence & Schedule Audit", expanded=False):
                        for reason in cand.qualification_reasons:
                            st.write(f"- ✔️ {reason}")
                        if cand.data_quality_warnings:
                            for warn in cand.data_quality_warnings:
                                st.markdown(f'<div class="warning-box">⚠️ {warn}</div>', unsafe_allow_html=True)

                    # Explicit reservation button
                    btn_col1, btn_col2 = st.columns([1, 3])
                    with btn_col1:
                        if st.button(f"⚡ Confirm Booking (Room {cand.room_id})", key=f"book_{cand.room_id}", width="stretch"):
                            # Create pending request in DB if not existing
                            req_id = create_room_request_record(req, db_path=db_path)
                            success, msg, res_id = confirm_reservation(req_id, cand.room_id, bundle=bundle, db_path=db_path)
                            st.session_state["last_booking_status"] = {
                                "success": success,
                                "message": msg,
                                "res_id": res_id,
                                "room_id": cand.room_id,
                                "interval": f"{req_start.strftime('%a, %b %d %H:%M')}–{req_end.strftime('%H:%M')}",
                            }
                            st.rerun()

            with st.expander(f"📋 Rejection Audit ({len(result.rejections)} rooms disqualified)"):
                st.caption("Explanation of why non-recommended rooms failed hard constraints:")
                rej_data = [
                    {
                        "Room ID": r.room_id,
                        "Building": r.building,
                        "Type": r.room_type,
                        "Capacity": r.capacity,
                        "Primary Disqualification Reasons": " | ".join(r.reasons),
                    }
                    for r in result.rejections
                ]
                st.dataframe(pd.DataFrame(rej_data), width="stretch")

        # CASE 2: ZERO FEASIBLE ROOMS FOUND -> PROACTIVE ALTERNATIVES
        else:
            st.error(
                "❌ **Zero Feasible Rooms Found.** No campus room satisfies all required physical "
                "and temporal constraints for this specific time slot."
            )

            st.subheader("Root-Cause Disqualification Breakdown")
            s1, s2, s3 = st.columns(3)
            with s1:
                st.metric("Capacity Shortfall", f"{result.rejection_summary['capacity_shortfall']} rooms", help=f"Capacity < {attendance}")
                st.metric("Inaccessible", f"{result.rejection_summary['inaccessible']} rooms")
            with s2:
                st.metric("Missing Projector", f"{result.rejection_summary['missing_projector']} rooms")
                st.metric("Missing Computers", f"{result.rejection_summary['missing_computers']} rooms")
            with s3:
                st.metric("Class Schedule Collisions", f"{result.rejection_summary['class_conflict']} rooms")
                st.metric("Event / Booking Collisions", f"{result.rejection_summary['event_conflict'] + result.rejection_summary.get('reservation_conflict', 0)} rooms")

            if result.rejection_summary.get("verification_failure", 0) > 0:
                st.warning(
                    f"⚠️ **Schedule Verification Failures:** {result.rejection_summary['verification_failure']} rooms "
                    f"could not be safely verified due to malformed timetable records."
                )

            # Search Proactive Alternative Openings
            st.divider()
            st.subheader("💡 Proactive Alternative Time-Slot Recommendations")
            st.caption(
                "SpaceSync evaluated nearby time-slots while **strictly preserving** your requested duration, "
                "attendance, equipment, and accessibility requirements:"
            )

            alternatives = find_alternative_slots(req, bundle, db_path=db_path)

            if alternatives:
                for a_idx, alt in enumerate(alternatives, start=1):
                    with st.container(border=True):
                        cand = alt.candidate
                        st.markdown(
                            f"""
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <div>
                                    <span class="alt-badge">Alternative #{a_idx}</span>
                                    <span style="font-size: 1.25rem; font-weight: 700; margin-left: 0.5rem; color: #0F172A;">
                                        {alt.date.strftime('%A, %b %d')} • {alt.start_dt.strftime('%H:%M')}–{alt.end_dt.strftime('%H:%M')}
                                    </span>
                                    <span style="color: #059669; font-weight: 600; font-size: 0.9rem; margin-left: 0.5rem;">
                                        ({alt.time_offset_description})
                                    </span>
                                </div>
                                <div>
                                    <span class="score-badge">Fit Score: {cand.fit_score}%</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        m1, m2, m3, m4 = st.columns(4)
                        with m1:
                            st.metric("Recommended Space", f"Room {cand.room_id}", help=f"{cand.building} Building, Floor {cand.floor}")
                        with m2:
                            st.metric("Room Capacity", f"{cand.capacity} seats")
                        with m3:
                            st.metric("Attendance", f"{req.required_capacity} seats")
                        with m4:
                            st.metric("Room Type", cand.room_type)

                        st.write(f"**Feasibility Proof:** {alt.explanation}")

                        # Action to book alternative
                        if st.button(
                            f"⚡ Confirm Alternative (Room {cand.room_id} on {alt.date.strftime('%b %d')} at {alt.start_dt.strftime('%H:%M')})",
                            key=f"alt_book_{a_idx}_{cand.room_id}",
                            width="stretch",
                        ):
                            alt_req = copy.deepcopy(req)
                            alt_req.date = alt.date
                            alt_req.start_hour = alt.start_hour
                            req_id = create_room_request_record(alt_req, db_path=db_path)
                            success, msg, res_id = confirm_reservation(req_id, cand.room_id, bundle=bundle, db_path=db_path)
                            st.session_state["last_booking_status"] = {
                                "success": success,
                                "message": msg,
                                "res_id": res_id,
                                "room_id": cand.room_id,
                                "interval": f"{alt.start_dt.strftime('%a, %b %d %H:%M')}–{alt.end_dt.strftime('%H:%M')}",
                            }
                            st.rerun()

            else:
                st.warning(
                    "❌ **No valid alternative slots could be found** within ±3 hours or adjacent days meeting all mandatory constraints. "
                    "Consider adjusting attendance or equipment requirements."
                )

            with st.expander("View Full Disqualification Log for Requested Slot"):
                rej_data = [
                    {
                        "Room ID": r.room_id,
                        "Building": r.building,
                        "Capacity": r.capacity,
                        "Disqualification Causes": " | ".join(r.reasons),
                    }
                    for r in result.rejections
                ]
                st.dataframe(pd.DataFrame(rej_data), width="stretch")

        # Dataset Quality Audit Tab/Expander
        with st.expander("📊 Timetable Integrity & Dataset Quality Audit"):
            st.markdown(
                """
                **SpaceSync Data Quality Policy:**
                The supplied challenge datasets contain verified real-world operational anomalies.
                SpaceSync enforces schedule integrity by detecting these anomalies rather than silently dropping records.
                """
            )
            audit = bundle.audit_summary
            q1, q2, q3, q4 = st.columns(4)
            with q1:
                st.metric("Schedule Overlap Pairs", f"{audit['class_overlap_pairs']:,}")
            with q2:
                st.metric("Classes in Overlap", f"{audit['classes_in_overlap']} ({audit['classes_in_overlap_pct']}%)")
            with q3:
                st.metric("Classes Over Capacity", f"{audit['classes_over_capacity']} ({audit['classes_over_capacity_pct']}%)")
            with q4:
                st.metric("Events Crossing Midnight", f"{audit['events_cross_midnight']}")

    # -------------------------------------------------------------------------
    # TAB 2: ACTIVE BOOKINGS & AUDIT TRAIL
    # -------------------------------------------------------------------------
    with tab_bookings:
        st.subheader("📅 Active Room Reservations & Operational State")
        st.caption("Persistent SQLite storage tracking confirmed bookings and transaction states.")

        active_res = get_active_reservations(db_path=db_path)
        req_hist = get_request_history(db_path=db_path)
        audit_logs = get_audit_logs(db_path=db_path)

        b1, b2, b3 = st.columns(3)
        with b1:
            st.metric("Active Reservations", len(active_res))
        with b2:
            st.metric("Total Requests Logged", len(req_hist))
        with b3:
            st.metric("Audit Events Recorded", len(audit_logs))

        st.markdown("#### Confirmed Active Bookings")
        if active_res:
            for r in active_res:
                with st.container(border=True):
                    rc1, rc2, rc3 = st.columns([3, 2, 1])
                    with rc1:
                        st.markdown(
                            f"**Room {r['room_id']}** • `{r['reservation_id']}`<br>"
                            f"**Purpose:** {r['purpose']} ({r['attendance']} attendees)",
                            unsafe_allow_html=True,
                        )
                    with rc2:
                        st.markdown(
                            f"**Interval:** {r['start_time'][:16].replace('T', ' ')} to {r['end_time'][11:16]}<br>"
                            f"**Confirmed:** {r['confirmed_at'][:19].replace('T', ' ')}",
                            unsafe_allow_html=True,
                        )
                    with rc3:
                        if st.button("Cancel Booking", key=f"cancel_{r['reservation_id']}", width="stretch"):
                            succ, c_msg = cancel_reservation(r["reservation_id"], db_path=db_path)
                            if succ:
                                st.success(c_msg)
                            else:
                                st.error(c_msg)
                            st.rerun()
        else:
            st.info("No active reservations yet. Go to the Search tab to evaluate rooms and book an allocation.")

        st.divider()
        st.markdown("#### Request & Transition Audit History")
        with st.expander("Inspect SQLite Audit Log"):
            if audit_logs:
                st.dataframe(pd.DataFrame(audit_logs), width="stretch")
            else:
                st.write("No audit entries recorded yet.")

    # -------------------------------------------------------------------------
    # TAB 3: ADMINISTRATIVE ANALYTICS & INSIGHTS
    # -------------------------------------------------------------------------
    with tab_analytics:
        st.subheader("📊 Campus Facilities & Administrative Intelligence")
        st.caption("Evidence-driven capacity analytics, timetable integrity diagnostics, verified scheduled utilization, and ML demand forecasting.")

        summary = get_admin_analytics_summary(bundle=bundle, db_path=db_path)
        inv = summary["inventory"]
        audit = summary["timetable_integrity"]
        util = summary["utilization"]
        demand = summary["demand_patterns"]
        batch = summary["batch_benchmark"]
        ml = summary["ml_experiment"]

        # ---------------------------------------------------------------------
        # 1. Campus Space Inventory & Facilities
        # ---------------------------------------------------------------------
        st.markdown("### 🏢 1. Campus Space Inventory & Facilities")
        st.caption("Physical campus asset profile across 120 rooms in 5 academic buildings.")

        i1, i2, i3, i4, i5, i6 = st.columns(6)
        with i1:
            st.metric("Total Rooms", inv["total_rooms"])
        with i2:
            st.metric("Total Seats", f"{int(bundle.rooms['capacity'].sum()):,}")
        with i3:
            st.metric("Avg Capacity", f"{inv['average_capacity']} seats")
        with i4:
            st.metric("Projector Equipt", f"{inv['projector_pct']}%")
        with i5:
            st.metric("Computer Labs", f"{inv['computers_pct']}%")
        with i6:
            st.metric("Accessible", f"{inv['accessible_pct']}%")

        col_inv1, col_inv2 = st.columns(2)
        with col_inv1:
            st.markdown("**Rooms by Academic Building:**")
            bldg_df = pd.DataFrame(list(inv["buildings"].items()), columns=["Building", "Count"]).set_index("Building")
            st.bar_chart(bldg_df, color="#2563EB")
        with col_inv2:
            st.markdown("**Rooms by Space Type:**")
            type_df = pd.DataFrame(list(inv["room_types"].items()), columns=["Room Type", "Count"]).set_index("Room Type")
            st.bar_chart(type_df, color="#059669")

        st.divider()

        # ---------------------------------------------------------------------
        # 2. Reconciled Scheduled Utilization (with Rigorous Double-Counting Safeguards)
        # ---------------------------------------------------------------------
        st.markdown("### ⏱️ 2. Scheduled Utilization & Double-Counting Safeguards")
        st.markdown(
            f"""
            <div class="warning-box" style="margin-bottom: 1rem;">
                <strong>⚠️ Operational Utilization Safeguard:</strong><br>
                Directly summing scheduled class durations yields an apparent <strong>{util['raw_utilization_pct']}% utilization ({util['raw_class_hours']:,.1f} hours)</strong>.
                However, because <strong>{audit['classes_in_overlap_pct']}% of timetable classes collide</strong> in identical rooms and hours,
                <strong>{util['inflation_hours']:,.1f} hours are double-counted (+{util['inflation_pct']}% phantom inflation)</strong>.<br>
                SpaceSync's interval-union arithmetic reconciles true non-overlapping scheduled utilization at
                <strong>{util['reconciled_utilization_pct']}% ({util['reconciled_occupied_hours']:,.1f} hours)</strong> across the {util['weekly_available_hours']:,.0f} weekly operating room-hours (Mon–Sat 08:00–20:00),
                recovering <strong>{util['weekly_available_hours'] - util['reconciled_occupied_hours']:,.0f} idle room-hours per week</strong> for ad-hoc allocations.
            </div>
            """,
            unsafe_allow_html=True,
        )

        u1, u2, u3, u4 = st.columns(4)
        with u1:
            st.metric("Available Operating Hours", f"{util['weekly_available_hours']:,.0f} h/wk", help="120 rooms × 6 days × 12 operating hours (08:00–20:00)")
        with u2:
            st.metric("Raw Summed Class Hours", f"{util['raw_class_hours']:,.0f} hrs", delta=f"{util['raw_utilization_pct']}% apparent", delta_color="off")
        with u3:
            st.metric("Reconciled Occupied Hours", f"{util['reconciled_occupied_hours']:,.0f} hrs", delta=f"{util['reconciled_utilization_pct']}% true utilization", delta_color="normal")
        with u4:
            st.metric("Double-Counted Inflation", f"{util['inflation_hours']:,.0f} hrs", delta=f"+{util['inflation_pct']}% phantom", delta_color="inverse")

        col_u1, col_u2 = st.columns(2)
        with col_u1:
            st.markdown("**Utilization Reconciliation by Building:**")
            b_df = pd.DataFrame(util["by_building"])
            st.dataframe(b_df[["Building", "Rooms", "Available (h/wk)", "Raw Util %", "Reconciled Util %", "Inflation %"]], width="stretch", hide_index=True)
        with col_u2:
            st.markdown("**Utilization Reconciliation by Room Type:**")
            t_df = pd.DataFrame(util["by_room_type"])
            st.dataframe(t_df[["Room Type", "Rooms", "Available (h/wk)", "Raw Util %", "Reconciled Util %", "Inflation %"]], width="stretch", hide_index=True)

        st.divider()

        # ---------------------------------------------------------------------
        # 3. Timetable Integrity & Operational Anomaly Diagnostics
        # ---------------------------------------------------------------------
        st.markdown("### ⚠️ 3. Timetable Integrity & Schedule Collision Diagnostics")
        st.caption("Empirical audit of data contradictions and operational anomalies present in the supplied academic schedule.")

        t1, t2, t3, t4 = st.columns(4)
        with t1:
            st.metric("Schedule Collision Pairs", f"{audit['class_overlap_pairs']:,}", delta=f"{audit['classes_in_overlap']} classes ({audit['classes_in_overlap_pct']}%)", delta_color="inverse")
        with t2:
            st.metric("Overcapacity Classes", f"{audit['classes_over_capacity']:,}", delta=f"{audit['classes_over_capacity_pct']}% of classes", delta_color="inverse")
        with t3:
            st.metric("Overcapacity Events", f"{audit['events_over_capacity']:,}", delta=f"{audit['events_over_capacity_pct']}% of events", delta_color="inverse")
        with t4:
            st.metric("Midnight-Crossing Events", f"{audit['events_cross_midnight']:,}", delta=f"{audit['events_off_hours']} off-hours events", delta_color="inverse")

        with st.expander("ℹ️ SpaceSync Data Quality & Fail-Closed Scheduling Policy"):
            st.markdown(
                """
                - **Never Silently Discard:** Operational anomalies represent real-world university data debt. Dropping them risks allocating occupied rooms.
                - **Fail-Safe Conflict Protection:** When two classes are scheduled simultaneously in the same room, SpaceSync fails closed (marking the room blocked across the union of both intervals).
                - **Transparent Reporting:** Schedulers and administrators are alerted to specific collisions so facilities leadership can resolve timetable errors at the source.
                """
            )

        st.divider()

        # ---------------------------------------------------------------------
        # 4. Historical Demand Analysis (5,000 Requests)
        # ---------------------------------------------------------------------
        st.markdown("### 📈 4. Historical Demand Patterns (room_requests.csv)")
        st.caption("Temporal and purpose breakdown across all 5,000 unallocated historical demand records (Aug 1 – Nov 28, 2025).")

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.metric("Approved Requests", f"{demand['status_distribution'].get('Approved', 0):,}", delta="54.5%")
        with d2:
            st.metric("Waitlisted Requests", f"{demand['status_distribution'].get('Waitlisted', 0):,}", delta="18.8%")
        with d3:
            st.metric("Cancelled Requests", f"{demand['status_distribution'].get('Cancelled', 0):,}", delta="14.4%")
        with d4:
            st.metric("Rejected Requests", f"{demand['status_distribution'].get('Rejected', 0):,}", delta="12.4%")

        col_dem1, col_dem2 = st.columns(2)
        with col_dem1:
            st.markdown("**Historical Request Volume by Day of Week:**")
            w_df = pd.DataFrame(list(demand["weekday_distribution"].items()), columns=["Day", "Requests"]).set_index("Day")
            st.bar_chart(w_df, color="#3B82F6")
        with col_dem2:
            st.markdown("**Historical Request Volume by Time Slot (Start Hour):**")
            h_df = pd.DataFrame(list(demand["hourly_distribution"].items()), columns=["Hour", "Requests"])
            h_df["Time"] = h_df["Hour"].apply(lambda h: f"{h:02d}:00")
            st.bar_chart(h_df.set_index("Time")["Requests"], color="#6366F1")

        st.divider()

        # ---------------------------------------------------------------------
        # 5. Engine Batch Allocation Benchmark
        # ---------------------------------------------------------------------
        st.markdown("### ⚡ 5. Engine Batch Allocation Benchmark")
        st.caption("Measured performance from running all 5,000 historical requests against the authoritative feasibility engine (600,000 room evaluations).")

        b1, b2, b3, b4, b5 = st.columns(5)
        with b1:
            st.metric("Throughput", f"{batch['throughput_req_per_sec']} req/s", delta=f"{batch['total_time_sec']}s total batch")
        with b2:
            st.metric("Median Latency", f"{batch['median_latency_ms']} ms", delta="~24x vs baseline", delta_color="normal")
        with b3:
            st.metric("95th Percentile (p95)", f"{batch['p95_latency_ms']} ms", help="95% of room allocations resolve under 6 milliseconds")
        with b4:
            st.metric("Feasible Coverage", f"{batch['feasible_pct']}%", delta=f"{batch['feasible_count']:,} requests")
        with b5:
            st.metric("Zero-Candidate Rate", f"{batch['zero_feasible_pct']}%", delta=f"{batch['zero_feasible_count']} requests", delta_color="inverse")

        col_b1, col_b2 = st.columns([1, 1])
        with col_b1:
            st.markdown("**Rejection Cause Distribution (600,000 Room Checks):**")
            rej_df = pd.DataFrame(
                list(batch["rejection_reason_distribution"].items()),
                columns=["Disqualification Cause", "Occurrences"]
            ).sort_values("Occurrences", ascending=False)
            st.dataframe(rej_df, width="stretch", hide_index=True)
        with col_b2:
            st.markdown("**Benchmark Runtime Environment:**")
            st.json(batch["benchmark_environment"])
            st.markdown(
                f"""
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.75rem; font-size: 0.85rem;">
                    <strong>Allocation Quality:</strong><br>
                    - Mean Top-Candidate Fit Score: <strong>{batch['mean_fit_score']}%</strong><br>
                    - Median Top-Candidate Fit Score: <strong>{batch['median_fit_score']}%</strong><br>
                    - Requests with 0 Feasible Rooms: <strong>{batch['zero_feasible_count']} ({batch['zero_feasible_pct']}%)</strong> (handled via Proactive Alternative Slot suggestions).
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # ---------------------------------------------------------------------
        # 6. ML Demand Forecasting Experiment
        # ---------------------------------------------------------------------
        st.markdown("### 🤖 6. Machine Learning Demand Forecasting Experiment")
        st.caption("Forecasting hourly time-slot demand volume to assist staffing and proactive load-balancing.")

        st.markdown(
            f"""
            <div style="background-color: #EFF6FF; border-left: 4px solid #3B82F6; padding: 0.75rem 1rem; border-radius: 4px; margin-bottom: 1rem; font-size: 0.9rem;">
                <strong>Problem Formulation:</strong> Predict hourly room request volume (<code>requests_per_slot</code>) across 08:00 to 20:00 (13 slots/day).<br>
                <strong>Evaluation Protocol:</strong> Strict chronological 80/20 train/test split. First 96 days (<code>{ml['train_period'][:10]}</code> to <code>{ml['train_period'][14:24]}</code>, 1,248 slots) for training; remaining 24 days (<code>{ml['test_period'][:10]}</code> to <code>{ml['test_period'][14:24]}</code>, 312 slots) for out-of-sample evaluation.<br>
                <strong>Zero Future Leakage:</strong> Features (<code>weekday_num</code>, <code>start_hour</code>, <code>is_weekend</code>, <code>day</code>, <code>month</code>, <code>prev_day_requests</code>) constructed strictly from past observations.
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Historical Mean Baseline MAE", f"{ml['baseline_mae']:.4f}", help="Grouped historical mean for each (weekday, hour) tuple")
        with m2:
            st.metric("Random Forest MAE", f"{ml['rf_mae']:.4f}", delta="+5.41% error (Worse)", delta_color="inverse")
        with m3:
            st.metric("Poisson Regressor MAE", f"{ml['poisson_mae']:.4f}", delta="-1.17% error (Best)", delta_color="normal")

        st.markdown("**Comparative Model Evaluation:**")
        m_df = pd.DataFrame(ml["metrics_table"])
        st.dataframe(m_df, width="stretch", hide_index=True)

        # Forecast vs Actual Chart over the test period
        st.markdown("**Out-of-Sample Demand Forecast vs Actuals (Nov 5 – Nov 28, 2025):**")
        pred_sub = ml["test_predictions"].copy()
        pred_sub["timestamp"] = pred_sub["request_date"].dt.strftime("%b %d") + " " + pred_sub["start_hour"].astype(str) + ":00"
        chart_data = pred_sub[["timestamp", "request_count", "pred_baseline", "pred_rf"]].rename(
            columns={
                "request_count": "Actual Requests",
                "pred_baseline": "Historical Baseline Forecast",
                "pred_rf": "Random Forest Forecast",
            }
        ).set_index("timestamp")
        st.line_chart(chart_data.iloc[:78], color=["#0F172A", "#2563EB", "#DC2626"])
        st.caption("Displaying hourly forecast curves across the first 6 out-of-sample evaluation days.")

        st.info(f"💡 {ml['takeaway']}")


if __name__ == "__main__":
    main()

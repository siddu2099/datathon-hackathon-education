# SpaceSync — Acceptance Criteria & Quality Gates

**Event:** DATATHON 2026, VIT-AP University  
**Track:** Education — Campus Space Optimization  
**Document Status:** Frozen (Phase 0)

---

## 1. Phase-by-Phase Acceptance Gates

Every phase must pass its strict acceptance gate before progressing to the next.

| Phase | Milestone | Acceptance Criteria | Verification Method |
| :--- | :--- | :--- | :--- |
| **Phase 0** | **Requirements & Context Freeze** | - Dataset files verified against disk.<br>- Empirical statistics reproduced.<br>- Core docs created (`PRODUCT_CONTEXT.md`, `ACCEPTANCE_CRITERIA.md`, `DECISIONS.md`).<br>- No application code created yet. | Inspection of disk and docs directory. |
| **Phase 1** | **Base Prototype (Round 1 MVP)** | - Exactly two files: `engine.py` and `app.py`.<br>- Successfully loads all 4 CSV datasets from disk.<br>- Interactive UI accepts date, time, attendance, facilities.<br>- Feasibility filter eliminates invalid rooms.<br>- Basic schedule conflict detection executes.<br>- Ranked cards display fit score and qualification reasons.<br>- Runs without unhandled crashes. | Live Streamlit run; manual end-to-end request test. |
| **Phase 2** | **Scheduling Correctness & Temporal Engine** | - Modularization of scheduling logic.<br>- Half-open interval overlap: `start_a < end_b and start_b < end_a`.<br>- Day-of-week derivation: `YYYY-MM-DD` mapped accurately to `Mon`-`Sat`.<br>- Dated event conflict checks with datetime parsing.<br>- Cross-midnight event handling.<br>- Data quality warnings surfaced for source collisions.<br>- Automated test suite covering temporal logic. | `pytest tests/test_scheduling.py` passes 100%. |
| **Phase 3** | **Ranking, Explanations & Alternatives** | - Deterministic capacity fit score: `(req / cap) * 100`.<br>- Hard constraint verification (facilities, accessibility).<br>- Detailed explanation per candidate (capacity, features, conflict check).<br>- Alternative slot discovery triggered when 0 rooms are feasible.<br>- Alternatives searched within ±1h to ±3h window.<br>- Alternatives obey identical feasibility rules without constraint relaxation. | `pytest tests/test_ranking_alternatives.py` passes 100%. |
| **Phase 4** | **Reservation Workflow & Persistence** | - SQLite database initialized with schema for reservations.<br>- Pre-confirmation conflict check inside database transaction.<br>- Atomic commit prevents concurrent double-booking of same room/slot.<br>- UI confirmation button is idempotent against Streamlit script reruns.<br>- Confirmed bookings instantly block subsequent room searches. | `pytest tests/test_reservations.py` passes 100%. |
| **Phase 5** | **Machine Learning & Demand Forecasting** | - Time-slot demand forecasting formulated cleanly (requests per slot).<br>- Chronological train/test split (no data leakage).<br>- Evaluated against naive historical slot-mean baseline.<br>- MAE / RMSE computed and reported honestly.<br>- Clear disclosure of whether ML beats baseline.<br>- ML does NOT override deterministic scheduling. | Reproducible evaluation script and test report. |
| **Phase 6** | **Administrative Operations & Analytics** | - Timetable integrity audit dashboard (overlap pairs, overcapacity records).<br>- Spatial demand breakdown by purpose, hour, and day.<br>- Reconciled room utilization metrics (room-hours occupied vs. available, no double-counting).<br>- Active reservation manager (view, cancel). | Visual verification and mathematical audit of metrics. |
| **Phase 7** | **Integration, Hardening & Final Freeze** | - Complete automated test suite passes with zero failures.<br>- End-to-end user workflow validated (search -> explain -> alternative -> reserve -> block).<br>- Clean startup (`streamlit run app.py`).<br>- Code frozen prior to 4:00 PM deadline. | Full regression test suite + live demo dry-run. |

---

## 2. Scheduling Correctness Criteria

1. **Interval Arithmetic:**
   - Two intervals $[S_1, E_1)$ and $[S_2, E_2)$ conflict if and only if:
     $$S_1 < E_2 \quad \text{AND} \quad S_2 < E_1$$
   - Adjacent intervals (e.g., 09:00–10:00 and 10:00–11:00) **do not** conflict.
2. **Temporal Alignment:**
   - Recurring class schedules use 3-letter uppercase/title weekdays: `['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']`.
   - Date inputs (e.g., `2025-09-15`) must derive the matching weekday (`Mon`).
   - Sunday requests must safely evaluate against 0 recurring classes.
3. **Event Timestamps:**
   - Events have full `YYYY-MM-DD HH:MM:SS` timestamps.
   - Matching against a request on date $D$ starting at hour $H$ for duration $L$ requires converting both to comparable datetime windows $[D + H, D + H + L)$.
   - Events crossing midnight must span across both calendar dates correctly.
4. **Data Contradiction Handling:**
   - When the underlying dataset contains two classes in Room $R$ at the same hour, Room $R$ is marked **unavailable** during that hour. The engine must never guess which class is valid; it must fail closed (treat occupied) and flag the conflict.

---

## 3. Feasibility & Ranking Correctness Criteria

1. **Hard Constraints (Zero Tolerance):**
   - If `room.capacity < request.required_capacity` $\rightarrow$ **REJECT**.
   - If `request.projector_required == 1` and `room.projector == 0` $\rightarrow$ **REJECT**.
   - If `request.computers_required == 1` and `room.computers == 0` $\rightarrow$ **REJECT**.
   - If `request.accessible_required == 1` and `room.accessible == 0` $\rightarrow$ **REJECT**.
   - If room has an overlapping class on that weekday $\rightarrow$ **REJECT**.
   - If room has an overlapping event on that date/time $\rightarrow$ **REJECT**.
   - If room has a confirmed reservation in SQLite $\rightarrow$ **REJECT**.
2. **Ranking Formula:**
   - Candidate rooms are ordered descending by:
     $$\text{capacity\_fit\_score} = \left(\frac{\text{required\_capacity}}{\text{room\_capacity}}\right) \times 100$$
   - Ties broken deterministically by: `(room.capacity ASC, room.power_outlets DESC, room.room_id ASC)`.
   - A 60-person room ranks higher than a 150-person room for a 55-person request.

---

## 4. Reservation Safety & Concurrency Criteria

1. **Atomic Booking:**
   - Reservations must be persisted to SQLite using an explicit transaction with an immediate lock (`BEGIN IMMEDIATE`).
   - A unique constraint on `(room_id, date, start_hour, duration_hours)` or a transactional collision check must precede insertion.
2. **Re-Verification at Confirmation:**
   - When the user clicks "Confirm Reservation", the engine must re-execute the full feasibility check against the database before committing.
   - If another user confirmed the room while the first user was reviewing recommendations, the confirmation must cleanly abort with: `"Room was booked by a concurrent user. Please choose another candidate."`
3. **Streamlit Lifecycle Protection:**
   - Form submissions and button clicks must utilize session state flags so that page refreshes or widget interactions do not trigger duplicate database inserts.

---

## 5. Machine Learning Evaluation Criteria

1. **Defensible Problem Formulation:**
   - Task: Predict hourly campus request volume (demand forecasting) to aid staffing and alternative time-slot recommendations.
2. **Target and Feature Hygiene:**
   - Group historical requests by `(request_date, start_hour)` to compute `request_count`.
   - Features allowed: `weekday`, `start_hour`, `day_of_month`, `month`, `is_weekend`, `lagged_rolling_demand`.
   - Strictly forbidden: Target leakage (using future request counts or future event data).
3. **Evaluation Protocol:**
   - Chronological split: First 80% of days for training, remaining 20% for testing.
   - Baseline: Historical mean requests for that `(weekday, start_hour)` tuple.
   - Metric: Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
   - Truthfulness Rule: If the trained ML model (e.g. Random Forest) fails to beat the baseline, this result must be documented and visualized as an empirical finding. No claims of artificial ML superiority.

---

## 6. Failure Modes & Edge Case Test Matrix

| ID | Test Scenario | Expected System Behavior |
| :--- | :--- | :--- |
| **TC-01** | Attendance exceeds maximum campus room capacity (e.g. 200). | Feasibility engine returns 0 candidates; explains capacity shortfall; suggests splitting or auditorium. |
| **TC-02** | Request on a Sunday (no classes scheduled). | Correctly checks only dated events and persistent reservations; does not crash on class lookup. |
| **TC-03** | Request during a slot where all matching rooms are occupied. | Triggers alternative slot engine; returns feasible times on the same date or adjacent dates. |
| **TC-04** | Cross-midnight event overlapping a late-evening request (e.g., event 23:00–02:00 vs. request 22:00–24:00). | Correctly detects overlap across date boundary and marks room unavailable. |
| **TC-05** | Requested interval is adjacent to an existing booking (e.g. 10:00–11:00 when booking is 11:00–12:00). | Correctly evaluates as feasible (no overlap). |
| **TC-06** | Invalid duration (<= 0) or end_hour > 24. | Input validation error shown; prevents engine execution. |
| **TC-07** | Double-click on "Confirm Booking" button in Streamlit. | First click commits; second click detects idempotency token or conflict; does not duplicate rows. |
| **TC-08** | Timetable anomaly in source dataset (room has 2 classes at same hour). | Safe fail-closed: room blocked; audit log warns of timetable contradiction. |

---

## 7. Final Release Gate (Phase 7 / 4:00 PM Freeze)

Before declaring the submission ready for presentation and evaluation:
1. `pytest` runs cleanly with 100% passing tests.
2. The Streamlit application starts with zero warnings or deprecation errors: `streamlit run app.py`.
3. An end-to-end reservation can be searched, explained, booked, and verified as blocking subsequent searches.
4. An end-to-end alternative search scenario succeeds on an overbooked slot.
5. The Admin Analytics view correctly displays dataset integrity statistics matching verified source numbers (3,426 class collisions, 2,009 overcapacity classes).
6. All documentation is committed and git status is clean.

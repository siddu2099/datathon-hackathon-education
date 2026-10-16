# SpaceSync — Product Context and Scope Freeze

**Event:** DATATHON 2026, VIT-AP University  
**Track:** Education — Campus Space Optimization  
**Date:** October 16, 2026  
**Document Status:** Frozen (Phase 0)

---

## 1. Problem Statement

Campus spaces (classrooms, labs, seminar halls, auditoriums, and study rooms) are among the most expensive physical capital assets in higher education. Despite perceived scarcity—where students, clubs, and event organizers face chronic room shortages and waitlists—real-world campuses frequently suffer from artificial shortages driven by:
1. **Siloed Records:** Academic recurring schedules, one-off university events, and ad-hoc student requests are managed in disconnected systems.
2. **Opaque Allocation:** Requests are approved or rejected without transparent criteria, leading to mismatched room sizes (e.g., small workshops in 150-seat auditoriums).
3. **Dead-End Rejections:** When a preferred slot is unavailable, requesters receive rejections with no actionable alternative recommendations.
4. **Data Contradictions:** Legacy timetable data often contains unresolvable schedule collisions, invalid capacities, and phantom bookings that erode trust.

### The Central Product Question
> *"Given a room request, what is the best feasible room, what verified evidence supports the recommendation, and what actionable alternative should be offered if no suitable room exists?"*

---

## 2. Product Vision & Target Users

**SpaceSync** is an explainable campus-space operations and intelligent allocation platform. It acts as a decision-support and capacity-recovery engine rather than a passive calendar or booking form.

### Product Positioning
> **"Find the right room. Recover idle capacity. Explain every allocation."**

### Target Personas & Use Cases
| Persona | Key Need | SpaceSync Value |
| :--- | :--- | :--- |
| **Student / Club Lead** | Fast room discovery for club meetings, workshops, hackathons. | Instant feasibility matching, transparent criteria, and proactive alternative slot suggestions. |
| **Faculty / Event Organizer** | Guaranteed room suitability with required equipment (projectors, PCs, accessibility). | Deterministic constraint enforcement; proof that equipment and accessibility exist. |
| **Campus Facilities Scheduler** | Decision support to resolve room collisions and approve bookings without double-booking. | Pre-checked conflict audit, atomic reservation confirmation, and anomaly alerts. |
| **University Leadership** | Understand spatial demand patterns and real capacity utilization. | Evidence-based demand analytics, timetable integrity audits, and underutilized space recovery. |

### Commercial Hypothesis
An annual institutional SaaS license for universities, integrated with student information systems (SIS/ERP). The initial entry point is club and departmental event reservations, progressively expanding to timetable optimization once institutional data maturity improves.

---

## 3. Dataset Inventory and Empirical Findings

The project uses the official synthetic challenge dataset located in `02_education_campus_space/02_education_campus_space/`.

### Verified Dataset Inventory

| File | Records | Columns | Primary Key / Joins | Core Attributes |
| :--- | :--- | :--- | :--- | :--- |
| `rooms.csv` | 120 | 9 | `room_id` | `room_id`, `building` (5), `floor` (0–5), `room_type` (5), `capacity` (15–179), `projector` (0/1), `computers` (0/1), `accessible` (0/1), `power_outlets` (5–30) |
| `class_schedule.csv` | 5,000 | 8 | `schedule_id` (joins on `room_id`) | `schedule_id`, `room_id`, `course_code`, `weekday` (`Mon`–`Sat`), `start_hour` (8–18), `duration_hours` (1–3), `expected_attendance`, `department` (5) |
| `events.csv` | 1,800 | 7 | `event_id` (joins on `room_id`) | `event_id`, `room_id`, `event_type` (5), `start_time` (timestamp), `duration_hours` (1–4), `expected_attendance`, `actual_attendance` |
| `room_requests.csv` | 5,000 | 9 | `request_id` (Demand log, no `room_id`) | `request_id`, `request_date` (2025-08-01 to 2025-11-28), `start_hour` (8–20), `duration_hours` (1–4), `required_capacity` (5–119), `projector_required`, `computers_required`, `purpose`, `status` |
| `README.txt` | — | — | — | Explicitly notes: synthetic dataset, join keys, and that requests represent unallocated demand logs without guaranteed `room_id`. |

### Empirical Findings & Anomaly Verification (Code-Verified)

A rigorous inspection of all 4 files using Python and Pandas established the following exact metrics:

1. **Structural Completeness:** Exactly **0 missing values** and **0 duplicate rows** across all 4 CSV files.
2. **Class Schedule Overlaps:**
   - **3,426 pairs** of class records share the same room, same weekday, and overlapping time intervals `[start_hour, start_hour + duration_hours)`.
   - **3,630 out of 5,000 class records (72.6%)** participate in at least one schedule collision.
3. **Capacity Violations:**
   - **2,009 out of 5,000 class records (40.18%)** have `expected_attendance > room.capacity`.
   - **722 out of 1,800 event records (40.11%)** have `expected_attendance > room.capacity`.
4. **Temporal Edge Cases in Events:**
   - **190 events cross midnight** into the following calendar date.
   - **650 events occur in off-peak/irregular hours** (starting before 07:00 or after 21:00, including midnight and early morning slots like 02:00, 04:00).
5. **Request Dataset Characteristics:**
   - Date range spans exactly 2025-08-01 to 2025-11-28 (120 calendar days).
   - Historical statuses: `Approved` (2,723, 54.5%), `Waitlisted` (939, 18.8%), `Cancelled` (718, 14.4%), `Rejected` (620, 12.4%).
   - Requests do not contain an `accessible_required` column, whereas `rooms.csv` tracks accessibility. The system will support accessibility filtering as a first-class user constraint in live requests.

### Data Reliability vs. Operational Validity
Structural integrity (no nulls) does not equate to operational validity. Because 72.6% of classes overlap in the synthetic timetable, treating all entries as hard conflicts would artificially paralyze the campus. SpaceSync establishes an explicit **Data Quality & Integrity Policy**:
- Report timetable anomalies transparently to administrators.
- In candidate filtering, evaluate conflicts against verified timetable slots and known reservations.
- Distinguish hard scheduling blocks from unverified contradictory timetable entries.

---

## 4. Core System Workflows

```
Requester Input (Date, Time, Duration, Capacity, Equipment, Accessibility)
    │
    ▼
[Feasibility Engine] ──(Hard Constraints: Capacity, Equipment, Accessibility, Classes, Events, DB Bookings)
    │
    ├── Feasible Rooms Found?
    │       ├── YES ──► [Ranking Engine] ──► [Explainability Breakdown] ──► User Selection ──► [Atomic Reservation]
    │       │
    │       └── NO  ──► [Alternative Generator] ──► (Search ±1h to ±3h time-slots & adjacent dates)
    │                                          └──► Rank valid alternatives by proximity & fit
    ▼
[Admin Operations & Analytics] ──► (Integrity Audit, Demand Patterns, Reconciled Utilization)
```

1. **Request Intake & Validation:** User specifies date, interval `[start, end)`, attendance, required facilities (projector, computers), and accessibility.
2. **Deterministic Feasibility Filtering:** Rooms failing any mandatory constraint are eliminated.
3. **Explainable Ranking:** Feasible rooms are scored based on capacity efficiency:
   $$\text{fit\_score} = \left(\frac{\text{requested\_attendance}}{\text{room\_capacity}}\right) \times 100$$
4. **Transparent Recommendations:** Clear cards highlighting why each room qualifies, room metadata, and audit trace.
5. **Proactive Alternatives:** If 0 rooms qualify, alternative time slots are tested against the exact same feasibility rules.
6. **Atomic Reservation Confirmation:** Confirmed reservations are verified for conflicts inside an atomic SQLite transaction, preventing race conditions or ghost bookings.

---

## 5. Scope Boundaries

### In Scope for DATATHON 2026 Prototype
- Local Python 3.10+ application with Streamlit user interface.
- 100% deterministic feasibility engine handling recurring classes, dated events, and confirmed bookings.
- Half-open interval temporal overlap logic `[start, end)`.
- Capacity-fit ranking and transparent explanation cards.
- Alternative time-slot discovery when no direct match exists.
- SQLite transactional persistence for confirmed reservations.
- Empirical demand analysis and baseline ML forecasting model.
- Admin analytics dashboard with data-quality audit.

### Out of Scope / Explicitly Deferred
- University Single Sign-On (SSO / OAuth2 / SAML).
- Live two-way synchronization with ERP/SIS software (Banner, PeopleSoft).
- Multi-region distributed databases.
- Circular ML ranking (training on heuristic outputs).

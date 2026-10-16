# SpaceSync — Architecture & Engineering Decisions

**Event:** DATATHON 2026, VIT-AP University  
**Track:** Education — Campus Space Optimization  
**Document Status:** Frozen (Phase 0)

---

## ADR-01: Framework Selection — Streamlit + Python

### Context
The DATATHON 2026 competition structure has two fast-paced deadlines:
- **Round 1 (10:00 AM – 12:00 PM):** Functional base prototype and data proof.
- **Round 2 (1:30 PM – 4:00 PM):** Complete working prototype with ML, analytics, and freeze.

### Decision
Build SpaceSync using **Python 3.10+ and Streamlit**, backed by **pandas, NumPy, scikit-learn, Plotly, and SQLite**. Reject a split frontend/backend stack (e.g., React + FastAPI/Node) for the hackathon MVP.

### Rationale
1. **Single Runtime & No Serialization Lag:** Eliminates API contract management, JSON deserialization bugs, CORS configuration, and disparate build toolchains (npm/vite).
2. **Execution Speed:** Directly renders rich, reactive UI components from Python objects while keeping state in-memory.
3. **Data Science Integration:** Native integration with Plotly charts and scikit-learn models without overhead.
4. **Reliability:** Leaves maximum time for core scheduling correctness, edge-case testing, and data engineering rather than boilerplate glue code.

### Trade-offs & Mitigations
- *Trade-off:* Streamlit reruns scripts from top to bottom on user interaction.
- *Mitigation:* Implement explicit session state caching (`st.session_state`) and isolate database write operations behind idempotent confirmation checks.

---

## ADR-02: Architectural Pattern — Modular Monolith

### Context
Complex microservice architectures or distributed queues are often proposed in hackathons but fail due to network partitioning, orchestration errors, and debugging friction.

### Decision
Adopt a **Modular Monolith** pattern:
- **Phase 1 (Round 1 Base Prototype):** Strictly two files (`engine.py` and `app.py`) to minimize cognitive overhead and reach a working MVP before noon.
- **Phase 2–7 (Round 2 Expansion):** Surgical refactoring into decoupled modules:
  - `spacesync/data/`: Dataset loaders, schema validators, anomaly detectors.
  - `spacesync/scheduling/`: Temporal interval engine, recurring and dated conflict logic.
  - `spacesync/ranking/`: Feasibility filter, capacity fit scorer, explanation generator.
  - `spacesync/alternatives/`: Time-slot neighborhood explorer.
  - `spacesync/reservations/`: SQLite transactional repository.
  - `spacesync/ml/`: Demand forecasting and baseline benchmarking.
  - `spacesync/analytics/`: Utilization reconciler and timetable audit metrics.

### Rationale
Provides clean separation of concerns and independent unit testability without the operational hazards of distributed systems.

---

## ADR-03: Feasibility & Scheduling — Deterministic Constraint Satisfaction vs. ML

### Context
There is often a temptation in hackathons to train black-box ML models (e.g. classification or neural networks) to predict whether a room is "available" or to score candidate suitability end-to-end.

### Decision
Feasibility checking must remain **100% deterministic**. Machine Learning will **never** override a hard constraint, mark an occupied room as available, or decide room eligibility.

### Rationale
1. **Zero Hallucination Tolerance:** In campus operations, assigning a student group to a room that has an active class causes immediate operational disruption. An ML model with 95% accuracy still double-books 5 out of 100 requests.
2. **Regulatory & Physical Reality:** Room capacity, wheelchair accessibility, and physical projector existence are immutable physical constraints, not probabilistic preferences.
3. **Auditability & Explainability:** Administrative staff and requesters require unambiguous explanations ("Room R004 is occupied by CS040 until 11:00 AM"), which deterministic rule systems provide by design.

---

## ADR-04: Machine Learning Scope — Time-Slot Demand Forecasting

### Context
The challenge requires a meaningful ML component where justified by the data. However, the supplied `room_requests.csv` does not include allocated room IDs or user selection feedback.

### Decision
Formulate ML as an **administrative demand forecasting problem**:
- **Target:** Predict campus-wide room request volume per time-slot (`requests_per_slot`).
- **Use Case:** Anticipate peak demand hours, help facilities administrators schedule staff, and guide the alternative-slot engine toward lower-congestion windows.
- **Strict Prohibition:** Do NOT fabricate historical room selection labels to train a supervised ranking model on top of heuristic outputs (circular evaluation).

### Evaluation Standard
Compare models (e.g. Random Forest Regressor) against a naive historical mean baseline using chronological train/test splitting. If the ML model does not beat the baseline, document that finding transparently.

---

## ADR-05: Dataset Integrity & Quality Policy

### Context
Empirical analysis of the synthetic dataset revealed major operational contradictions:
- 3,426 overlapping same-room class pairs (involving 72.6% of class records).
- 2,009 classes and 722 events exceeding listed room capacity.
- 190 events crossing midnight; 650 events in off-peak hours (e.g., 02:00, 04:00).

### Decision
**Expose and protect; do not silently clean or falsify.**
1. **Never Silently Drop Records:** Anomalous records are part of the challenge data and represent real-world university data debt.
2. **Defensive Scheduling:** In the presence of a class schedule conflict in the source data, fail safe (mark the room occupied for the duration of all listed classes).
3. **Operational Visibility:** Build a dedicated Timetable Integrity Audit in the administrator dashboard to surface these contradictions with exact counts.
4. **Honest Utilization Accounting:** Calculate room utilization using non-overlapping reconciled room-hours, rather than inflating numbers through double-counted collisions.

---

## ADR-06: Database & Concurrency Strategy

### Context
Room booking systems are vulnerable to race conditions where two users request the same room at the same time. In Streamlit, page reruns can also accidentally trigger repeated database operations.

### Decision
Use **SQLite with WAL (Write-Ahead Logging) and immediate transactional locks**:
1. Confirmed bookings are written inside a transaction:
   ```sql
   BEGIN IMMEDIATE;
   -- Re-check availability within transaction
   -- Insert reservation
   COMMIT;
   ```
2. Streamlit state guards (`st.session_state`) track confirmation state to ensure idempotent execution across UI interactions.

---

## ADR-07: Scope Management & Deferred Production Work

### Context
A 6-hour hackathon requires disciplined scope control. Feigning enterprise features leads to fragile demos.

### Deferred Capabilities (Post-Hackathon Roadmap)
| Feature | Reason for Deferral | MVP Alternative |
| :--- | :--- | :--- |
| **Enterprise SSO / LDAP** | Out of scope for hackathon; adds security risk. | Role-switcher in UI (Student Requester vs. Campus Scheduler). |
| **Live SIS/ERP Sync** | Requires institutional API access not available in sandbox. | Direct CSV data ingestion with hot-reload capability. |
| **Email/SMS Notifications** | Non-core notification plumbing. | In-app confirmation badge and printable reservation receipt. |
| **Multi-Campus Geographic Routing** | Single campus dataset provided. | Building and floor distance heuristics within the campus dataset. |

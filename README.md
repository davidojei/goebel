# Goebel — A Multi-Asset Predictive Maintenance Platform

Named for Kai Goebel, the NASA prognostics researcher behind two of the
four public datasets this project builds on (C-MAPSS and the IMS
bearing dataset).

## What this actually is

Four independent ML sub-systems, each solving a different real
predictive-maintenance problem, on a different public industrial
dataset, tied together by one shared explainability layer and served
through one API and one dashboard:

| Sub-system | Task | Dataset | Final result |
|---|---|---|---|
| Turbine | RUL regression | NASA C-MAPSS (FD001, FD003) | RMSE 14.15, PHM08 296.7 |
| Bearing — Fault | 4-class classification | CWRU, 40 files across 4 loads | 70.0% mean LOFO accuracy, 0.997 ROC-AUC |
| Bearing — RUL | 2-stage degradation detection + regression | NASA/IMS, 4 real failures | 76% Stage 1 accuracy (100% precision), 0.163 MAE Stage 2 |
| Hydraulic | 4 independent component health targets | UCI Hydraulic (2205 cycles) | 100% / 97.0% / 98.9% / 91.0% across targets |

Every number above is a genuinely held-out result, not a training-set
score, and every one of them is explained — including the numbers that
are worse than you'd see in a typical portfolio project, and why they're
worse for defensible, documented reasons rather than oversight.

## The actual point of this project

Anyone can report a final accuracy. What's harder, and what this
project is really about, is the discipline behind getting a number you
can trust:

- **Catching evaluation leakage before it inflates a result.** Every
  sub-system hit this at least once — Turbine's engine-level splits,
  Bearing's Leave-One-File-Out methodology (after a random split scored
  a fake 94%), Hydraulic's block-aware splitting (after Cooler's first
  pass scored a fake 100%), and IMS's cross-bearing evaluation (after a
  single-bearing split leaked so badly that near-identical 10-minute
  neighbors sat on both sides of the train/test line).
- **Diagnosing root causes with evidence, not guesses.** Turbine's
  single biggest fix (RMSE 18.66→14.29) came from tracing one specific
  bad prediction — a short-lived engine, true RUL 21, predicted 66 —
  back to a real mechanism (compressed degradation curves confusing
  absolute-value comparisons), not from trying features at random.
- **Keeping negative results, not hiding them.** Bearing's investigation
  includes three separate rejected band-selection strategies and the
  real reason each one failed. Hydraulic's Accumulator section documents
  four rejected attempts to beat its own best result, all converging on
  the same conclusion. These are treated as real findings, not deleted
  scratch work.
- **Checking against external, published evidence.** Bearing's
  cross-file accuracy (70%) was checked against literature specifically
  critiquing CWRU as a benchmark, which reports the same 50-60% range
  under equivalent methodology. Hydraulic's Accumulator result (91%) was
  checked against the original researchers' own reported baseline
  (~90.4% before their specialized feature-selection work). Both matched.

## What's running

Four models served live through one FastAPI backend
(`/predict/turbine`, `/predict/bearing`, `/predict/hydraulic`,
`/predict/ims/stage1`, `/predict/ims/stage2`), each returning a
prediction plus a SHAP explanation translated into plain language — not
raw feature names, but sentences like *"how far the HPC outlet pressure
has drifted from this engine's healthy starting point strongly pointed
toward this result."* One Streamlit dashboard consumes all five
endpoints, with two modes per sub-system: a manual single-sample mode,
and a live-feed mode that streams a real trajectory through the system
in chronological order, updating predictions as each reading arrives.

**Turbine — predicted vs. true RUL, tracked live across a real engine's
full trajectory:**

![Turbine live feed](docs/screenshots/turbine_live_feed.png)

**Bearing — fault classification streaming through a real recording,
confidence tracked per window:**

![Bearing live feed](docs/screenshots/bearing_live_feed.png)
![Bearing confidence over stream](docs/screenshots/bearing_confidence_chart.png)

**IMS — a real bearing's entire life, streamed in order. Flat near-zero
probability for the first ~100 readings, a genuine climb with an
authentic mid-stream dip, ending near failure. The dashed line is the
decision threshold, selected honestly from training data only — this
specific bearing was never part of that selection:**

![IMS degradation probability](docs/screenshots/ims_probability_chart.png)
![IMS degradation alert](docs/screenshots/ims_degradation_alert.png)

**Hydraulic — four independent components assessed from one shared
sensor reading per cycle, each with its own plain-English explanation:**

![Hydraulic live feed](docs/screenshots/hydraulic_live_feed.png)
![Hydraulic manual sample with explainability](docs/screenshots/hydraulic_manual_sample.png)

## Architecture

```
Streamlit dashboard (8502)  ──HTTP──▶  FastAPI backend (8003)
        ▲                                    │
        │                                    ▼
   sample_data/                    models/*.pkl + SHAP TreeExplainer
   (exported rows,                 (loaded once at startup,
   live-feed trajectories)          not per-request)
```

`src/shared/shap_utils.py` is the one file every endpoint depends on —
`TreeExplainer` construction, top-feature extraction (handling both old
list-based and current 3D-array SHAP output formats, since the library's
API shape changed mid-project and broke things twice before this was
made defensive), and per-sub-system plain-English feature translation.

Both services are containerized (`docker/Dockerfile`,
`docker/docker-compose.yml`) and networked together — the dashboard
reaches the API via Docker's internal service-name resolution
(`http://api:8000`), not a hardcoded IP, so the same compose file runs
correctly on any machine with Docker installed, not just the one it was
built on.

## Running it

```bash
# Docker (recommended) — one command, both services, correctly networked
docker compose -f docker/docker-compose.yml up --build
# Dashboard: http://localhost:8502   API: http://localhost:8003

# Without Docker
pip install -r requirements.txt
uvicorn api.main:app --reload          # terminal 1
streamlit run dashboard/app.py         # terminal 2
# Dashboard: http://localhost:8501   API: http://localhost:8000
```

Raw data isn't committed (combined size across four datasets runs into
multiple GB). Each sub-system README documents its dataset's official
source, expected folder structure, and any known packaging
inconsistencies in the public download (IMS in particular ships with a
documented, non-obvious folder-naming quirk from the original providers
— covered in that README so it isn't rediscovered from scratch).

## Project structure

```
goebel/
├── notebooks/              # full exploratory + modeling history, one per sub-system
├── src/
│   ├── turbine/  bearing/  hydraulic/     # feature engineering, training, evaluation
│   └── shared/shap_utils.py               # explainability layer, used by every endpoint
├── models/                  # saved final models (.pkl) + feature name lists
├── api/main.py
├── dashboard/{app.py, sample_data/}
├── docker/
└── docs/                    # per-sub-system READMEs, mastery guides, screenshots
```

## Full documentation

- [`docs/turbine_README.md`](docs/turbine_README.md)
- [`docs/bearing_README.md`](docs/bearing_README.md)
- [`docs/ims_README.md`](docs/ims_README.md)
- [`docs/hydraulic_README.md`](docs/hydraulic_README.md)

Each includes the complete methodology, every mistake caught and how,
and an honest limitations section. `*_MASTERY_GUIDE.md` files alongside
them give a plain-language walkthrough of the same material with an
interview-question bank, for anyone (including future-me) who needs to
re-derive the reasoning without re-reading raw notebook history.

## Known limitations, stated directly

- **IMS RUL is a percentage of each bearing's degrading-window length,
  not raw hours.** Real deployment would need a second model estimating
  expected window length first — this is documented as unresolved future
  work in the IMS README, not glossed over.
- **Bearing's 70% cross-file accuracy is well below the 95%+ commonly
  cited for this dataset.** That gap is intentional and externally
  validated (see above) — the higher numbers in circulation almost
  universally use an easier, same-recording evaluation that this project
  specifically avoided.
- **Hydraulic's Accumulator target retains a real, twice-diagnosed
  residual confusion** between three of its four classes, traced via two
  independent SHAP investigations to genuine physical signal overlap —
  not an unexplored feature gap, but not a solved one either.
- **Stage 1 IMS recall is 49%**, with 100% precision — a deliberately
  conservative model that misses roughly half of true degradation cases
  but never false-alarms. Two independent fixes (calibration,
  baseline-relative features) failed to close this gap; it's an open
  problem, named as one.

Nothing here is presented as finished. It's presented as honestly
measured — which was always the actual goal.

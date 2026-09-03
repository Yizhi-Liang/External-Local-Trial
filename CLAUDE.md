# CLAUDE.md — 01_External_Local_RCT

**Purpose:** Bayesian VOI + power-prior framework evaluating four adoption-policy options for treatments with uncertain external evidence applicability; sintilimab/NSCLC case study.

**Study type:** VOI / HTA methods paper (not an empirical RCT or RWE study).

**Layout (non-standard — differs from global template):**
- `01_data/surv_curve_data/digitized/` — raw digitized survival CSVs; **read-only, never modify**
- `01_data/surv_curve_data/` — clean/fitted inputs safe to read
- `02_code/` — all analysis (Python Jupyter notebooks)
- `03_output/` — derived outputs (params, prices, ENBS, figures)
- `00_journal/` — manuscript versions by journal

**Code entry points (run in order):**
1. `02_code/01_find_survival_curve_distribution.ipynb` — fit survival distributions
2. `02_code/02_value-based_price.ipynb` — ICER / value-based price
3. `02_code/03_psa_nmb.ipynb` — PSA / NMB / ENBS across policy scenarios
4. `02_code/04_plots.ipynb` — figures for manuscript

**Status:** Under review at *Value in Health* (VIH-2025-1313); first major revision submitted; in proof/final-check stage as of 2026-06.

**HEOR workflow:** `pre-submission-audit` for any final checks; `hem-qc-checklist` to verify model outputs; `health-economic-modeling` skill if extending the model.

*(Remove this file to opt out of project-specific Claude guidance.)*


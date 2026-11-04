#!/usr/bin/env python3
"""Round-2 audit (R2-3②): explicit truncation verification for the 48-hour feature pipeline.

Reviewer 2 (Comment 3) states the published feature-generation code does not clearly
implement the truncation "at death, extubation, or ICU discharge". This script:

  PART A - replicates the ORIGINAL rule verbatim (points 0-48h post ICU admission,
           MV candidates 0-48h, ±2h nearest-neighbor, tie -> preceding record) from the
           published trajectory table, and verifies that the recomputed features match
           1_队列数据/MIMIC_IV_analysis_ready.csv exactly (formula calibration).
  PART B - adds the manuscript-stated truncation rule: the observation window ends at
           min(48 h, death, first extubation (end of the first InvasiveVent segment),
           ICU discharge), with pairs admitted only if both the ABG time and the
           ventilator-setting time fall inside that window; recomputes all features,
           re-applies the >=3-pairs gate, and diffs against the original analysis set.
  PART C - determines the VR slope definition actually used in analysis_ready
           (two-point net slope delta/span vs per-point least-squares polyfit).

Outputs: truncation_verification.csv + truncation_verification_log.md
""""""

AUTHORS-SIDE AUDIT: inputs are the pre-revision published artifacts (archived trajectory and
analysis-ready files); not part of the fresh-cohort pipeline run. If they are absent, the script
skips with a message. The committed chain (run_mimic_rebuild_round2.py) implements the explicit
truncation directly and its boundary decomposition is printed in its own log.
"""
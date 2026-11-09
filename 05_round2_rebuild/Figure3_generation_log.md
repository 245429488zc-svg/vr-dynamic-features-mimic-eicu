# Figure 3 forest plot generation log

## Data source

- Table 2 source: /Volumes/tongtong/论文/icuVR与死亡的关联/42_GitHub仓库/vr-dynamic-features-code/05_round2_rebuild/Table2_early_VR_mortality.csv

## Skill use

- Used forest-plot-styler workflow conventions: standardized study/or/ci_lower/ci_upper input table and forest-plot output.
- The packaged skill entrypoint was checked with --help and run on Figure3_forest_input_for_skill.csv.
- A custom manuscript-format plot was also generated to add right-side OR, P value, and N columns.

## Plotted data

- Number of VR dynamic features: 8
- Estimates: Model 3 adjusted ORs from Table 2.
- Core final-model features were emphasized: VR initial and VR delta adjusted for VR initial.

## Output files

- Figure3_VR_dynamic_features_forest.png
- Figure3_VR_dynamic_features_forest.pdf
- Figure3_VR_dynamic_features_forest.svg
- Figure3_forest_input_for_skill.csv
- Figure3_forest_model3_data.csv
- Figure3_skill_native_forest.png
- Figure3_skill_native_forest.pdf

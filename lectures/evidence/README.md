# Archived report tables

`benchmark.csv` and `ablation.csv` are transcriptions of the eight-section benchmark
and ST-colon1 backbone/integration ablation tables in the course report, using the
supplied project manuscript as their original source.

The presentation recomputes unweighted section means and table differences; it does
not rerun the historical experiments. In particular:

- SIGMA mean ARI: 0.4901; MCGAE mean ARI: 0.4204 across eight sections.
- SIGMA exceeds MCGAE on four sections and trails on four.
- The benchmark MCGAE row and the ablation's physical-only baseline represent
  different table entries; they must not be treated as the same run.
- Upgraded physical-only: ARI 0.6712, Macro Dice 0.7383.
- Upgraded hybrid: ARI 0.6775, Macro Dice 0.7319.
- Upgraded regularization has the highest ablation ARI (0.6803), while original
  attention has the highest Macro Dice (0.8060).

These are descriptive archived results, with no repeat-run uncertainty or new
significance tests. Conflicting prose and figures in the original material are
not combined with these tables. These metrics assess clustering/compartment
agreement, not validated patient-level invasion prediction.

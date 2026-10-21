# Archived report tables

`benchmark.csv` and `ablation.csv` are transcriptions of the eight-section benchmark
and ST-colon1 backbone/integration ablation tables in the course report, using the
supplied project manuscript for the benchmark. The ablation was superseded by
the author's corrected eight-group table supplied on 22 October 2026. All four
metrics in that table are retained.

The presentation recomputes unweighted section means and table differences; it does
not rerun the historical experiments. In particular:

- SIGMA mean ARI: 0.4901; MCGAE mean ARI: 0.4204 across eight sections.
- SIGMA exceeds MCGAE on four sections and trails on four.
- The benchmark MCGAE row and the ablation's physical-only baseline represent
  different table entries; they must not be treated as the same run.
- Upgraded physical-only: ARI 0.6345, NMI 0.5574, Macro Dice 0.7292, IoU 0.6091.
- Upgraded SIGMA: ARI 0.7328, NMI 0.6595, Macro Dice 0.8618, IoU 0.7630.
- Upgraded SIGMA leads all eight ablation variants on all four metrics; its
  gains over the upgraded baseline are 0.0983 ARI and 0.1326 Macro Dice.
- The benchmark retains SIGMA ST-colon1 ARI 0.6775. It is a separate comparison
  and was not replaced with the corrected ablation entry.

These are descriptive archived results, with no repeat-run uncertainty or new
significance tests. Conflicting prose and figures in the original material are
not combined with these tables. These metrics assess clustering/compartment
agreement, not validated patient-level invasion prediction.

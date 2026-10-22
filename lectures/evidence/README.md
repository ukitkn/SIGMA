# Report evidence

These files follow the course report revised on 23 October 2026:

- `ablation.csv`: the author's corrected eight-group ST-colon1 experiment,
  crossing two-layer/three-layer backbones with Base, Reg, Attn and Hybrid.
- `benchmark.csv`: the separate four-method comparison across eight sections.
- `configuration_top3.csv`: the separate seven-section top-three ranking; this
  incomplete ranking must not be averaged as if it contained complete fixed-model results.
- `report_reference.json`: report data statistics, key results, biological
  associations, the revised conclusion and the exact Software 3.0 paragraph.

The presentation computes summaries from these tables, without rerunning the
report experiments. Key checks are:

- Upgraded SIGMA: ARI 0.7328, NMI 0.6595, Macro Dice 0.8618, IoU 0.7630.
  It leads all eight ablation variants on all four metrics.
- Matching upgraded Base: 0.6345 / 0.5574 / 0.7292 / 0.6091.
  The original two-layer Hybrid performs below its corresponding baseline.
- Eight-section mean ARI: SIGMA 0.4901; MCGAE 0.4204; Seurat 0.3941;
  stLearn 0.2790. Improvements depend on the tissue section.
- The benchmark's ST-colon1 SIGMA ARI is 0.6775, distinct from ablation 0.7328.
- In the configuration ranking, upgraded Hybrid ranks first on four sections and
  in the top three on six of seven. These scores remain separate from the benchmark.
- Reported biological interpretation uses 1,665 tumor-core spots, with EMT
  r = 0.780, ECM r = 0.765 and nearest-CAF-distance r = −0.286. The earlier
  Slingshot–DPT comparison gives approximately rho = 0.85 on a shared embedding.

The biological associations are interpretation checks, not independent temporal
validation. The field supplies candidate progression directions from static data;
it does not measure tumor velocity or provide a validated patient-risk forecast.
No synthetic result is inserted into the report tables. The removed section-wise
ARI-difference figure is not included in the presentation.

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sigma.analysis import analyze
from sigma.config import Config
from sigma.pipeline import fit_model
from sigma.utils import save_json


def run_ablation(config: Config, adata, output_dir: Path) -> pd.DataFrame:
    model_cfg = config.section("model")
    settings = [
        ("shallow_spatial_only", 2, False, False, "concat", False),
        ("shallow_ccc_regularized", 2, False, False, "concat", True),
        ("shallow_ccc_attention", 2, True, True, "attention", False),
        ("shallow_ccc_hybrid", 2, True, True, "concat", False),
        ("full_spatial_only", 3, False, False, "concat", False),
        ("full_ccc_regularized", 3, False, False, "concat", True),
        ("full_ccc_attention", 3, True, True, "attention", False),
        ("sigma_dual_hybrid", 3, True, True, "concat", False),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    truth_key = config.section("data").get("truth_key", "ident.annot")
    analysis_cfg = config.section("analysis")
    for name, layers, use_dual, use_ccc_loss, fusion, jk in settings:
        run_dir = output_dir / name
        overrides = {
            "layers": layers,
            "use_spatial_view": use_dual,
            "use_ccc_view": use_dual,
            "use_ccc_loss": use_ccc_loss,
            "fusion": fusion,
            "jumping_knowledge": jk,
        }
        result = fit_model(config, adata.copy(), run_dir, overrides=overrides)
        result = analyze(
            result, run_dir,
            truth_key=truth_key,
            resolution=float(analysis_cfg.get("leiden_resolution", 0.4)),
            annotation_min_score=float(analysis_cfg.get("annotation_min_score", 0.05)),
            trajectory_neighbors=int(analysis_cfg.get("trajectory_neighbors", 20)),
            field_grid_size=int(analysis_cfg.get("field_grid_size", 80)),
            field_neighbors=int(analysis_cfg.get("field_neighbors", 80)),
            field_bandwidth=float(analysis_cfg.get("field_bandwidth", 80.0)),
        )
        metric_path = run_dir / "evaluation_metrics.json"
        with metric_path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        ari = metrics.get("metrics", {}).get("sigma_cluster", {}).get("ari")
        rows.append({
            "variant": name,
            "layers": layers,
            "dual_view": use_dual,
            "ccc_objective": use_ccc_loss,
            "fusion": fusion,
            "jumping_knowledge": jk,
            "ari": ari,
            "output": str(run_dir),
        })
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "ablation_summary.csv", index=False)
    save_json(output_dir / "ablation_design.json", {
        "design": "Two encoder depths crossed with four communication and fusion settings.",
        "variants": table.to_dict(orient="records"),
        "inherited_model_config": model_cfg,
    })
    return table

from __future__ import annotations

import argparse
from pathlib import Path

import scanpy as sc

from sigma.analysis import analyze, compare_slingshot
from sigma.config import load_config, resolve_path
from sigma.experiments import run_ablation
from sigma.pipeline import fit_model, load_or_prepare, prepare_data
from sigma.rda import convert_seurat_rda


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sigma", description="Run the SIGMA spatial analysis pipeline.")
    parser.add_argument("command", choices=["convert-rda", "prepare", "train", "analyze", "run", "ablate", "compare-slingshot"])
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--slingshot-csv", default=None)
    parser.add_argument("--id-column", default="spot_id")
    parser.add_argument("--time-column", default="pseudotime")
    parser.add_argument("--rda-input", default=None)
    parser.add_argument("--h5ad-output", default=None)
    parser.add_argument("--assay", default=None)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    config = load_config(args.config)
    configured_output = config.section("outputs").get("directory", "outputs/ST-colon1")
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else resolve_path(config, configured_output)
    if args.command == "convert-rda":
        if not args.rda_input:
            raise ValueError("--rda-input is required for convert-rda.")
        configured_input = resolve_path(config, config.section("data").get("input", "data/processed/input.h5ad"))
        h5ad_output = Path(args.h5ad_output).expanduser().resolve() if args.h5ad_output else configured_input
        convert_seurat_rda(args.rda_input, h5ad_output, assay=args.assay)
        return
    if args.command == "prepare":
        prepare_data(config, output_dir)
        return
    if args.command == "train":
        adata = load_or_prepare(config, output_dir)
        fit_model(config, adata, output_dir)
        return
    if args.command == "analyze":
        path = output_dir / "embeddings.h5ad"
        if not path.is_file():
            raise FileNotFoundError(f"Trained AnnData file not found: {path}")
        _analyze_from_config(config, sc.read_h5ad(path), output_dir)
        return
    if args.command == "run":
        adata = load_or_prepare(config, output_dir)
        embeddings = fit_model(config, adata, output_dir)
        _analyze_from_config(config, embeddings, output_dir)
        return
    if args.command == "ablate":
        adata = load_or_prepare(config, output_dir)
        run_ablation(config, adata, output_dir / "ablations")
        return
    if args.command == "compare-slingshot":
        if not args.slingshot_csv:
            raise ValueError("--slingshot-csv is required for compare-slingshot.")
        analyzed_path = output_dir / "analyzed.h5ad"
        if not analyzed_path.is_file():
            raise FileNotFoundError(f"Analyzed AnnData file not found: {analyzed_path}")
        compare_slingshot(args.slingshot_csv, sc.read_h5ad(analyzed_path),
                          output_dir / "slingshot_comparison.json", args.id_column,
                          args.time_column)


def _analyze_from_config(config, adata, output_dir: Path) -> None:
    data_cfg = config.section("data")
    analysis_cfg = config.section("analysis")
    analyze(
        adata,
        output_dir,
        truth_key=data_cfg.get("truth_key", "ident.annot"),
        resolution=float(analysis_cfg.get("leiden_resolution", 0.4)),
        annotation_min_score=float(analysis_cfg.get("annotation_min_score", 0.05)),
        trajectory_neighbors=int(analysis_cfg.get("trajectory_neighbors", 20)),
        field_grid_size=int(analysis_cfg.get("field_grid_size", 80)),
        field_neighbors=int(analysis_cfg.get("field_neighbors", 80)),
        field_bandwidth=float(analysis_cfg.get("field_bandwidth", 80.0)),
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import anndata as ad
import pandas as pd
from scipy.io import mmread


def convert_seurat_rda(input_path: str | Path, output_path: str | Path,
                       assay: str | None = None) -> Path:
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Seurat RDA file not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "export_seurat_rda.R"
    if not script_path.is_file():
        raise FileNotFoundError(f"Seurat export script not found: {script_path}")

    with tempfile.TemporaryDirectory(prefix="sigma_rda_", dir=output_path.parent) as temporary:
        export_dir = Path(temporary) / "matrix_export"
        command = ["Rscript", str(script_path), str(input_path), str(export_dir)]
        if assay:
            command.append(assay)
        subprocess.run(command, check=True)

        matrix = mmread(export_dir / "matrix.mtx").tocsr().transpose().tocsr()
        barcodes = pd.read_csv(export_dir / "barcodes.tsv", header=None, sep="\t")[0].astype(str)
        features = pd.read_csv(export_dir / "features.tsv", header=None, sep="\t")
        metadata = pd.read_csv(export_dir / "metadata.csv", low_memory=False)
        coordinates = pd.read_csv(export_dir / "spatial.csv")
        if matrix.shape != (len(barcodes), len(features)):
            raise ValueError("The exported count matrix does not match its barcode and feature tables.")
        if metadata["barcode"].astype(str).duplicated().any():
            raise ValueError("The exported metadata contains duplicate barcodes.")
        if coordinates["barcode"].astype(str).duplicated().any():
            raise ValueError("The exported spatial table contains duplicate barcodes.")

        obs = metadata.set_index(metadata["barcode"].astype(str)).drop(columns="barcode")
        obs = obs.reindex(barcodes.to_list())
        spatial = coordinates.assign(barcode=coordinates["barcode"].astype(str)).set_index("barcode")
        spatial = spatial.reindex(barcodes.to_list())
        if obs.isna().all(axis=1).any() or spatial[["x", "y"]].isna().any().any():
            raise ValueError("The exported metadata or spatial coordinates do not cover all count barcodes.")
        obs.index = pd.Index(barcodes.to_numpy(), name="spot_id")
        var_names = features.iloc[:, 1].astype(str).to_numpy()
        var = pd.DataFrame(index=pd.Index(var_names, name="gene_symbol"))
        var["feature_id"] = features.iloc[:, 0].astype(str).to_numpy()
        adata = ad.AnnData(X=matrix, obs=obs, var=var)
        adata.var_names_make_unique()
        adata.obsm["spatial"] = spatial[["x", "y"]].to_numpy(dtype="float32")
        adata.layers["counts"] = adata.X.copy()
        source_metadata = (export_dir / "source_metadata.txt").read_text(encoding="utf-8")
        adata.uns["seurat_visium_export"] = {
            "input_file": str(input_path),
            "metadata": source_metadata,
            "coordinate_columns": ["imagecol", "imagerow"],
        }
        adata.write_h5ad(output_path, compression="gzip")
    return output_path

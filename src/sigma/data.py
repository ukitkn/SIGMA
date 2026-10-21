from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from sigma.utils import to_dense_array


def load_and_preprocess(
    input_path: str | Path,
    *,
    counts_layer: str | None = None,
    min_cells: int = 1,
    n_hvg: int = 3000,
    target_sum: float = 10000,
    truth_key: str = "ident.annot",
    spatial_key: str = "spatial",
) -> ad.AnnData:
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"AnnData input not found: {input_path}")
    adata = sc.read_h5ad(input_path)
    if adata.n_obs < 3:
        raise ValueError("At least three observations are required.")
    if spatial_key not in adata.obsm:
        raise KeyError(f"Spatial coordinates are missing from adata.obsm['{spatial_key}'].")
    coords = np.asarray(adata.obsm[spatial_key])
    if coords.ndim != 2 or coords.shape[0] != adata.n_obs or coords.shape[1] < 2:
        raise ValueError("Spatial coordinates must have shape (n_spots, at_least_2).")
    if not np.isfinite(coords[:, :2]).all():
        raise ValueError("Spatial coordinates contain non-finite values.")

    source = adata.layers[counts_layer] if counts_layer else adata.X
    counts = source.copy() if sparse.issparse(source) else np.asarray(source).copy()
    has_negative = np.any(counts.data < 0) if sparse.issparse(counts) else np.any(counts < 0)
    if has_negative:
        raise ValueError("Expression input must contain non-negative counts or abundances.")
    adata.X = counts
    sc.pp.filter_genes(adata, min_cells=min_cells)
    if adata.n_vars < 3:
        raise ValueError("Fewer than three genes remain after gene filtering.")

    adata.layers["counts"] = adata.X.copy()
    n_top = min(n_hvg, adata.n_vars)
    flavor = "seurat_v3" if _is_integer_like(adata.layers["counts"]) else "seurat"
    sc.pp.highly_variable_genes(adata, flavor=flavor, n_top_genes=n_top)
    if "highly_variable" not in adata.var or int(adata.var["highly_variable"].sum()) == 0:
        raise RuntimeError("Highly variable gene selection returned no genes.")

    full = adata.copy()
    sc.pp.normalize_total(full, target_sum=target_sum)
    sc.pp.log1p(full)
    adata.uns["preprocessing"] = {
        "counts_layer": counts_layer or "X",
        "min_cells": min_cells,
        "hvg_flavor": flavor,
        "n_hvg_requested": n_hvg,
        "n_hvg_used": int(adata.var["highly_variable"].sum()),
        "target_sum": target_sum,
    }

    adata = adata[:, adata.var["highly_variable"].to_numpy()].copy()
    adata.X = full[:, adata.var_names].X.copy()
    adata.raw = full
    if truth_key in full.obs:
        adata.obs[truth_key] = full.obs[truth_key].astype(str).to_numpy()
    return adata


def get_full_expression(adata: ad.AnnData) -> tuple[np.ndarray, list[str]]:
    if adata.raw is None:
        raise RuntimeError("Full-gene log-normalized expression was not retained.")
    return to_dense_array(adata.raw.X), adata.raw.var_names.astype(str).tolist()


def _is_integer_like(matrix: object, tolerance: float = 1e-6) -> bool:
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if values.size == 0:
        return False
    return bool(np.all(np.abs(values - np.round(values)) < tolerance))


def align_labels(adata: ad.AnnData, label_path: str | Path, key: str = "truth") -> None:
    table = pd.read_csv(label_path, sep=None, engine="python", index_col=0)
    if table.shape[1] == 0:
        raise ValueError("The label file must contain at least one column.")
    labels = table.iloc[:, 0].copy()
    labels.index = labels.index.astype(str).str.strip()
    obs_names = adata.obs_names.astype(str).str.strip()
    aligned = labels.reindex(obs_names)
    if aligned.isna().any():
        missing = int(aligned.isna().sum())
        raise ValueError(f"The label file is missing {missing} observation IDs.")
    adata.obs[key] = aligned.astype(str).to_numpy()

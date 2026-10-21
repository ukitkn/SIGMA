from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             classification_report, normalized_mutual_info_score)

from sigma.data import get_full_expression
from sigma.utils import save_json


SIGNATURES = {
    "TumorEpithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "CEACAM5", "MSLN"],
    "NormalEpithelial": ["MUC2", "ALPI", "CA1", "AGR2", "TFF3", "CLCA1"],
    "Fibroblast": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "FAP", "ACTA2"],
    "Immune": ["PTPRC", "CD3D", "CD3E", "LYZ", "CD68", "MS4A1"],
    "Endothelial": ["PECAM1", "VWF", "KDR", "EMCN"],
}
EMT_GENES = [
    "VIM", "FN1", "SNAI1", "SNAI2", "ZEB1", "ZEB2", "TWIST1", "MMP9",
    "ITGA5", "TAGLN", "COL1A1", "COL1A2",
]
EPITHELIAL_GENES = ["CDH1", "EPCAM"]
ECM_GENES = ["COL1A1", "COL3A1", "COL5A1", "COL6A3", "LUM", "DCN", "THBS1", "SPARC"]


def analyze(adata: ad.AnnData, output_dir: Path, *, truth_key: str = "ident.annot",
            resolution: float = 0.4, annotation_min_score: float = 0.05,
            trajectory_neighbors: int = 20, field_grid_size: int = 80,
            field_neighbors: int = 80, field_bandwidth: float = 80.0) -> ad.AnnData:
    output_dir.mkdir(parents=True, exist_ok=True)
    if "X_sigma" not in adata.obsm:
        raise KeyError("The trained embedding adata.obsm['X_sigma'] is missing.")
    _cluster_full_embeddings(adata, resolution)
    _annotate_compartments(adata, annotation_min_score)
    _cluster_tumor_cells(adata, resolution)
    _trajectory(adata, trajectory_neighbors)
    _spatial_field(adata, output_dir, field_grid_size, field_neighbors, field_bandwidth)
    metrics = _evaluate_clusters(adata, truth_key)
    save_json(output_dir / "evaluation_metrics.json", metrics)
    adata.write_h5ad(output_dir / "analyzed.h5ad", compression="gzip")
    return adata


def _annotate_compartments(adata: ad.AnnData, minimum: float) -> None:
    expression, genes = get_full_expression(adata)
    gene_index = {gene.upper(): i for i, gene in enumerate(genes)}
    gene_z = (expression - expression.mean(axis=0, keepdims=True)) / (
        expression.std(axis=0, keepdims=True) + 1e-6
    )
    scores = {}
    for label, signature in SIGNATURES.items():
        columns = [gene_index[g] for g in signature if g in gene_index]
        scores[label] = gene_z[:, columns].mean(axis=1) if columns else np.zeros(adata.n_obs)
    score_frame = pd.DataFrame(scores, index=adata.obs_names)
    adata.obsm["marker_scores"] = score_frame.to_numpy(dtype=np.float32)
    adata.uns["marker_score_names"] = score_frame.columns.tolist()
    predicted_type = score_frame.idxmax(axis=1).astype(str)
    best_score = score_frame.max(axis=1).to_numpy()
    predicted_type[best_score < minimum] = "Unclassified"
    tumor_score = scores["TumorEpithelial"] - scores["NormalEpithelial"]
    stroma_score = np.maximum.reduce([scores["Fibroblast"], scores["Immune"], scores["Endothelial"]])
    compartment = np.full(adata.n_obs, "Other", dtype=object)
    tumor_mask = (tumor_score >= stroma_score) & (tumor_score >= minimum)
    stroma_mask = (stroma_score > tumor_score) & (stroma_score >= minimum)
    compartment[tumor_mask] = "Tumor"
    compartment[stroma_mask] = "Stroma"
    adata.obs["marker_cell_type"] = predicted_type.to_numpy()
    adata.obs["compartment"] = pd.Categorical(compartment, categories=["Tumor", "Stroma", "Other"])
    adata.obs["marker_tumor_score"] = tumor_score
    adata.obs["marker_stroma_score"] = stroma_score


def _cluster_tumor_cells(adata: ad.AnnData, resolution: float) -> None:
    tumor_mask = adata.obs["compartment"].astype(str).to_numpy() == "Tumor"
    adata.obs["tumor_subtype"] = "NotTumor"
    if tumor_mask.sum() < 3:
        return
    subset = adata[tumor_mask].copy()
    n_neighbors = min(15, subset.n_obs - 1)
    sc.pp.neighbors(subset, use_rep="X_sigma", n_neighbors=n_neighbors, metric="cosine")
    sc.tl.leiden(subset, resolution=resolution, key_added="tumor_subtype", random_state=1234)
    adata.obs.loc[subset.obs_names, "tumor_subtype"] = subset.obs["tumor_subtype"].astype(str).to_numpy()


def _cluster_full_embeddings(adata: ad.AnnData, resolution: float) -> None:
    if adata.n_obs >= 3 and adata.n_vars >= 3:
        expression = adata.copy()
        n_components = min(50, expression.n_obs - 1, expression.n_vars - 1)
        sc.pp.pca(expression, n_comps=n_components, random_state=1234)
        sc.pp.neighbors(expression, use_rep="X_pca", n_neighbors=min(15, expression.n_obs - 1))
        sc.tl.leiden(expression, resolution=resolution, key_added="expression_leiden",
                     random_state=1234)
        adata.obs["expression_leiden"] = expression.obs["expression_leiden"].astype(str).to_numpy()
    for representation, key in [
        ("X_mcgae_base", "base_embedding_leiden"),
        ("X_sigma", "fused_embedding_leiden"),
    ]:
        if representation not in adata.obsm or adata.n_obs < 3:
            continue
        subset = adata.copy()
        neighbors = min(15, subset.n_obs - 1)
        sc.pp.neighbors(subset, use_rep=representation, n_neighbors=neighbors, metric="cosine")
        sc.tl.leiden(subset, resolution=resolution, key_added=key, random_state=1234)
        adata.obs[key] = subset.obs[key].astype(str).to_numpy()


def _trajectory(adata: ad.AnnData, neighbors: int) -> None:
    expression, genes = get_full_expression(adata)
    gene_index = {gene.upper(): i for i, gene in enumerate(genes)}
    emt = (_centered_signature(expression, gene_index, EMT_GENES)
           - _centered_signature(expression, gene_index, EPITHELIAL_GENES))
    ecm = _centered_signature(expression, gene_index, ECM_GENES)
    coords = np.asarray(adata.obsm["spatial"])[:, :2]
    fibroblasts = adata.obs["marker_cell_type"].astype(str).to_numpy() == "Fibroblast"
    if fibroblasts.any():
        distance = cKDTree(coords[fibroblasts]).query(coords, k=1)[0]
    else:
        distance = np.linalg.norm(coords - np.median(coords, axis=0), axis=1)
    early_score = -_zscore(emt) - _zscore(ecm) + _zscore(distance)
    root = int(np.argmax(early_score))
    adata.obs["emt_score"] = emt
    adata.obs["ecm_score"] = ecm
    adata.obs["fibroblast_distance"] = distance
    adata.obs["early_root_score"] = early_score
    adata.uns["dpt_root_index"] = root
    adata.uns["trajectory_direction"] = "0 is the selected early root; 1 is the high-DPT end."

    # Embedding dimensions differ from the number of measured genes.
    work = ad.AnnData(X=adata.obsm["X_sigma"].copy(), obs=adata.obs.copy())
    n_neighbors = min(max(2, neighbors), adata.n_obs - 1)
    sc.pp.neighbors(work, n_neighbors=n_neighbors, use_rep="X", random_state=1234)
    state = adata.obs["marker_cell_type"].astype(str).to_numpy()
    work.obs["state_for_paga"] = pd.Categorical(state)
    try:
        sc.tl.paga(work, groups="state_for_paga")
        adata.uns["paga_connectivities"] = work.uns["paga"]["connectivities"]
        adata.uns["paga_state_names"] = work.obs["state_for_paga"].cat.categories.astype(str).tolist()
    except (ValueError, KeyError, RuntimeError):
        adata.uns["paga_connectivities"] = np.zeros((0, 0), dtype=np.float32)
        adata.uns["paga_state_names"] = []
    try:
        sc.tl.diffmap(work)
        work.uns["iroot"] = root
        sc.tl.dpt(work, n_dcs=min(10, max(2, adata.n_obs - 1)))
        pseudotime = np.asarray(work.obs["dpt_pseudotime"], dtype=np.float64)
        if np.isfinite(pseudotime).any():
            low, high = np.nanmin(pseudotime), np.nanmax(pseudotime)
            invasion = (pseudotime - low) / max(high - low, 1e-8)
        else:
            invasion = np.full(adata.n_obs, np.nan)
    except (ValueError, KeyError, RuntimeError, IndexError):
        invasion = 1.0 - _rank01(early_score)
        adata.uns["trajectory_fallback"] = "Rank-normalized marker and distance root score."
    adata.obs["dpt_pseudotime"] = invasion
    adata.obs["invasion_progress"] = invasion


def _spatial_field(adata: ad.AnnData, output_dir: Path, grid_size: int,
                   neighbors: int, bandwidth: float) -> None:
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)[:, :2]
    values = np.asarray(adata.obs["invasion_progress"], dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(coords).all(axis=1)
    if valid.sum() < 3:
        return
    coords = coords[valid]
    values = values[valid]
    size = max(8, int(grid_size))
    x_grid = np.linspace(coords[:, 0].min(), coords[:, 0].max(), size)
    y_grid = np.linspace(coords[:, 1].min(), coords[:, 1].max(), size)
    gx, gy = np.meshgrid(x_grid, y_grid)
    points = np.column_stack([gx.ravel(), gy.ravel()])
    k = min(max(1, neighbors), len(coords))
    distances, indices = cKDTree(coords).query(points, k=k)
    if k == 1:
        distances, indices = distances[:, None], indices[:, None]
    scale = max(float(bandwidth), 1e-6)
    weights = np.exp(-0.5 * (distances / scale) ** 2)
    p = (weights * values[indices]).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-12)
    p = p.reshape(size, size)
    u = 1.0 - p
    dy = float(y_grid[1] - y_grid[0])
    dx = float(x_grid[1] - x_grid[0])
    grad_y, grad_x = np.gradient(p, dy, dx)
    vx, vy = grad_x, grad_y
    field = pd.DataFrame({
        "x": gx.ravel(), "y": gy.ravel(), "invasion_potential": p.ravel(),
        "complementary_potential": u.ravel(), "velocity_x": vx.ravel(), "velocity_y": vy.ravel(),
    })
    field.to_csv(output_dir / "spatial_invasion_field.csv", index=False)
    np.savez_compressed(output_dir / "spatial_invasion_field.npz", x=gx, y=gy,
                        invasion_potential=p, complementary_potential=u,
                        velocity_x=vx, velocity_y=vy)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].scatter(coords[:, 0], coords[:, 1], c=values, s=7, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("Spot-level invasion progress")
    axes[1].streamplot(x_grid, y_grid, vx, vy, color=p, cmap="viridis", density=1.0)
    axes[1].set_title("Spatial gradient field")
    for axis in axes:
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.set_xlabel("Spatial x")
        axis.set_ylabel("Spatial y")
    fig.savefig(output_dir / "spatial_invasion_field.png", dpi=200)
    plt.close(fig)


def _evaluate_clusters(adata: ad.AnnData, truth_key: str) -> dict[str, object]:
    result: dict[str, object] = {"truth_key": truth_key, "n_observations": int(adata.n_obs)}
    if truth_key not in adata.obs:
        result["status"] = "Ground-truth labels are absent; clustering metrics were not computed."
        return result
    truth = adata.obs[truth_key].astype(str).to_numpy()
    metrics = {}
    for key in ["sigma_cluster", "expression_leiden", "base_embedding_leiden",
                "fused_embedding_leiden", "tumor_subtype"]:
        if key not in adata.obs:
            continue
        predicted = adata.obs[key].astype(str).to_numpy()
        metrics[key] = {
            "ari": float(adjusted_rand_score(truth, predicted)),
            "nmi": float(normalized_mutual_info_score(truth, predicted)),
            "ami": float(adjusted_mutual_info_score(truth, predicted)),
        }
        mapped = _majority_map(truth, predicted)
        metrics[key]["majority_mapped_ari_descriptive_only"] = float(adjusted_rand_score(truth, mapped))
        metrics[key]["majority_mapped_nmi_descriptive_only"] = float(normalized_mutual_info_score(truth, mapped))
        metrics[key]["mapping_report"] = classification_report(truth, mapped, output_dict=True,
                                                                 zero_division=0)
    result["metrics"] = metrics
    result["label_mapping_note"] = (
        "Majority-mapped scores use the same labels to construct the mapping and are descriptive, "
        "not independent held-out prediction estimates."
    )
    return result


def _majority_map(truth: np.ndarray, cluster: np.ndarray) -> np.ndarray:
    mapping = {}
    for name in np.unique(cluster):
        labels, counts = np.unique(truth[cluster == name], return_counts=True)
        mapping[name] = labels[np.argmax(counts)]
    return np.asarray([mapping[name] for name in cluster])


def compare_slingshot(pseudotime_path: str | Path, adata: ad.AnnData,
                      output_path: str | Path, id_column: str = "spot_id",
                      time_column: str = "pseudotime") -> dict[str, float]:
    table = pd.read_csv(pseudotime_path)
    if id_column not in table or time_column not in table:
        raise KeyError(f"Slingshot table must contain '{id_column}' and '{time_column}'.")
    left = pd.DataFrame({"id": adata.obs_names.astype(str),
                         "sigma": np.asarray(adata.obs["invasion_progress"], dtype=float)})
    right = table[[id_column, time_column]].copy()
    right[id_column] = right[id_column].astype(str).str.strip()
    joined = left.merge(right, left_on="id", right_on=id_column, how="inner").dropna()
    if len(joined) < 3:
        raise ValueError("At least three matched observations are required for trajectory comparison.")
    rho, pvalue = spearmanr(joined["sigma"], joined[time_column])
    result = {"n_matched": int(len(joined)), "spearman_rho": float(rho), "p_value": float(pvalue)}
    save_json(output_path, result)
    return result


def _centered_signature(expression: np.ndarray, gene_index: dict[str, int],
                        genes: list[str]) -> np.ndarray:
    columns = [gene_index[g] for g in genes if g in gene_index]
    if not columns:
        return np.zeros(expression.shape[0], dtype=np.float64)
    values = expression[:, columns]
    return (values - values.mean(axis=0, keepdims=True)).mean(axis=1)


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - np.nanmean(values)) / (np.nanstd(values) + 1e-8)


def _rank01(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values, method="average")
    return (ranks - 1.0) / max(len(ranks) - 1, 1)

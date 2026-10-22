"""Executable teaching versions of the current report's reference constructions.

These functions consume synthetic teaching inputs. They do not rerun the report's
experimental comparisons or replace the main pipeline's analysis defaults.
"""

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import rankdata


REPORT_EMT = ["VIM", "FN1", "MMP9", "TWIST1", "ZEB1"]
REPORT_EPITHELIAL = ["CDH1", "EPCAM"]
REPORT_ECM = ["COL1A1", "COL1A2", "FN1", "TNC"]
REPORT_COMPARTMENTS = {
    "Tumor": ["IFITM1", "SNHG6"],
    "Stroma": ["IGFBP7", "VIM", "LGALS1"],
    "Other": ["ZG16", "ITLN1", "CLCA1"],
}


def annotate_compartments(adata):
    """Illustrate the report's compartment marker programs without reference labels."""
    expression = np.asarray(adata.X)
    genes = adata.var_names.tolist()
    gene_z = (expression - expression.mean(axis=0)) / (expression.std(axis=0) + 1e-6)
    scores = np.column_stack([
        gene_z[:, [genes.index(gene) for gene in program]].mean(axis=1)
        for program in REPORT_COMPARTMENTS.values()
    ])
    names = list(REPORT_COMPARTMENTS)
    labels = np.asarray(names)[scores.argmax(axis=1)]
    adata.obs["compartment"] = pd.Categorical(labels, categories=names)
    adata.obsm["report_marker_scores"] = scores


def morphology_graph(coordinates, features, candidates):
    """Weight symmetric spatial candidates by distance and patch-feature similarity."""
    row, col = candidates.nonzero()
    spatial_distance = np.linalg.norm(coordinates[row] - coordinates[col], axis=1)
    feature_distance = np.linalg.norm(features[row] - features[col], axis=1)
    tau_c = max(float(np.median(spatial_distance)), 1e-8)
    tau_m = max(float(np.median(feature_distance)), 1e-8)
    weights = np.exp(-spatial_distance / tau_c) * np.exp(
        -feature_distance ** 2 / (2 * tau_m ** 2)
    )
    graph = sparse.csr_matrix((weights, (row, col)), shape=candidates.shape)
    return graph.maximum(graph.T), {"spatial_scale": tau_c, "morphology_scale": tau_m}


def _centered_program(expression, gene_names, program):
    indices = [gene_names.index(gene) for gene in program]
    values = expression[:, indices]
    return (values - values.mean(axis=0)).mean(axis=1)


def _zscore(values):
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / max(float(values.std()), 1e-8)


def rooted_pseudotime(adata, neighbors=20):
    """Choose an early tumor state, rank its spots, and run rooted DPT without fallback."""
    expression = np.asarray(adata.X)
    genes = adata.var_names.tolist()
    emt = (_centered_program(expression, genes, REPORT_EMT)
           - _centered_program(expression, genes, REPORT_EPITHELIAL))
    ecm = _centered_program(expression, genes, REPORT_ECM)
    coordinates = np.asarray(adata.obsm["spatial"])
    labels = adata.obs["compartment"].astype(str).to_numpy()
    stromal = labels == "Stroma"
    tumor = labels == "Tumor"
    if not stromal.any() or tumor.sum() < 3:
        raise ValueError("The oriented teaching trajectory requires tumor and stromal anchors.")
    distance = cKDTree(coordinates[stromal]).query(coordinates)[0]
    subset = adata[tumor].copy()
    sc.pp.neighbors(subset, n_neighbors=min(15, subset.n_obs - 1),
                    use_rep="X_sigma", metric="cosine", random_state=1234)
    sc.tl.leiden(subset, resolution=0.4, key_added="tumor_state", random_state=1234)
    states = np.full(adata.n_obs, "NotTumor", dtype=object)
    states[tumor] = subset.obs["tumor_state"].astype(str).to_numpy()
    summaries = pd.DataFrame({"state": states[tumor], "EMT": emt[tumor],
                              "ECM": ecm[tumor], "distance": distance[tumor]})
    summaries = summaries.groupby("state", sort=True).median()
    early = (_zscore(summaries.distance) - _zscore(summaries.EMT)
             - _zscore(summaries.ECM))
    early_state = str(summaries.index[int(np.argmax(early))])
    candidates = np.flatnonzero(tumor & (states == early_state))
    rank_sum = (rankdata(emt[candidates]) + rankdata(ecm[candidates])
                + rankdata(-distance[candidates]))
    root = int(candidates[np.argmin(rank_sum)])
    work = ad.AnnData(X=adata.obsm["X_sigma"].copy())
    sc.pp.neighbors(work, n_neighbors=min(neighbors, adata.n_obs - 1),
                    use_rep="X", random_state=1234)
    groups = np.where(tumor, np.char.add("Tumor ", states.astype(str)), labels)
    work.obs["state"] = pd.Categorical(groups)
    sc.tl.paga(work, groups="state")
    sc.tl.diffmap(work)
    work.uns["iroot"] = root
    sc.tl.dpt(work, n_dcs=min(10, adata.n_obs - 1))
    raw = work.obs["dpt_pseudotime"].to_numpy(dtype=float)
    finite = np.isfinite(raw)
    if finite.sum() < 3 or np.ptp(raw[finite]) <= 0:
        raise ValueError("DPT must yield at least three finite, nonconstant values.")
    progression = np.full(adata.n_obs, np.nan)
    progression[finite] = ((raw[finite] - raw[finite].min()) / np.ptp(raw[finite]))
    assert tumor[root] and states[root] == early_state
    assert np.isclose(progression[root], 0.0)
    adata.obs["emt_score"], adata.obs["ecm_score"] = emt, ecm
    adata.obs["fibroblast_distance"] = distance
    adata.obs["tumor_state"] = states
    adata.obs["invasion_progress"] = progression
    adata.uns["dpt_root_index"] = root
    adata.uns["early_tumor_state"] = early_state
    adata.uns["unreachable_spots"] = int((~finite).sum())
    adata.uns["paga_connectivities"] = work.uns["paga"]["connectivities"]
    return progression


def axis_smoothed_field_input(adata, window=31):
    """Project onto the normal-to-stromal axis and apply the report's centered mean."""
    coordinates = np.asarray(adata.obsm["spatial"])
    labels = adata.obs["compartment"].astype(str).to_numpy()
    if not (labels == "Other").any() or not (labels == "Stroma").any():
        raise ValueError("The teaching axis requires normal/other and stromal anchors.")
    early = coordinates[labels == "Other"].mean(axis=0)
    late = coordinates[labels == "Stroma"].mean(axis=0)
    axis = late - early
    if np.linalg.norm(axis) <= 1e-8:
        raise ValueError("The anatomical anchor centroids coincide.")
    axis /= np.linalg.norm(axis)
    projection = (coordinates - early) @ axis
    values = adata.obs["invasion_progress"].to_numpy(dtype=float)
    valid = np.isfinite(values)
    indices = np.flatnonzero(valid)
    order = indices[np.argsort(projection[valid], kind="stable")]
    smoothed = np.full(len(values), np.nan)
    smoothed[order] = (pd.Series(values[order]).rolling(window, center=True, min_periods=1)
                       .mean().to_numpy())
    work = adata.copy()
    work.obs["invasion_progress"] = smoothed
    adata.obs["axis_smoothed_progress"] = smoothed
    return work, axis

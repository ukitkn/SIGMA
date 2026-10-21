from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from sigma.data import get_full_expression


def knn_graph(coordinates: np.ndarray, k: int, *, include_self: bool = False) -> sparse.csr_matrix:
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[0] < 2:
        raise ValueError("Coordinates must be a two-dimensional array with at least two rows.")
    k = max(1, min(int(k), coordinates.shape[0] - 1))
    neighbors = NearestNeighbors(n_neighbors=k + 1).fit(coordinates)
    distances, indices = neighbors.kneighbors(coordinates)
    rows, cols, values = [], [], []
    for i in range(coordinates.shape[0]):
        for distance, j in zip(distances[i], indices[i], strict=True):
            if i == j:
                continue
            rows.append(i)
            cols.append(int(j))
            values.append(float(np.exp(-distance / (np.median(distances[:, 1:]) + 1e-8))))
    graph = sparse.csr_matrix((values, (rows, cols)), shape=(len(coordinates), len(coordinates)))
    graph = graph.maximum(graph.T)
    if include_self:
        graph.setdiag(1.0)
    graph.eliminate_zeros()
    return graph


def expression_knn_graph(expression: np.ndarray, k: int = 15) -> sparse.csr_matrix:
    """Build a symmetric cosine-neighbor graph from normalized expression."""
    expression = np.asarray(expression, dtype=np.float32)
    if expression.ndim != 2 or expression.shape[0] < 2:
        raise ValueError("Expression must be a two-dimensional array with at least two rows.")
    k = max(1, min(int(k), expression.shape[0] - 1))
    neighbors = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    neighbors.fit(expression)
    distances, indices = neighbors.kneighbors(expression)
    rows, cols, values = [], [], []
    for i in range(expression.shape[0]):
        for distance, j in zip(distances[i], indices[i], strict=True):
            if i == j:
                continue
            rows.append(i)
            cols.append(int(j))
            values.append(float(max(1.0 - distance, 1e-6)))
    graph = sparse.csr_matrix((values, (rows, cols)), shape=(len(expression), len(expression)))
    graph = graph.maximum(graph.T)
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph


def normalize_adjacency(graph: sparse.spmatrix, add_self_loops: bool = True) -> sparse.coo_matrix:
    graph = sparse.coo_matrix(graph, dtype=np.float32)
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    if add_self_loops:
        graph = graph + sparse.eye(graph.shape[0], dtype=np.float32)
    degree = np.asarray(graph.sum(axis=1)).ravel()
    inv_sqrt = np.power(np.maximum(degree, 1e-12), -0.5)
    normalized = sparse.diags(inv_sqrt) @ graph @ sparse.diags(inv_sqrt)
    return normalized.tocoo()


def build_ccc_graph(
    adata: ad.AnnData,
    lr_path: str | Path,
    spatial_graph: sparse.spmatrix,
    *,
    ligand_column: str = "ligand",
    receptor_column: str = "receptor",
    top_k: int = 8,
    permutations: int = 30,
    min_zscore: float = 0.0,
    seed: int = 1234,
) -> tuple[sparse.csr_matrix, dict[str, object]]:
    lr_path = Path(lr_path)
    if not lr_path.is_file():
        raise FileNotFoundError(f"Ligand-receptor table not found: {lr_path}")
    pairs = pd.read_csv(lr_path)
    if ligand_column not in pairs or receptor_column not in pairs:
        raise KeyError(f"Ligand-receptor table must contain '{ligand_column}' and '{receptor_column}'.")
    pairs = pairs[[ligand_column, receptor_column]].dropna().astype(str).drop_duplicates()
    expression, genes = get_full_expression(adata)
    gene_index = {gene.upper(): i for i, gene in enumerate(genes)}
    usable = []
    for ligand, receptor in pairs.itertuples(index=False, name=None):
        ligand_genes = tuple(gene_index[g.strip().upper()] for g in ligand.split("&")
                             if g.strip().upper() in gene_index)
        receptor_genes = tuple(gene_index[g.strip().upper()] for g in receptor.split("&")
                               if g.strip().upper() in gene_index)
        if ligand_genes and receptor_genes:
            usable.append((ligand_genes, receptor_genes))
    if not usable:
        raise ValueError("None of the ligand-receptor genes occur in the input expression matrix.")

    expression = np.expm1(np.maximum(expression, 0.0)).astype(np.float32)
    ligands = np.column_stack([
        _complex_expression(expression, ligand_genes) for ligand_genes, _ in usable
    ])
    receptors = np.column_stack([
        _complex_expression(expression, receptor_genes) for _, receptor_genes in usable
    ])
    candidates = sparse.triu(spatial_graph, k=1).tocoo()
    if candidates.nnz == 0:
        raise ValueError("The spatial graph contains no candidate cell-cell edges.")
    source, target = candidates.row, candidates.col
    observed = np.sum(ligands[source] * receptors[target] + ligands[target] * receptors[source], axis=1)

    rng = np.random.default_rng(seed)
    null_sum = np.zeros_like(observed, dtype=np.float64)
    null_sq = np.zeros_like(observed, dtype=np.float64)
    for _ in range(max(0, permutations)):
        shuffled = rng.permutation(expression.shape[0])
        permuted_receptors = receptors[shuffled]
        scores = np.sum(
            ligands[source] * permuted_receptors[target]
            + ligands[target] * permuted_receptors[source],
            axis=1,
        )
        null_sum += scores
        null_sq += scores.astype(np.float64) ** 2
    if permutations > 1:
        null_mean = null_sum / permutations
        null_std = np.sqrt(np.maximum(null_sq / permutations - null_mean**2, 1e-12))
        scores = (observed - null_mean) / null_std
    else:
        scores = observed

    edge_keep = scores > min_zscore
    if not np.any(edge_keep):
        edge_keep[np.argmax(scores)] = True
    score_graph = sparse.csr_matrix(
        (np.maximum(scores[edge_keep], 1e-6), (source[edge_keep], target[edge_keep])),
        shape=spatial_graph.shape,
    )
    score_graph = score_graph + score_graph.T
    score_graph = _top_k_symmetric(score_graph, top_k)
    metadata = {
        "lr_pairs_input": int(len(pairs)),
        "lr_pairs_used": int(len(usable)),
        "candidate_spatial_edges": int(len(source)),
        "ccc_edges": int(score_graph.nnz // 2),
        "permutations": int(max(0, permutations)),
        "min_zscore": float(min_zscore),
        "score_definition": "Spatially constrained ligand-receptor product sum with receptor-shuffle z-scores.",
    }
    return score_graph.tocsr(), metadata


def _complex_expression(expression: np.ndarray, genes: tuple[int, ...]) -> np.ndarray:
    if len(genes) == 1:
        return expression[:, genes[0]]
    return np.min(expression[:, genes], axis=1)


def _top_k_symmetric(graph: sparse.csr_matrix, top_k: int) -> sparse.csr_matrix:
    graph = graph.tocsr()
    rows, cols, values = [], [], []
    for i in range(graph.shape[0]):
        start, end = graph.indptr[i], graph.indptr[i + 1]
        row_cols = graph.indices[start:end]
        row_data = graph.data[start:end]
        if row_cols.size == 0:
            continue
        keep = np.argsort(row_data)[-max(1, top_k):]
        rows.extend([i] * len(keep))
        cols.extend(row_cols[keep].tolist())
        values.extend(row_data[keep].tolist())
    result = sparse.csr_matrix((values, (rows, cols)), shape=graph.shape)
    result = result.maximum(result.T)
    result.setdiag(0.0)
    result.eliminate_zeros()
    return result

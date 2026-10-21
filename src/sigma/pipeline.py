from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse

from sigma.config import Config, resolve_path
from sigma.data import load_and_preprocess
from sigma.graphs import build_ccc_graph, expression_knn_graph, knn_graph, normalize_adjacency
from sigma.model import SigmaModel, train_model
from sigma.utils import choose_device, save_json, set_seed, to_dense_array


def prepare_data(config: Config, output_dir: Path) -> ad.AnnData:
    data_cfg = config.section("data")
    graph_cfg = config.section("graph")
    project_cfg = config.section("project")
    adata = load_and_preprocess(
        resolve_path(config, data_cfg["input"]),
        counts_layer=data_cfg.get("counts_layer"),
        min_cells=int(data_cfg.get("min_cells", 1)),
        n_hvg=int(data_cfg.get("n_hvg", 3000)),
        target_sum=float(data_cfg.get("target_sum", 10000)),
        truth_key=data_cfg.get("truth_key", "ident.annot"),
        spatial_key=data_cfg.get("spatial_key", "spatial"),
    )
    coordinates = np.asarray(adata.obsm[data_cfg.get("spatial_key", "spatial")])[:, :2]
    spatial = knn_graph(coordinates, int(graph_cfg.get("spatial_neighbors", 8)))
    expression = expression_knn_graph(
        to_dense_array(adata.X), int(graph_cfg.get("expression_neighbors", 15))
    )
    ccc_candidates = knn_graph(coordinates, int(graph_cfg.get("ccc_neighbors", 8)))
    ccc, ccc_meta = build_ccc_graph(
        adata,
        resolve_path(config, graph_cfg["ligand_receptor_csv"]),
        ccc_candidates,
        ligand_column=graph_cfg.get("ligand_column", "ligand"),
        receptor_column=graph_cfg.get("receptor_column", "receptor"),
        top_k=int(graph_cfg.get("ccc_top_k", 8)),
        permutations=int(graph_cfg.get("ccc_permutations", 30)),
        min_zscore=float(graph_cfg.get("ccc_min_zscore", 0.0)),
        seed=int(project_cfg.get("seed", 1234)),
    )
    adata.obsp["spatial_graph"] = spatial
    adata.obsp["expression_graph"] = expression
    adata.obsp["ccc_graph"] = ccc
    adata.uns["ccc_graph_metadata"] = ccc_meta
    adata.uns["project"] = {
        "name": project_cfg.get("name", "SIGMA"),
        "seed": int(project_cfg.get("seed", 1234)),
        "config": str(config.path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = output_dir / "prepared.h5ad"
    adata.write_h5ad(prepared_path, compression="gzip")
    save_json(output_dir / "graph_metadata.json", ccc_meta)
    return adata


def fit_model(config: Config, adata: ad.AnnData, output_dir: Path,
              overrides: dict[str, Any] | None = None) -> ad.AnnData:
    project_cfg = config.section("project")
    model_cfg = {**config.section("model"), **(overrides or {})}
    model_cfg["clusters"] = min(int(model_cfg.get("clusters", 20)), adata.n_obs)
    set_seed(int(project_cfg.get("seed", 1234)))
    device = choose_device(project_cfg.get("device", "auto"))
    x_array = to_dense_array(adata.X)
    if not np.isfinite(x_array).all():
        raise ValueError("Preprocessed gene expression contains non-finite values.")
    spatial = adata.obsp["spatial_graph"]
    ccc = adata.obsp["ccc_graph"]
    expression_graph = adata.obsp.get("expression_graph")
    if expression_graph is None:
        expression_graph = expression_knn_graph(
            x_array, int(config.section("graph").get("expression_neighbors", 15))
        )
        adata.obsp["expression_graph"] = expression_graph
    spatial_adj = normalize_adjacency(spatial)
    ccc_adj = normalize_adjacency(ccc)
    expression_adj = normalize_adjacency(expression_graph)
    x = torch.as_tensor(x_array, dtype=torch.float32, device=device)
    spatial_tensor = _torch_sparse(spatial_adj, device)
    ccc_tensor = _torch_sparse(ccc_adj, device)
    expression_tensor = _torch_sparse(expression_adj, device)
    spatial_edges = _edge_tensor(spatial, device)
    ccc_edges = _edge_tensor(ccc, device)
    model = SigmaModel(
        input_dim=x.shape[1],
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        latent_dim=int(model_cfg.get("latent_dim", 32)),
        layers=int(model_cfg.get("layers", 3)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        clusters=int(model_cfg.get("clusters", 20)),
        fusion=model_cfg.get("fusion", "concat"),
        use_spatial_view=bool(model_cfg.get("use_spatial_view", True)),
        use_ccc_view=bool(model_cfg.get("use_ccc_view", True)),
        jumping_knowledge=bool(model_cfg.get("jumping_knowledge", False)),
    ).to(device)
    trained = train_model(
        model, x, spatial_tensor, ccc_tensor, spatial_edges, ccc_edges,
        epochs=int(model_cfg.get("epochs", 400)),
        learning_rate=float(model_cfg.get("learning_rate", 0.001)),
        weight_decay=float(model_cfg.get("weight_decay", 0.0005)),
        expression_weight=float(model_cfg.get("expression_weight", 1.5)),
        spatial_weight=float(model_cfg.get("spatial_weight", 0.5)),
        cluster_weight=float(model_cfg.get("cluster_weight", 0.1)),
        ccc_weight=(float(model_cfg.get("ccc_weight", 0.1))
                    if bool(model_cfg.get("use_ccc_loss", True)) else 0.0),
        contrastive_weight=float(model_cfg.get("contrastive_weight", 0.1)),
        graph_smoothness_weight=float(model_cfg.get("graph_smoothness_weight", 0.05)),
        cluster_warmup_epochs=int(model_cfg.get("cluster_warmup_epochs", 60)),
        patience=int(model_cfg.get("patience", 50)),
        min_delta=float(model_cfg.get("min_delta", 0.0001)),
        min_lr=float(model_cfg.get("scheduler_min_lr", 0.00001)),
        edge_batch_size=int(model_cfg.get("edge_batch_size", 65536)),
        seed=int(project_cfg.get("seed", 1234)),
        expression_adj=expression_tensor,
        vae_weight=float(model_cfg.get("vae_weight", 0.1)),
    )
    trained.model.eval()
    with torch.no_grad():
        result = trained.model(x, spatial_tensor, ccc_tensor, expression_tensor)
    adata.obsm["X_sigma"] = result["z"].detach().cpu().numpy()
    adata.obsm["X_mcgae_base"] = result["z_base"].detach().cpu().numpy()
    adata.obsm["X_sigma_spatial"] = result["z_spatial"].detach().cpu().numpy()
    adata.obsm["X_sigma_ccc"] = result["z_ccc"].detach().cpu().numpy()
    adata.obs["sigma_cluster"] = result["q"].argmax(dim=1).detach().cpu().numpy().astype(str)
    output_dir.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_dir / "embeddings.h5ad", compression="gzip")
    torch.save({"state_dict": trained.model.state_dict(), "config": model_cfg},
               output_dir / "sigma_model.pt")
    pd.DataFrame(trained.history).to_csv(output_dir / "training_history.csv", index=False)
    save_json(output_dir / "training_summary.json", {
        "best_epoch": trained.best_epoch,
        "epochs_completed": len(trained.history),
        "final_training_objective": trained.history[-1]["total"] if trained.history else None,
        "device": str(device),
        "model": model_cfg,
    })
    return adata


def load_or_prepare(config: Config, output_dir: Path) -> ad.AnnData:
    prepared_path = output_dir / "prepared.h5ad"
    if prepared_path.is_file():
        return sc.read_h5ad(prepared_path)
    return prepare_data(config, output_dir)


def _torch_sparse(matrix: sparse.spmatrix, device: torch.device) -> torch.Tensor:
    coo = matrix.tocoo().astype(np.float32)
    indices = torch.as_tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
    values = torch.as_tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, coo.shape, device=device).coalesce()


def _edge_tensor(graph: sparse.spmatrix, device: torch.device) -> torch.Tensor:
    upper = sparse.triu(graph, k=1).tocoo()
    return torch.as_tensor(np.vstack([upper.row, upper.col]), dtype=torch.long, device=device)

"""Synthetic inputs, presentation plots and checks around the SIGMA implementation."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.nn import functional as F
from lectrace import plot, text

from sigma.analysis import EMT_GENES, ECM_GENES, SIGNATURES, _spatial_field
from sigma.graphs import expression_knn_graph, normalize_adjacency
from sigma.model import SigmaModel, train_model
from sigma.pipeline import _edge_tensor, _torch_sparse
from sigma.utils import set_seed
from _report_protocol import (REPORT_EMT, REPORT_ECM, REPORT_COMPARTMENTS, axis_smoothed_field_input,
                              morphology_graph, rooted_pseudotime)


HERE = Path(__file__).resolve().parent
COLORS = {"Tumor": "#bf4b45", "Stroma": "#258982", "Other": "#8491a5"}


def explain(title, happening, changes, matters):
    text(f"## {title}\n\n**What:** {happening}\n\n"
         f"**Changes:** {changes}\n\n**Why:** {matters}")


def show(figure):
    plot(figure)


def show_intro():
    text("""## Healthcare problem, motivation and approach

**Problem:** Characterize colorectal tumor–stroma compartments and invasion-associated tissue transitions.

**Motivation:** H&E supplies tissue architecture; spatial transcriptomics supplies localized molecular programs; ligand–receptor pairs supply potential communication context.

**Question:** Can these complementary sources improve compartment recovery and support a biologically oriented spatial ordering?

**Approach:** Physical / communication encoders → concat fusion → expression + topology reconstruction → compartments / tumor states → rooted DPT → invasion potential field.

**Report data:** Eight colon/liver sections, 31,296 spots, 36,601 gene features per section. R conversion aligns counts, coordinates and metadata in a Visium-compatible layout; preprocessing retains full-gene expression for biological scoring.

*Executable example: 60 synthetic spots, synthetic patch descriptors and one illustrative LR pair. The report's experimental scores are shown separately at the end.*
""")
    fig, ax = plt.subplots(figsize=(10, 2.8), layout="constrained")
    labels = ["Tissue inputs\nexpression + morphology", "Dual graph encoders\nphysical + CCC",
              "Hybrid reconstruction\nexpression + topology", "Spatial interpretation\nDPT → U, −∇U"]
    for i, label in enumerate(labels):
        ax.text(i * 2.5, 0.5, label, ha="center", va="center", fontsize=11,
                bbox={"boxstyle": "round,pad=0.65", "fc": "#eff5f7", "ec": "#688391"})
        if i < 3:
            ax.annotate("", (i * 2.5 + 1.55, 0.5), (i * 2.5 + 0.95, 0.5),
                        arrowprops={"arrowstyle": "->", "color": "#34495e"})
    ax.set(xlim=(-1.2, 8.8), ylim=(0, 1))
    ax.axis("off")
    show(fig)


class Demo:
    """Keep presentation-only choices outside the five visible execution states."""

    def __init__(self):
        set_seed(1234)
        torch.set_num_threads(1)
        self.scratch = TemporaryDirectory(prefix="sigma-lecture-")
        self.directory = Path(self.scratch.name)
        rng = np.random.default_rng(1234)
        self.coordinates = np.array([(x * 100, y * 100) for y in range(6) for x in range(10)],
                                    dtype=np.float32)
        column = self.coordinates[:, 0] / 100
        tumor, stroma, normal = (column >= 2) & (column <= 6), column >= 7, column <= 1
        self.genes = sorted(set(sum(SIGNATURES.values(), []) + EMT_GENES + ECM_GENES
                                + REPORT_EMT + REPORT_ECM + sum(REPORT_COMPARTMENTS.values(), [])
                                + ["CDH1", "CXCL12", "CXCR4"]))
        # Smooth synthetic programs expose a continuum rather than three disconnected islands.
        normal_strength = np.clip((3 - column) / 3, 0, 1)
        tumor_strength = np.clip(1 - np.abs(column - 4) / 4, 0, 1)
        stroma_strength = np.clip((column - 4) / 5, 0, 1)
        rates = np.full((60, len(self.genes)), 1.5)
        for label, mask in [("TumorEpithelial", tumor_strength), ("Fibroblast", stroma_strength),
                            ("NormalEpithelial", normal_strength)]:
            indices = [self.genes.index(g) for g in SIGNATURES[label]]
            rates[:, indices] += 24 * mask[:, None]
        for label, mask in [("Tumor", tumor_strength), ("Stroma", stroma_strength), ("Other", normal_strength)]:
            indices = [self.genes.index(gene) for gene in REPORT_COMPARTMENTS[label]]
            rates[:, indices] += 24 * mask[:, None]
        for gene in set(EMT_GENES + ECM_GENES):
            rates[:, self.genes.index(gene)] += 5 * column / 9
        rates[:, self.genes.index("CDH1")] += 12 * (normal | tumor)
        rates[:, self.genes.index("CXCL12")] += 24 * stroma_strength + 8 * (column == 6)
        rates[:, self.genes.index("CXCR4")] += 20 * tumor_strength + 6 * stroma_strength
        rates *= rng.uniform(0.8, 1.2, (60, 1))
        counts = rng.poisson(rates).astype(np.float32)
        self.adata = ad.AnnData(counts)
        self.adata.var_names = self.genes
        self.adata.obs_names = [f"spot_{i:02d}" for i in range(60)]
        self.adata.obsm["spatial"] = self.coordinates.copy()
        self.adata.layers["counts"] = counts.copy()
        sc.pp.normalize_total(self.adata, target_sum=10000)
        sc.pp.log1p(self.adata)
        self.adata.raw = self.adata.copy()
        self.lr_path = self.directory / "teaching_lr.csv"
        pd.DataFrame({"ligand": ["CXCL12"], "receptor": ["CXCR4"]}).to_csv(
            self.lr_path, index=False)
        # Synthetic patch descriptors illustrate morphology weighting, not a real H&E extractor.
        patch_rng = np.random.default_rng(5678)
        self.morphology = np.column_stack([tumor_strength, stroma_strength, normal_strength]).astype(np.float32)
        self.morphology += patch_rng.normal(0, 0.04, self.morphology.shape)
        self.adata.obsm["teaching_patch_features"] = self.morphology
        self.selected = np.array([32, 36, 38])
        plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                             "axes.spines.right": False, "figure.facecolor": "white"})

    def physical_graph(self, candidates):
        graph, self.graph_scales = morphology_graph(
            self.coordinates, self.morphology, candidates)
        return graph

    def rooted_pseudotime(self):
        return rooted_pseudotime(self.adata, neighbors=20)

    def fit(self, physical, communication):
        self.physical, self.communication = physical, communication
        expression = expression_knn_graph(self.adata.X, k=15)
        self.x = torch.as_tensor(np.asarray(self.adata.X), dtype=torch.float32)
        device = torch.device("cpu")
        self.a_sp, self.a_ccc, self.a_expr = [
            _torch_sparse(normalize_adjacency(g), device)
            for g in (physical, communication, expression)
        ]
        self.model = SigmaModel(input_dim=self.x.shape[1], hidden_dim=16, latent_dim=8,
                                layers=3, dropout=0.2, clusters=3, fusion="concat",
                                jumping_knowledge=False)
        trained = train_model(
            self.model, self.x, self.a_sp, self.a_ccc,
            _edge_tensor(physical, device), _edge_tensor(communication, device),
            epochs=80, learning_rate=0.001, weight_decay=0.0005,
            expression_weight=1.5, spatial_weight=0.5, cluster_weight=0.1,
            ccc_weight=0.1, contrastive_weight=0.1, graph_smoothness_weight=0.05,
            cluster_warmup_epochs=60, patience=80, min_delta=0.0001,
            min_lr=0.00001, edge_batch_size=2048, seed=1234,
            expression_adj=self.a_expr, vae_weight=0.1,
        )
        self.history = trained.history
        self.model.eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.optimizer.zero_grad(set_to_none=True)
        source, target = np.triu_indices(60, k=1)
        self.edge_source = torch.as_tensor(source, dtype=torch.long)
        self.edge_target = torch.as_tensor(target, dtype=torch.long)
        labels = (communication.toarray()[source, target] > 0).astype(np.float32)
        self.edge_labels = torch.from_numpy(labels)
        self.positive_weight = torch.tensor((len(labels) - labels.sum()) / labels.sum())
        return self.model, self.x, self.a_sp, self.a_ccc, self.a_expr

    def forward(self):
        return self.model(self.x, self.a_sp, self.a_ccc, self.a_expr)

    def measure_update(self, loss_before, z_before):
        with torch.no_grad():
            output = self.forward()
            logits = self.model.ccc_logits(output["z"], self.edge_source, self.edge_target)
            after = 1.5 * F.mse_loss(output["x_reconstruction"], self.x) + 0.1 * (
                F.binary_cross_entropy_with_logits(logits, self.edge_labels,
                                                   pos_weight=self.positive_weight))
            result = {"selected_loss_before": float(loss_before.detach()),
                      "selected_loss_after": float(after),
                      "embedding_change_L2": float(torch.linalg.vector_norm(output["z"] - z_before))}
        assert result["embedding_change_L2"] > 0
        assert result["selected_loss_after"] < result["selected_loss_before"]
        return {key: round(value, 5) for key, value in result.items()}

    def attach_embeddings(self, output, cluster_id):
        self.adata.obsm["X_sigma"] = output["z"].detach().numpy()
        self.adata.obsm["X_mcgae_base"] = output["z_base"].detach().numpy()
        self.adata.obs["sigma_cluster"] = cluster_id.detach().numpy().astype(str)

    def spatial_field(self):
        work, axis = axis_smoothed_field_input(self.adata, window=31)
        _spatial_field(work, self.directory, grid_size=80, neighbors=80, bandwidth=80)
        self.field_table = pd.read_csv(self.directory / "spatial_invasion_field.csv")
        root = self.adata.uns["dpt_root_index"]
        return {"root_spot": str(self.adata.obs_names[root]),
                "early_tumor_state": self.adata.uns["early_tumor_state"],
                "unreachable_spots": self.adata.uns["unreachable_spots"],
                "anatomical_axis": np.round(axis, 3).tolist(),
                "moving_average_window": 31, "grid_shape": [80, 80],
                "neighbors": min(80, self.adata.n_obs), "bandwidth": 80,
                "potential": "U = 1 - P", "direction": "F = -gradient(U) = gradient(P)",
                "units": "relative score / input coordinate"}

    def _tissue_axes(self, axes):
        for ax in np.atleast_1d(axes):
            ax.set(xlabel="Spatial x", ylabel="Spatial y")
            ax.set_aspect("equal")
            ax.invert_yaxis()

    def graph_figure(self, physical, communication):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), layout="constrained")
        for ax, graph, name in zip(axes[:2], (physical, communication), ("Physical graph", "CCC graph")):
            source, target = graph.nonzero()
            lines = self.coordinates[np.column_stack([source, target])]
            ax.add_collection(LineCollection(lines, colors="#7b8a99", alpha=0.35, linewidths=0.7))
            ax.scatter(*self.coordinates.T, c=self.coordinates[:, 0], cmap="cividis", s=24, zorder=2)
            ax.autoscale()
            ax.set_title(f"{name}: {graph.nnz // 2} edges")
        i, j = np.unravel_index(np.argmax(communication.toarray()), communication.shape)
        normalized = np.expm1(self.adata.X)
        li, ri = self.genes.index("CXCL12"), self.genes.index("CXCR4")
        product = normalized[i, li] * normalized[j, ri] + normalized[j, li] * normalized[i, ri]
        rows = [["Spot pair", f"{i} ↔ {j}"], ["L(i)R(j) + L(j)R(i)", f"{product:.0f}"],
                ["Shuffle z-score", f"{communication[i, j]:.3f}"],
                ["Receptor shuffles", "30"], ["Graph direction", "Symmetrized"]]
        axes[2].axis("off")
        table = axes[2].table(cellText=rows, colLabels=["Observed quantity", "Value"],
                              loc="center", cellLoc="left", colWidths=[0.68, 0.32])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.7)
        self._tissue_axes(axes[:2])
        fig.suptitle("Synthetic expression + patch descriptors · one teaching LR pair · symmetric graphs")
        return fig

    def embedding_figure(self, base, spatial, ccc, fused):
        fig, axes = plt.subplots(1, 4, figsize=(11, 3), layout="constrained")
        states = [tensor.detach().numpy()[self.selected] for tensor in (base, spatial, ccc, fused)]
        limit = max(np.abs(state).max() for state in states)
        for ax, values, title in zip(axes, states, ("Base: Z₀", "Physical: Zsp", "Communication: Zccc", "Fused: Z")):
            ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
            ax.set(title=title, xlabel="Latent component", yticks=range(3),
                   yticklabels=[f"spot {i}" for i in self.selected])
        assert torch.allclose(fused, self.model.fusion_layer(torch.cat([spatial, ccc], 1)).relu())
        fig.suptitle("Actual SigmaModel tensors · 60 × 8 per view · 3 representative spots")
        return fig

    def decoder_figure(self, encoded, edge_logits, update):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), layout="constrained")
        observed = self.x.detach().numpy()[self.selected[1], :8]
        predicted = encoded["x_reconstruction"].detach().numpy()[self.selected[1], :8]
        axes[0].plot(observed, "o-", label="Input X")
        axes[0].plot(predicted, "s--", label="Decoded X")
        axes[0].set(title="Expression reconstruction", xlabel="First 8 genes", ylabel="log expression")
        axes[0].legend(fontsize=8)
        probability = edge_logits.detach().sigmoid().numpy()
        edge_labels = self.edge_labels.numpy().astype(bool)
        axes[1].hist(probability[~edge_labels], bins=15, alpha=0.6, label="Non-edge", density=True)
        axes[1].hist(probability[edge_labels], bins=15, alpha=0.6, label="CCC edge", density=True)
        axes[1].set(title="Bilinear decoder: sigmoid(zᵢᵀWzⱼ)", xlabel="Reconstructed edge probability")
        axes[1].legend(fontsize=8)
        axes[2].bar(["Before", "After"], [update["selected_loss_before"], update["selected_loss_after"]],
                    color=["#8491a5", "#258982"])
        axes[2].set(title="One extra gradient update", ylabel="Selected reconstruction objective")
        for i, key in enumerate(["selected_loss_before", "selected_loss_after"]):
            axes[2].text(i, update[key] + 0.35, f"{update[key]:.4f}", ha="center", fontsize=9)
        delta = update["selected_loss_after"] - update["selected_loss_before"]
        axes[2].text(0.5, 0.6, f"Δ loss = {delta:.4f}\n‖ΔZ‖ = {update['embedding_change_L2']:.4f}",
                     ha="center", transform=axes[2].transAxes, fontsize=10)
        fig.suptitle("Selected expression + CCC loss terms · other full-training terms are not stepped through")
        return fig

    def compartment_figure(self, q, cluster_id):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), layout="constrained")
        probabilities = q.detach().numpy()[self.selected]
        axes[0].imshow(probabilities, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        axes[0].set(title="Student-t soft assignments", xlabel="Cluster index", xticks=range(3),
                    yticks=range(3), yticklabels=[f"spot {i}" for i in self.selected])
        axes[1].scatter(*self.coordinates.T, c=cluster_id.detach().numpy(), cmap="Set2", s=38)
        axes[1].set_title("Unsupervised cluster IDs")
        labels = self.adata.obs["compartment"].astype(str).to_numpy()
        for label, color in COLORS.items():
            mask = labels == label
            axes[2].scatter(*self.coordinates[mask].T, color=color, s=38, label=label)
        axes[2].set_title("Separate marker-based annotations")
        axes[2].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22),
                       ncol=3, frameon=False)
        self._tissue_axes(axes[1:])
        assert np.allclose(q.detach().numpy().sum(axis=1), 1, atol=1e-6)
        fig.suptitle("Synthetic demonstration · cluster confidence and cell-type probability are different quantities")
        return fig

    def field_figure(self, field):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
        values = self.adata.obs["invasion_progress"].to_numpy()
        points = axes[0].scatter(*self.coordinates.T, c=values, cmap="viridis", vmin=0, vmax=1, s=42)
        root = self.adata.uns["dpt_root_index"]
        axes[0].scatter(*self.coordinates[root], marker="*", c="#e74c3c", s=170, label="Selected root")
        axes[0].set_title("DPT: root-relative graph ordering")
        axes[0].legend(fontsize=8)
        table = self.field_table
        p = table.invasion_potential.to_numpy().reshape(80, 80)
        vx = table.velocity_x.to_numpy().reshape(80, 80)
        vy = table.velocity_y.to_numpy().reshape(80, 80)
        xx, yy = table.x.to_numpy().reshape(80, 80), table.y.to_numpy().reshape(80, 80)
        axes[1].contourf(xx, yy, p, levels=np.linspace(0, 1, 11), cmap="viridis", vmin=0, vmax=1)
        axes[1].quiver(xx[::10, ::10], yy[::10, ::10], vx[::10, ::10], vy[::10, ::10], color="white")
        axes[1].set_title("P = 1 − U; arrows: F = −∇U")
        self._tissue_axes(axes)
        fig.colorbar(points, ax=list(axes), label="Relative invasion-associated score", shrink=0.8)
        fig.suptitle("State-guided root → DPT → anatomical-axis smoothing → Gaussian field")
        return fig


def show_results():
    benchmark = pd.read_csv(HERE / "evidence/benchmark.csv")
    ablation = pd.read_csv(HERE / "evidence/ablation.csv")
    ranking = pd.read_csv(HERE / "evidence/configuration_top3.csv")
    means = benchmark[["stLearn", "Seurat", "MCGAE", "SIGMA"]].mean()
    base = ablation[(ablation.backbone == "Upgraded") & (ablation.integration == "Physical only")].iloc[0]
    hybrid = ablation[(ablation.backbone == "Upgraded") & (ablation.integration == "Hybrid")].iloc[0]
    selected = ranking[(ranking.backbone == "Upgraded") & (ranking.integration == "Hybrid")]
    assert round(means.SIGMA, 4) == 0.4901 and round(means.MCGAE, 4) == 0.4204
    text(f"""## Key findings and healthcare interpretation

**ST-colon1 ablation:** Three-layer SIGMA leads all eight variants: ARI **{hybrid.ARI:.4f}**,
NMI **{hybrid.NMI:.4f}**, Macro Dice **{hybrid.Macro_Dice:.4f}**, IoU **{hybrid.IoU:.4f}**.
Matching Base: ARI **{base.ARI:.4f}**, Dice **{base.Macro_Dice:.4f}**.

**Eight-section benchmark:** Mean ARI **{means.SIGMA:.4f}** (SIGMA) versus **{means.MCGAE:.4f}** (MCGAE).
ST-colon1 here is **0.6775**, separate from ablation **0.7328**.
Separate configuration ranking: upgraded Hybrid first on **{int((selected['rank'] == 1).sum())} of 7**,
top three on **{len(selected)} of 7**; these settings are not pooled.

**What we learned:** Communication and encoder capacity must be designed together:
two-layer Hybrid falls below its baseline; upgraded Hybrid performs best.

**Biological checks (report):** **1,665 tumor-core spots**; EMT **r = 0.780**,
ECM **r = 0.765**, nearest-CAF distance **r = −0.286**;
shared-embedding Slingshot–DPT **ρ ≈ 0.85**. These support interpretation without validating direction.

**Healthcare meaning:** Rooted DPT and the potential field turn static molecular patterns into
candidate invasion-associated transitions for pathology review.

**Limits and next steps:** Mixed spots, inferred LR edges, root choice and smoothing affect interpretation.
Test seeds, module ablations and field sensitivity; validate candidate routes with independent pathology,
longitudinal data or functional experiments.

*Report results are separate from this synthetic teaching execution.*
""")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5), layout="constrained")
    axes[0].bar(means.index, means.values, color=["#aab8c2", "#8fa2af", "#526d82", "#258982"])
    axes[0].set(title="Eight-section benchmark: mean ARI", ylabel="ARI", ylim=(0, 0.8))
    for i, value in enumerate(means):
        axes[0].text(i, value + 0.015, f"{value:.4f}", ha="center", fontsize=9)
    metrics = ["ARI", "NMI", "Macro_Dice", "IoU"]
    positions = np.arange(4)
    axes[1].bar(positions - 0.18, base[metrics].to_numpy(dtype=float), 0.36,
                label="Upgraded Base", color="#8491a5")
    bars = axes[1].bar(positions + 0.18, hybrid[metrics].to_numpy(dtype=float), 0.36,
                       label="Upgraded SIGMA", color="#258982")
    axes[1].bar_label(bars, labels=[f"{hybrid[m]:.4f}" for m in metrics], fontsize=8, padding=3)
    axes[1].set(title="ST-colon1 ablation: matching backbone", ylim=(0, 1),
                xticks=positions, xticklabels=["ARI", "NMI", "Macro Dice", "IoU"])
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Report results · two distinct comparisons · the teaching run does not reproduce these scores")
    show(fig)
    reference = json.loads((HERE / "evidence/report_reference.json").read_text())
    text("## Software 3.0 assistance\n\n" + reference["software_3_assistance"])

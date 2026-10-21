from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F


class GraphConvolution(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = torch.sparse.mm(adjacency, self.linear(x))
        hidden = self.norm(hidden)
        return self.dropout(F.relu(hidden) + self.residual(x))


class GraphEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layers: int,
                 dropout: float, jumping_knowledge: bool = False) -> None:
        super().__init__()
        if layers < 2:
            raise ValueError("Graph encoders require at least two layers.")
        dimensions = [input_dim] + [hidden_dim] * (layers - 1) + [output_dim]
        self.blocks = nn.ModuleList(
            GraphConvolution(dimensions[i], dimensions[i + 1], dropout)
            for i in range(layers)
        )
        self.jumping_knowledge = jumping_knowledge
        self.jump = nn.Linear(sum(dimensions[1:]), output_dim) if jumping_knowledge else None

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        states = []
        for block in self.blocks:
            x = block(x, adjacency)
            states.append(x)
        if self.jump is not None:
            return self.jump(torch.cat(states, dim=1))
        return x


class FeatureVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, latent_dim * 2), nn.GELU(),
                                     nn.Dropout(dropout))
        self.mean = nn.Linear(latent_dim * 2, latent_dim)
        self.log_variance = nn.Linear(latent_dim * 2, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        mean = self.mean(hidden)
        log_variance = self.log_variance(hidden).clamp(-8.0, 8.0)
        if self.training:
            latent = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
        else:
            latent = mean
        return latent, mean, log_variance, self.decoder(latent)


class ViewAttention(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(latent_dim, max(8, latent_dim // 2)), nn.Tanh(),
                                   nn.Linear(max(8, latent_dim // 2), 1, bias=False))

    def forward(self, views: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack(views, dim=1)
        weights = torch.softmax(self.score(stacked).squeeze(-1), dim=1)
        return (stacked * weights.unsqueeze(-1)).sum(dim=1), weights


class SigmaModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, layers: int,
                 dropout: float, clusters: int, fusion: str = "concat",
                 use_spatial_view: bool = True, use_ccc_view: bool = True,
                 jumping_knowledge: bool = False) -> None:
        super().__init__()
        if fusion not in {"concat", "attention"}:
            raise ValueError("fusion must be 'concat' or 'attention'.")
        self.use_spatial_view = use_spatial_view
        self.use_ccc_view = use_ccc_view
        self.fusion = fusion
        self.original_encoder = GraphEncoder(input_dim, hidden_dim, latent_dim, layers, dropout,
                                             jumping_knowledge)
        self.feature_vae = FeatureVAE(input_dim, latent_dim, dropout)
        self.augmented_feature_encoder = GraphEncoder(latent_dim, hidden_dim, latent_dim, layers,
                                                       dropout, jumping_knowledge)
        self.augmented_graph_encoder = GraphEncoder(input_dim, hidden_dim, latent_dim, layers,
                                                     dropout, jumping_knowledge)
        self.base_pair_attention = ViewAttention(latent_dim)
        self.base_attention = ViewAttention(latent_dim)
        self.corruption_rate = 0.2
        self.spatial_encoder = GraphEncoder(latent_dim, hidden_dim, latent_dim, layers, dropout,
                                            jumping_knowledge)
        self.ccc_encoder = GraphEncoder(latent_dim, hidden_dim, latent_dim, layers, dropout,
                                        jumping_knowledge)
        self.gate = nn.Linear(latent_dim * 2, 2) if fusion == "attention" else None
        self.fusion_layer = nn.Linear(latent_dim * 2, latent_dim)
        self.expression_decoder = nn.Linear(latent_dim, input_dim)
        self.cluster_centers = nn.Parameter(torch.empty(clusters, latent_dim))
        nn.init.xavier_uniform_(self.cluster_centers)
        self.ccc_bilinear = nn.Parameter(torch.empty(latent_dim, latent_dim))
        nn.init.xavier_uniform_(self.ccc_bilinear)

    def forward(self, x: torch.Tensor, spatial_adj: torch.Tensor,
                ccc_adj: torch.Tensor, expression_adj: torch.Tensor | None = None
                ) -> dict[str, torch.Tensor]:
        if expression_adj is None:
            expression_adj = spatial_adj
        z_original = self.original_encoder(x, spatial_adj)
        if self.training:
            keep = torch.rand_like(x) >= self.corruption_rate
            corrupted = x * keep
        else:
            corrupted = x
        z_corrupted = self.original_encoder(corrupted, spatial_adj)
        vae_features, vae_mean, vae_log_variance, vae_reconstruction = self.feature_vae(x)
        z_augmented_feature = self.augmented_feature_encoder(vae_features, spatial_adj)
        z_augmented_graph = self.augmented_graph_encoder(x, expression_adj)
        original_graph, _ = self.base_pair_attention([z_original, z_augmented_graph])
        original_feature, _ = self.base_pair_attention([z_original, z_augmented_feature])
        z_base, base_weights = self.base_attention([original_graph, original_feature])
        z_spatial = self.spatial_encoder(z_base, spatial_adj) if self.use_spatial_view else z_base
        z_ccc = self.ccc_encoder(z_base, ccc_adj) if self.use_ccc_view else z_base
        if self.fusion == "attention":
            weights = torch.softmax(self.gate(torch.cat([z_spatial, z_ccc], dim=1)), dim=1)
            z_dual = weights[:, :1] * z_spatial + weights[:, 1:] * z_ccc
        else:
            z_dual = F.relu(self.fusion_layer(torch.cat([z_spatial, z_ccc], dim=1)))
        q = student_t_assignments(z_dual, self.cluster_centers)
        return {
            "z_base": z_base,
            "z_original": z_original,
            "z_corrupted": z_corrupted,
            "vae_mean": vae_mean,
            "vae_log_variance": vae_log_variance,
            "vae_reconstruction": vae_reconstruction,
            "base_attention": base_weights,
            "z_spatial": z_spatial,
            "z_ccc": z_ccc,
            "z": z_dual,
            "x_reconstruction": self.expression_decoder(z_dual),
            "q": q,
        }

    def ccc_logits(self, z: torch.Tensor, source: torch.Tensor,
                   target: torch.Tensor) -> torch.Tensor:
        transformed = z @ self.ccc_bilinear
        return (transformed[source] * z[target]).sum(dim=1)

    def initialize_clusters(self, embeddings: np.ndarray, seed: int) -> None:
        clusters = self.cluster_centers.shape[0]
        labels = KMeans(n_clusters=clusters, n_init=20, random_state=seed).fit_predict(embeddings)
        centers = np.stack([embeddings[labels == i].mean(axis=0) for i in range(clusters)])
        with torch.no_grad():
            self.cluster_centers.copy_(torch.as_tensor(centers, device=self.cluster_centers.device))


def student_t_assignments(z: torch.Tensor, centers: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    distance = torch.sum((z[:, None, :] - centers[None, :, :]) ** 2, dim=2)
    q = (1.0 + distance / alpha).pow(-(alpha + 1.0) / 2.0)
    return q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)


def target_distribution(q: torch.Tensor) -> torch.Tensor:
    weight = q.square() / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
    return weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-12)


@dataclass
class TrainResult:
    model: SigmaModel
    history: list[dict[str, float]]
    best_epoch: int


def train_model(
    model: SigmaModel,
    x: torch.Tensor,
    spatial_adj: torch.Tensor,
    ccc_adj: torch.Tensor,
    spatial_edges: torch.Tensor,
    ccc_edges: torch.Tensor,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    expression_weight: float,
    spatial_weight: float,
    cluster_weight: float,
    ccc_weight: float,
    contrastive_weight: float,
    graph_smoothness_weight: float,
    cluster_warmup_epochs: int,
    patience: int,
    min_delta: float,
    min_lr: float,
    edge_batch_size: int,
    seed: int = 1234,
    expression_adj: torch.Tensor | None = None,
    vae_weight: float = 0.1,
) -> TrainResult:
    device = x.device
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=min_lr
    )
    model.eval()
    with torch.no_grad():
        initial = model(x, spatial_adj, ccc_adj, expression_adj)["z"].detach().cpu().numpy()
    model.initialize_clusters(initial, seed=seed)

    positives = spatial_edges
    ccc_positives = ccc_edges
    n = x.shape[0]
    target_q = None
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch, stale = 0, 0
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        output = model(x, spatial_adj, ccc_adj, expression_adj)
        if epoch % 10 == 0 or target_q is None:
            target_q = target_distribution(output["q"].detach())
        x_loss = F.mse_loss(output["x_reconstruction"], x)
        spatial_loss = _sampled_graph_loss(model, output["z"], positives, n, edge_batch_size,
                                           graph="spatial")
        ccc_loss = _sampled_graph_loss(model, output["z"], ccc_positives, n, edge_batch_size,
                                      graph="ccc")
        contrastive_loss = _cross_view_contrastive(
            output["z_original"], output["z_corrupted"], batch_size=min(1024, n)
        )
        vae_reconstruction_loss = F.mse_loss(output["vae_reconstruction"], x)
        vae_kl_loss = -0.5 * torch.mean(
            1.0 + output["vae_log_variance"] - output["vae_mean"].square()
            - output["vae_log_variance"].exp()
        )
        cluster_loss = F.kl_div(output["q"].clamp_min(1e-8).log(), target_q,
                                reduction="batchmean")
        smooth_loss = _graph_smoothness(output["z"], positives)
        cluster_scale = 1.0 if epoch >= cluster_warmup_epochs else 0.0
        loss = (
            expression_weight * x_loss
            + spatial_weight * spatial_loss
            + cluster_scale * cluster_weight * cluster_loss
            + ccc_weight * ccc_loss
            + contrastive_weight * contrastive_loss
            + vae_weight * (vae_reconstruction_loss + 0.01 * vae_kl_loss)
            + graph_smoothness_weight * smooth_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        values = {
            "epoch": float(epoch + 1),
            "total": float(loss.detach().cpu()),
            "expression": float(x_loss.detach().cpu()),
            "spatial": float(spatial_loss.detach().cpu()),
            "ccc": float(ccc_loss.detach().cpu()),
            "contrastive": float(contrastive_loss.detach().cpu()),
            "vae_reconstruction": float(vae_reconstruction_loss.detach().cpu()),
            "vae_kl": float(vae_kl_loss.detach().cpu()),
            "cluster": float(cluster_loss.detach().cpu()),
            "smoothness": float(smooth_loss.detach().cpu()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(values)
        score = values["total"]
        if score < best_loss - min_delta:
            best_loss = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    return TrainResult(model=model, history=history, best_epoch=best_epoch)


def _sampled_graph_loss(model: SigmaModel, z: torch.Tensor, edges: torch.Tensor,
                        n_nodes: int, batch_size: int, graph: str) -> torch.Tensor:
    if edges.shape[1] == 0:
        return z.new_zeros(())
    positive_count = min(edges.shape[1], max(1, batch_size // 2))
    positive_idx = torch.randperm(edges.shape[1], device=z.device)[:positive_count]
    positive = edges[:, positive_idx]
    negative_count = positive_count
    negative = torch.randint(0, n_nodes, (2, negative_count), device=z.device)
    mask = negative[0] != negative[1]
    negative = negative[:, mask]
    if graph == "ccc":
        pos_logits = model.ccc_logits(z, positive[0], positive[1])
        neg_logits = model.ccc_logits(z, negative[0], negative[1])
    else:
        pos_logits = (z[positive[0]] * z[positive[1]]).sum(dim=1)
        neg_logits = (z[negative[0]] * z[negative[1]]).sum(dim=1)
    logits = torch.cat([pos_logits, neg_logits])
    labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
    pos_weight = logits.new_tensor(max(1.0, len(neg_logits) / max(len(pos_logits), 1)))
    return F.binary_cross_entropy_with_logits(logits, labels,
                                               pos_weight=pos_weight if graph == "ccc" else None)


def _graph_smoothness(z: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    if edges.shape[1] == 0:
        return z.new_zeros(())
    count = min(edges.shape[1], 32768)
    selected = edges[:, torch.randperm(edges.shape[1], device=z.device)[:count]]
    return (z[selected[0]] - z[selected[1]]).square().sum(dim=1).mean()


def _cross_view_contrastive(first: torch.Tensor, second: torch.Tensor,
                            batch_size: int, temperature: float = 0.2) -> torch.Tensor:
    if batch_size < 2:
        return first.new_zeros(())
    indices = torch.randperm(first.shape[0], device=first.device)[:batch_size]
    first_batch = F.normalize(first[indices], dim=1)
    second_batch = F.normalize(second[indices], dim=1)
    logits = first_batch @ second_batch.T / temperature
    targets = torch.arange(batch_size, device=first.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))

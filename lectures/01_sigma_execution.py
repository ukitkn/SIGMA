"""Five selected SIGMA execution states for a ten-minute presentation."""

import torch
from torch.nn import functional as F
from lectrace import text

from sigma.graphs import build_ccc_graph, knn_graph
from sigma.model import student_t_assignments
from sigma.analysis import _annotate_compartments, _trajectory
from _sigma_demo import Demo, explain, show, show_intro, show_results


def main():
    text("# SIGMA: Learning the Tumor–Stroma Interface")
    show_intro()
    demo = Demo()  # @invisible

    # State 1 | 2:30–3:40 | Expression and location become graph inputs.
    explain("State 1 · Build physical and communication graphs",
            "Normalize a small expression panel; connect nearby tissue spots.",
            "Ligand–receptor scores filter and weight a second neighborhood.",
            "Physical proximity and expression-compatible signaling encode different context.")
    adata, coordinates = demo.adata, demo.coordinates  # @inspect coordinates
    physical = knn_graph(coordinates, k=8)  # @inspect physical
    communication, graph_info = build_ccc_graph(
        adata, demo.lr_path, physical, top_k=8, permutations=30, seed=1234,
    )  # @inspect graph_info
    show(demo.graph_figure(physical, communication))

    # State 2 | 3:40–5:05 | The same base embedding takes two message-passing paths.
    explain("State 2 · Encode two views and fuse their features",
            "An MCGAE-inspired base feeds separate residual GCN branches.",
            "Concatenate the spatial and CCC embeddings, then project with Linear + ReLU.",
            "The final representation can retain both tissue layout and communication context.")
    model, x, a_sp, a_ccc, a_expr = demo.fit(physical, communication)  # @invisible
    encoded = model(x, a_sp, a_ccc, a_expr)
    z_base, z_sp, z_ccc = encoded["z_base"], encoded["z_spatial"], encoded["z_ccc"]
    z = F.relu(model.fusion_layer(torch.cat([z_sp, z_ccc], dim=1)))  # @inspect z_base z_sp z_ccc z
    show(demo.embedding_figure(z_base, z_sp, z_ccc, z))

    # State 3 | 5:05–6:30 | Show one real update, not the complete training trace.
    explain("State 3 · Reconstruct expression and communication topology",
            "Decode gene expression and score edges with a bilinear CCC decoder.",
            "One gradient update changes the encoder and decoder parameters.",
            "The fused embedding must explain measurable features and the inferred graph.")
    x_hat = encoded["x_reconstruction"]
    edge_logits = model.ccc_logits(z, demo.edge_source, demo.edge_target)
    reconstruction = F.mse_loss(x_hat, x)
    topology = F.binary_cross_entropy_with_logits(
        edge_logits, demo.edge_labels, pos_weight=demo.positive_weight,
    )
    loss = 1.5 * reconstruction + 0.1 * topology  # @inspect reconstruction topology loss
    loss.backward()
    demo.optimizer.step()
    update = demo.measure_update(loss, z)  # @inspect update
    show(demo.decoder_figure(encoded, edge_logits, update))

    # State 4 | 6:30–7:35 | Cluster membership is different from a tissue label.
    explain("State 4 · Turn an embedding into interpretable tissue groups",
            "Initialize centers on the learned embedding; Student-t distances give soft assignments.",
            "Marker programs separately annotate Tumor, Stroma and Other spots.",
            "A cluster ID needs biological interpretation; q is not a tumor probability.")
    final = demo.forward()  # @invisible
    model.initialize_clusters(final["z"].detach().numpy(), seed=1234)
    q = student_t_assignments(final["z"], model.cluster_centers)  # @inspect q
    cluster_id = q.argmax(dim=1)  # @inspect cluster_id
    demo.attach_embeddings(final, cluster_id)  # @invisible
    _annotate_compartments(adata, minimum=0.05)
    show(demo.compartment_figure(q, cluster_id))

    # State 5 | 7:35–8:45 | A spatial hypothesis follows from rooted graph ordering.
    explain("State 5 · Convert rooted pseudotime into a spatial score field",
            "Choose a marker-guided early root, then run diffusion pseudotime on the embedding.",
            "Kernel interpolation turns spot scores into P(x,y); the displayed direction is ∇P.",
            "This prioritizes spatial patterns for follow-up; pseudotime is not elapsed time.")
    _trajectory(adata, neighbors=20)
    pseudotime = adata.obs["invasion_progress"].to_numpy()  # @inspect pseudotime
    field = demo.spatial_field()  # @inspect field
    show(demo.field_figure(field))

    # Conclusion | 8:45–10:00 | Archived results are separate from this execution.
    show_results()


if __name__ == "__main__":
    main()

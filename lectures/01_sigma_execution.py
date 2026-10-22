"""Five selected SIGMA execution states for a ten-minute presentation."""

import torch
from torch.nn import functional as F
from lectrace import text

from sigma.graphs import build_ccc_graph, knn_graph
from sigma.model import student_t_assignments
from _report_protocol import annotate_compartments
from _sigma_demo import Demo, explain, show, show_intro, show_results


def main():
    text("# SIGMA: Learning the Tumor–Stroma Interface")
    show_intro()
    demo = Demo()  # @invisible

    # State 1 | 2:30–3:40 | Expression and location become graph inputs.
    explain("State 1 · Build physical and communication graphs",
            "Spatial neighbors are weighted by coordinate distance and synthetic morphology descriptors; nearby pairs receive CXCL12→CXCR4 scores.",
            "Thirty receptor shuffles and top-k selection produce a symmetric, weighted CCC graph.",
            "The physical view represents tissue structure; the CCC view supplies expression-compatible communication context. Patch descriptors here are synthetic, not extracted H&E.")
    adata, coordinates = demo.adata, demo.coordinates  # @inspect coordinates
    spatial_candidates = knn_graph(coordinates, k=8)
    physical = demo.physical_graph(spatial_candidates)  # @inspect physical
    communication, graph_info = build_ccc_graph(
        adata, demo.lr_path, spatial_candidates, top_k=8, permutations=30, seed=1234,
    )  # @inspect graph_info
    show(demo.graph_figure(physical, communication))

    # State 2 | 3:40–5:05 | The same base embedding takes two message-passing paths.
    explain("State 2 · Encode two views and fuse their features",
            "The MCGAE-inspired base feeds separate three-layer residual GCNs for physical and CCC graphs.",
            "The two embeddings are concatenated and projected with Linear + ReLU; this run uses no cross-view attention fusion.",
            "The fused spot representation carries both tissue-neighborhood and communication context.")
    model, x, a_sp, a_ccc, a_expr = demo.fit(physical, communication)  # @invisible
    encoded = model(x, a_sp, a_ccc, a_expr)
    z_base, z_sp, z_ccc = encoded["z_base"], encoded["z_spatial"], encoded["z_ccc"]
    z = F.relu(model.fusion_layer(torch.cat([z_sp, z_ccc], dim=1)))  # @inspect z_base z_sp z_ccc z
    show(demo.embedding_figure(z_base, z_sp, z_ccc, z))

    # State 3 | 5:05–6:30 | Show one real update, not the complete training trace.
    explain("State 3 · Reconstruct expression and communication topology",
            "An expression decoder reconstructs features; a bilinear decoder scores inferred CCC edges.",
            "One visible Adam update uses only the selected expression and CCC loss terms.",
            "The update fits expression and graph structure; the full trainer also uses other objectives.")
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
            "Student-t distances assign each spot softly to three learned cluster centers.",
            "Argmax gives cluster IDs; report marker programs separately annotate Tumor (IFITM1/SNHG6), F/S (IGFBP7/VIM/LGALS1) and normal/other (ZG16/ITLN1/CLCA1).",
            "A cluster ID is not a tumor probability, and marker labels are not independent histology validation.")
    final = demo.forward()  # @invisible
    model.initialize_clusters(final["z"].detach().numpy(), seed=1234)
    q = student_t_assignments(final["z"], model.cluster_centers)  # @inspect q
    cluster_id = q.argmax(dim=1)  # @inspect cluster_id
    demo.attach_embeddings(final, cluster_id)  # @invisible
    annotate_compartments(adata)
    show(demo.compartment_figure(q, cluster_id))

    # State 5 | 7:35–8:45 | A spatial hypothesis follows from rooted graph ordering.
    explain("State 5 · Convert rooted pseudotime into a spatial score field",
            "Tumor states are scored by median EMT, ECM and stromal distance; ranks select an early root for DPT.",
            "Anatomical-axis smoothing and Gaussian interpolation produce P(x,y), U=1−P and F=−∇U.",
            "The field can guide follow-up review; it is not elapsed time, physical motion, or validated patient risk.")
    pseudotime = demo.rooted_pseudotime()  # @inspect pseudotime
    field = demo.spatial_field()  # @inspect field
    show(demo.field_figure(field))

    # Conclusion | 8:45–10:00 | Report results, lessons, limitations and next steps.
    show_results()


if __name__ == "__main__":
    main()

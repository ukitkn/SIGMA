"""A 10-minute LecTrace walkthrough of the SIGMA computation."""

from __future__ import annotations

import math

try:
    from lectrace import note, text
except ImportError:
    text = None
    note = None


def main() -> None:
    text("# SIGMA: From Visium Spots to an Invasion-Associated Tissue Map")
    text(
        "This 10-minute walkthrough follows five selected execution states. The five-spot "
        "matrix is synthetic teaching data, not a measurement from the colon samples or "
        "a clinical prediction. SIGMA asks whether spatial structure and cell-cell "
        "communication can help describe tumor–stroma regions associated with invasion."
    )
    note(
        "0:00–2:20 | Healthcare problem and motivation. Colorectal tumors grow inside a "
        "tissue ecosystem. At the invasive margin, malignant epithelium and surrounding "
        "stroma can exchange signals, remodel extracellular matrix, and create a path for "
        "tumor spread. A tissue-average expression profile can hide that boundary. Visium "
        "adds location to gene expression, so our project asks whether spatial context and "
        "cell-cell communication can help describe tumor and stromal regions and highlight "
        "an invasion-associated transition. SIGMA combines a graph autoencoder inspired "
        "by MCGAE with spatial and ligand–receptor views. We will follow one small, "
        "fictional slice through preprocessing, graph construction, message passing, "
        "fusion, and a tissue-level score. The toy values are for teaching only; they do "
        "not estimate a patient's risk."
    )

    spots = ["T1", "T2", "IF", "S1", "S2"]
    genes = ["EPCAM", "COL1A1", "CXCL12", "CXCR4", "VIM"]
    coordinates = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    counts = [
        [9, 1, 1, 1, 2],
        [7, 2, 2, 2, 4],
        [4, 5, 7, 6, 7],
        [1, 9, 8, 8, 8],
        [1, 8, 4, 4, 6],
    ]  # @inspect spots genes counts coordinates
    text("## State 1 — Each spot becomes a feature vector\nRows are spots; columns are genes. T = tumor-like, IF = interface, S = stroma-like.\n\nCounts: `" + str(counts) + "`\nCoordinates: `" + str(coordinates) + "`")  # @stepover
    note(
        "2:20–3:25 | What is happening: a spatial-transcriptomics spot is represented by "
        "its measured genes and its tissue coordinates. EPCAM is used here as an "
        "epithelial marker, while COL1A1 and VIM illustrate matrix and mesenchymal "
        "programs. What changes: nothing is predicted yet; raw counts are simply organized "
        "as rows and features. Why it matters: the model can only learn from information "
        "that is represented in these vectors. In the full pipeline, the input has many "
        "more genes and thousands of spots, and preprocessing selects highly variable "
        "genes while retaining full-gene expression for marker and communication analysis."
    )
    expression, spatial_edges, expression_edges = prepare_feature_state(counts, coordinates)  # @inspect expression spatial_edges expression_edges @stepover
    text("## State 2 — Normalize expression and build two neighborhood views\nLibrary-size normalization plus `log1p` gives: `" + str(rounded_matrix(expression)) + "`\n\nSpatial edges: `" + str(edge_names(spatial_edges, spots)) + "`\nExpression-similarity edges: `" + str(edge_names(expression_edges, spots)) + "`")  # @stepover
    note(
        "3:25–4:40 | What is happening: each spot is normalized by its library size, "
        "scaled to a common total, and log transformed. We then make two graphs. The "
        "spatial graph connects physically nearby spots. The expression graph connects "
        "spots with similar profiles, even when their coordinates differ. What changes: "
        "the same spot now has both a feature vector and two possible neighborhoods. Why "
        "it matters: spatial adjacency captures tissue layout, while expression "
        "similarity captures molecular resemblance. Keeping these views separate lets "
        "the model compare local anatomy with shared cell programs instead of treating "
        "all neighbors as equivalent."
    )
    communication_edges, communication_scores = communication_graph(expression, spatial_edges, genes.index("CXCL12"), genes.index("CXCR4"))  # @inspect communication_edges communication_scores @stepover
    text("## State 3 — Add a ligand–receptor communication view\nFor CXCL12 → CXCR4, score both directions on spatial candidate edges, then use receptor shuffles for a small null background.\n\nScores: `" + str(edge_score_names(communication_scores, spots)) + "`\nRetained edges: `" + str(edge_names(communication_edges, spots)) + "`")  # @stepover
    note(
        "4:40–5:55 | What is happening: on spatial candidate edges, the demo combines "
        "ligand expression at one spot with receptor expression at its neighbor, and also "
        "checks the reverse direction. It shuffles receptor profiles to estimate a "
        "background and retains stronger communication candidates. What changes: a "
        "spatial edge receives a communication score instead of representing distance "
        "alone. Why it matters: a stromal spot near tumor tissue may be important because "
        "of a plausible signaling relationship, not just because it is close. The full "
        "project uses curated CellPhoneDB pairs, complex subunits, more permutations, and "
        "top-k sparsification; this walkthrough uses one pair and fewer spots for clarity."
    )

    spatial_embedding, communication_embedding, fused_embedding, attention_weights = encode_views(expression, spatial_edges, communication_edges)  # @inspect spatial_embedding communication_embedding attention_weights fused_embedding @stepover
    text("## State 4 — Dual graph message passing and attention fusion\nSeparate encoders aggregate spatial and communication neighbors; an illustrative gate blends their embeddings per spot.\n\nSpatial: `" + str(rounded_matrix(spatial_embedding)) + "`\nCommunication: `" + str(rounded_matrix(communication_embedding)) + "`\nAttention [spatial, communication]: `" + str(rounded_matrix(attention_weights)) + "`\nFused: `" + str(rounded_matrix(fused_embedding)) + "`")  # @stepover
    note(
        "5:55–7:35 | What is happening: separate graph encoders aggregate information "
        "over the spatial and communication graphs. An attention gate then combines the "
        "two embeddings for each spot. What changes: a spot is no longer described only "
        "by its own gene values; its representation also reflects messages from the "
        "neighbors specified by each graph. The gate can weight the views differently "
        "across spots. Why it matters: the tumor interior may rely more on local spatial "
        "structure, while a boundary may gain information from communication partners. "
        "In the full implementation, this sits after the MCGAE-inspired multi-view base "
        "encoder and is trained together with reconstruction, graph, clustering, "
        "contrastive, and smoothness objectives. The fixed arithmetic here is only an "
        "execution illustration, not trained SIGMA weights."
    )

    cluster_probabilities, invasion_score, ranked_spots = summarize_result(fused_embedding, expression, communication_edges, spots, genes)  # @inspect cluster_probabilities invasion_score ranked_spots @stepover
    text("## State 5 — Clustering and an invasion-associated score\nStudent-t assignments compare embeddings with two illustrative centers. The transparent score combines boundary co-expression, VIM, and communication strength.\n\nCluster probabilities: `" + str(rounded_matrix(cluster_probabilities)) + "`\nInvasion-associated ranking: `" + str([(spot, round(score, 3)) for spot, score in ranked_spots]) + "`")  # @stepover
    note(
        "7:35–9:15 | What is happening: the fused embeddings are compared with two "
        "illustrative centers using a Student-t assignment, the same family of soft "
        "assignments used in DEC-style clustering. We also rank spots using a transparent "
        "combination of EPCAM/COL1A1 co-expression, VIM, and communication-edge "
        "strength. What changes: the "
        "model outputs become tissue-level groupings and a spatially interpretable "
        "candidate score. Why it matters: a pathologist or researcher can inspect "
        "whether a transition zone combines tumor-associated features, matrix remodeling, "
        "and signaling. In this toy slice, the mixed interface spot is prioritized for "
        "inspection. This is a hypothesis-generating view, not a validated diagnostic "
        "or a forecast of an individual patient's outcome."
    )

    text("## Conclusion — What this execution teaches us\nSIGMA turns spatial measurements into multiple graphs, learns separate representations from tissue structure and ligand–receptor context, and fuses them before downstream analysis. This can prioritize tumor–stroma transition regions for biological validation; it does not replace pathology or make clinical decisions.")
    note(
        "9:15–10:00 | Close by returning to the healthcare question: spatial structure "
        "and communication may reveal an invasion-associated tumor–stroma boundary that "
        "expression-only analysis misses. Our next scientific step is to run the full "
        "pipeline on the real Visium samples, evaluate segmentation and invasion-related "
        "patterns against available annotations, and validate candidate signals with "
        "independent biological evidence. The example shown today is synthetic, so no "
        "empirical performance claim is being made."
    )


def prepare_feature_state(
    counts: list[list[int]], coordinates: list[tuple[int, int]]
) -> tuple[list[list[float]], list[tuple[int, int]], list[tuple[int, int]]]:
    expression = log_normalize(counts)
    spatial_edges = nearest_edges(coordinates)
    expression_edges = nearest_edges(expression, metric="cosine")
    return expression, spatial_edges, expression_edges


def encode_views(
    expression: list[list[float]],
    spatial_edges: list[tuple[int, int]],
    communication_edges: list[tuple[int, int]],
) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[list[float]]]:
    spatial_embedding = graph_encoder(expression, spatial_edges)
    communication_embedding = graph_encoder(expression, communication_edges)
    fused_embedding, attention_weights = attention_fuse(
        spatial_embedding, communication_embedding
    )
    return spatial_embedding, communication_embedding, fused_embedding, attention_weights


def summarize_result(
    fused_embedding: list[list[float]],
    expression: list[list[float]],
    communication_edges: list[tuple[int, int]],
    spots: list[str],
    genes: list[str],
) -> tuple[list[list[float]], list[float], list[tuple[str, float]]]:
    cluster_probabilities = student_t_assignments(fused_embedding, [0, 3])
    invasion_score = invasion_association_score(
        expression,
        communication_edges,
        genes.index("EPCAM"),
        genes.index("COL1A1"),
        genes.index("CXCR4"),
        genes.index("VIM"),
    )
    ranked_spots = sorted(zip(spots, invasion_score), key=lambda item: item[1], reverse=True)
    return cluster_probabilities, invasion_score, ranked_spots


def log_normalize(counts: list[list[int]], target_sum: float = 1000.0) -> list[list[float]]:
    normalized = []
    for row in counts:
        total = sum(row)
        normalized.append([math.log1p(value * target_sum / total) for value in row])
    return normalized


def nearest_edges(values: list[tuple[float, ...]] | list[list[float]], metric: str = "euclidean") -> list[tuple[int, int]]:
    edges = set()
    for source, row in enumerate(values):
        candidates = []
        for target, other in enumerate(values):
            if source == target:
                continue
            if metric == "cosine":
                similarity = cosine_similarity(row, other)
                distance = 1.0 - similarity
            else:
                distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(row, other)))
            candidates.append((distance, target))
        _, nearest = min(candidates)
        edges.add(tuple(sorted((source, nearest))))
    return sorted(edges)


def cosine_similarity(first: list[float], second: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(first, second))
    denominator = math.sqrt(sum(a * a for a in first) * sum(b * b for b in second))
    return numerator / denominator if denominator else 0.0


def communication_graph(
    expression: list[list[float]],
    spatial_edges: list[tuple[int, int]],
    ligand_index: int,
    receptor_index: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int, float]]]:
    ligand = [math.expm1(row[ligand_index]) for row in expression]
    receptor = [math.expm1(row[receptor_index]) for row in expression]
    observed = []
    for source, target in spatial_edges:
        score = ligand[source] * receptor[target] + ligand[target] * receptor[source]
        observed.append((source, target, score))

    null_scores = [[] for _ in observed]
    for shift in range(1, len(receptor)):
        shuffled = receptor[shift:] + receptor[:shift]
        for index, (source, target, _) in enumerate(observed):
            null_scores[index].append(
                ligand[source] * shuffled[target] + ligand[target] * shuffled[source]
            )

    z_scored = []
    for (source, target, score), null in zip(observed, null_scores):
        mean = sum(null) / len(null)
        variance = sum((value - mean) ** 2 for value in null) / len(null)
        z_score = (score - mean) / max(math.sqrt(variance), 1e-8)
        z_scored.append((source, target, z_score))

    positive = [item for item in z_scored if item[2] > 0]
    if not positive:
        positive = [max(z_scored, key=lambda item: item[2])]
    retained = sorted(positive, key=lambda item: item[2], reverse=True)[:2]
    edges = sorted((source, target) for source, target, _ in retained)
    return edges, z_scored


def graph_encoder(
    expression: list[list[float]], edges: list[tuple[int, int]]
) -> list[list[float]]:
    """Apply a compact message-passing layer with fixed teaching weights."""
    projection = [
        [0.8, 0.0, 0.1],
        [0.0, 0.8, 0.1],
        [0.1, 0.2, 0.6],
        [0.1, 0.1, 0.7],
        [0.1, 0.4, 0.3],
    ]
    transformed = [
        [sum(row[g] * projection[g][axis] for g in range(len(projection))) for axis in range(3)]
        for row in expression
    ]
    neighbors = [[] for _ in expression]
    for source, target in edges:
        neighbors[source].append(target)
        neighbors[target].append(source)
    encoded = []
    for node, row in enumerate(transformed):
        messages = [row] + [transformed[neighbor] for neighbor in neighbors[node]]
        encoded.append([
            max(0.0, sum(message[axis] for message in messages) / len(messages) + row[axis])
            for axis in range(3)
        ])
    return encoded


def attention_fuse(
    spatial: list[list[float]], communication: list[list[float]]
) -> tuple[list[list[float]], list[list[float]]]:
    fused, weights = [], []
    for spatial_row, communication_row in zip(spatial, communication):
        logits = [sum(spatial_row) / len(spatial_row), sum(communication_row) / len(communication_row)]
        exponentials = [math.exp(value - max(logits)) for value in logits]
        total = sum(exponentials)
        row_weights = [value / total for value in exponentials]
        weights.append(row_weights)
        fused.append([
            row_weights[0] * spatial_value + row_weights[1] * communication_value
            for spatial_value, communication_value in zip(spatial_row, communication_row)
        ])
    return fused, weights


def student_t_assignments(
    embeddings: list[list[float]], center_indices: list[int], alpha: float = 1.0
) -> list[list[float]]:
    centers = [embeddings[index] for index in center_indices]
    assignments = []
    for row in embeddings:
        similarities = [
            (1.0 + sum((value - center_value) ** 2 for value, center_value in zip(row, center)) / alpha)
            ** (-(alpha + 1.0) / 2.0)
            for center in centers
        ]
        total = sum(similarities)
        assignments.append([value / total for value in similarities])
    return assignments


def invasion_association_score(
    expression: list[list[float]],
    communication_edges: list[tuple[int, int]],
    epithelial_index: int,
    matrix_gene_index: int,
    receptor_index: int,
    emt_gene_index: int,
) -> list[float]:
    epithelial = minmax([row[epithelial_index] for row in expression])
    matrix = minmax([row[matrix_gene_index] for row in expression])
    emt = minmax([row[emt_gene_index] for row in expression])
    receptor = [math.expm1(row[receptor_index]) for row in expression]
    communication = [0.0] * len(expression)
    for source, target in communication_edges:
        strength = receptor[source] + receptor[target]
        communication[source] += strength
        communication[target] += strength
    communication = minmax(communication)
    boundary_mixture = [epithelial[i] * matrix[i] for i in range(len(expression))]
    return [0.6 * boundary_mixture[i] + 0.2 * emt[i] + 0.2 * communication[i]
            for i in range(len(expression))]


def minmax(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    if high == low:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def rounded_matrix(matrix: list[list[float]], digits: int = 3) -> list[list[float]]:
    return [[round(value, digits) for value in row] for row in matrix]


def edge_names(edges: list[tuple[int, int]], spots: list[str]) -> list[str]:
    return [f"{spots[source]}–{spots[target]}" for source, target in edges]


def edge_score_names(
    scores: list[tuple[int, int, float]], spots: list[str]
) -> list[tuple[str, float]]:
    return [(f"{spots[source]}–{spots[target]}", round(score, 3))
            for source, target, score in scores]


if text is None or note is None:
    def text(message: str, **_: object) -> None:
        print(message)

    def note(message: str) -> None:
        print(f"[Presenter note] {message}")


if __name__ == "__main__":
    main()

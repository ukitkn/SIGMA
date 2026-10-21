# SIGMA: ten-minute presenter notes

Use these notes for rehearsal; they are not rendered into the LecTrace webpage.
Pause on the numerical outputs and figures while explaining. The five numbered
states are conceptual checkpoints, not a requirement to make exactly five clicks.

## 0:00–2:30 — Healthcare problem, motivation and approach

Colorectal cancer develops within a tissue ecosystem. A tumor contains malignant
epithelium, surrounding stromal cells, extracellular matrix and other cell types.
The tumor–stroma interface is especially interesting because local signaling and
matrix remodeling may accompany invasion. Our healthcare question is whether a
spatial expression map can help identify and interpret these tissue relationships.

Spatial transcriptomics measures gene expression together with tissue coordinates.
A spot can contain several cells, so our segmentation task assigns spots to tissue
compartments; it does not outline individual cell membranes. Expression tells us
which programs are active. Coordinates tell us which spots are close. Neither one
alone tells us whether nearby spots also express a compatible ligand and receptor.

SIGMA extends an MCGAE-inspired graph autoencoder with physical and communication
views. A base representation feeds two graph encoders. One follows physical
neighbors; the other follows a ligand–receptor-derived graph. Their embeddings are
combined and trained with reconstruction objectives. The outputs then support
clustering, marker-based interpretation and an invasion-associated spatial field.

The central technical question today is how changing the graph changes the
messages a neural network receives, and how reconstruction objectives train the
resulting embedding. We will inspect five states rather than every training epoch.

The executable example has sixty synthetic spots and uses the repository's actual
model functions. Its small dimensions let us inspect the tensors. At the end I
will discuss the archived report results separately; this teaching run does not
reproduce the original benchmark.

**Action:** Stay on the overview until the problem, research question and two-view
approach are clear. Then move to State 1.

## 2:30–3:40 — State 1: physical and communication graphs

Each row of the input is a spot and each column is a gene. The example normalizes
library totals and applies log one plus expression. The first graph connects nearby
coordinates. For the communication graph, the current code starts from spatial
candidates and combines ligand expression at one endpoint with receptor expression
at the other. This small example uses CXCL12 and CXCR4.

Look at the same spot locations in both panels. The locations have not changed,
but the edge set and weights have. Thirty receptor shuffles provide a background
for the edge scores; the resulting z-scores are not CellPhoneDB p-values. Edges are
symmetrized in this implementation. The learning consequence is that a spot will
aggregate a different neighborhood in each view.

**Action:** Inspect `graph_info`, then compare the physical and CCC graphs. Point
out one connection that is retained differently. Avoid reading every edge aloud.

## 3:40–5:05 — State 2: dual encoding and hybrid fusion

The MCGAE-inspired base combines original and augmented representations. Its
embedding then enters the physical and CCC encoders. Each encoder uses three
graph-convolution layers, residual paths and Layer Normalization. The graph controls
which neighbors contribute messages. The residual path preserves information from
the current spot while the propagated features add neighborhood context.

The heatmaps show three representative spots at successive representation stages.
Each view has eight latent components in this reduced example. The same spot
can acquire different physical and communication representations because the
graphs and branch parameters differ.

The important fusion line is the concatenation followed by a learned linear map
and ReLU. This is the repository's hybrid setting, rather than its optional
attention-fusion variant. Concatenation makes both feature sets available to a
joint projection. We inspect one forward pass after a short training run; the
eighty-epoch loop is deliberately omitted from this presentation trace.

**Action:** Point to `z_base`, `z_sp`, `z_ccc` and `z`. Follow one heatmap row across
the panels. Explain the operation, rather than assigning a biological name to
individual latent dimensions.

## 5:05–6:30 — State 3: reconstruction and a real parameter update

An embedding becomes useful through its training objectives. The expression decoder
maps it back to gene expression. The communication decoder uses a bilinear score
between two spot embeddings and converts that score into an edge probability.
Its target is inferred edge presence, not a directly measured signaling event.

We now inspect a real gradient update. For clarity, this selected update uses only
the expression mean-squared error and weighted communication binary cross-entropy.
The complete trainer also includes spatial reconstruction, contrastive, VAE,
clustering and smoothness terms. The displayed subset should not be mistaken for
the complete training objective.

The backward call computes gradients and the Adam step changes parameters. Compare
the same selected loss before and after the step, and inspect the nonzero embedding
change. A lower reconstruction loss shows the update is doing computation; it does
not, by itself, establish better tumor segmentation or clinical performance.

**Action:** Pause on `loss`, `loss.backward()`, the optimizer step and `update`.
Explain why a numerical parameter update differs from simply drawing a graph.

## 6:30–7:35 — State 4: clustering and tissue interpretation

We initialize three centers on the current embedding using the repository's KMeans
initializer, without tissue labels. The Student-t assignment then compares each embedding with those centers and
normalizes the resulting similarities. Each row sums to one. Taking the largest
entry gives a cluster index, but the index has no inherent tumor or stroma meaning.
It is also not a calibrated cell-type probability.

The current analysis code separately scores expression marker programs to annotate
Tumor, Stroma and Other. Compare the cluster panel with the marker panel. They are
different outputs computed by different rules. Marker agreement can help interpret
the representation, but labels built from those same expression programs are not
independent biological validation. Evaluation against available tissue references
is a further step, and the archived report assesses that agreement.

**Action:** Inspect one row of `q`, then the `cluster_id` vector. Emphasize the
difference between an unsupervised partition and a marker-based compartment label.

## 7:35–8:45 — State 5: pseudotime and spatial interpretation

The downstream code chooses an early root using EMT, ECM and distance to
fibroblast-labeled spots. Diffusion pseudotime then orders spots relative to that
root on the learned graph. In this implementation the DPT graph covers the full
teaching slice; tumor-only subclustering is a separate analysis in the full pipeline.

The left panel shows root-relative ordering. Gaussian interpolation produces the
continuous score in the right panel. The arrows follow the gradient of that score.
Equivalently, if U equals one minus P, they follow minus the gradient of U.

The change is from discrete spot scores to a spatial landscape that can guide
inspection. Root choice, marker programs and smoothing influence that landscape.
There are no calibrated time, speed or force units. It is an invasion-associated
interpretation of a static section, rather than an observed movie of migrating cells.

**Action:** Point out the chosen root, the relative score scale and the gradient
arrows. Do not identify the largest synthetic value as a clinical finding.

## 8:45–10:00 — Key result and healthcare interpretation

Return to the actual report tables. Across eight sections, the archived mean ARI
is 0.4901 for SIGMA and 0.4204 for MCGAE. The section-wise differences matter:
SIGMA exceeds MCGAE on four sections and falls below it on four. The average gain
does not mean every tissue section improves.

The ST-colon1 ablation also shows a trade-off. With the upgraded backbone,
physical-only to hybrid changes ARI from 0.6712 to 0.6775, but Macro Dice changes
from 0.7383 to 0.7319. Therefore hybrid reconstruction is not uniformly best.
These archived metrics describe clustering or compartment agreement; they do not
measure prospective patient-level invasion prediction.

The healthcare interpretation is that communication-aware representations can
support investigation of tumor–stroma organization. The downstream field supplies
a spatial hypothesis for pathology review and biological follow-up. The execution
explains how the model produces those representations and visualizations, while
the report supplies the separate, qualified evidence about segmentation performance.

**Action:** Finish on the report comparison figure. State one gain, one trade-off
and the distinction between an invasion-associated map and a validated forecast.

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
compartments; it does not outline individual cell membranes. H&E reveals tissue architecture and morphological boundaries. Expression tells us
which molecular programs are active, while coordinates locate the spots.
Ligand–receptor information adds potential communication context. The report uses
eight colon/liver sections containing 31,296 spots, with 36,601 gene features per
section; R conversion aligns the count matrix, coordinates and metadata in a
Visium-compatible structure before filtering, normalization and feature selection.

SIGMA extends an MCGAE-inspired graph autoencoder with physical and communication
views. A base representation feeds two graph encoders. One follows physical
neighbors; the other follows a ligand–receptor-derived graph. Their embeddings are
combined and trained with reconstruction objectives. The outputs then support
clustering, marker-based interpretation and an invasion-associated spatial field.

The central technical question today is how changing the graph changes the
messages a neural network receives, and how reconstruction objectives train the
resulting embedding. We will inspect five states rather than every training epoch.

The executable example has sixty synthetic spots and uses the current reconstructed
model code. Its small dimensions let us inspect the tensors. This teaching run does
not reproduce the historical model or benchmark. At the end I will discuss the
report's archived table values separately.

**Action:** Stay on the overview until the problem, research question and two-view
approach are clear. Then move to State 1.

## 2:30–3:40 — State 1: physical and communication graphs

Each row of the input is a spot and each column is a gene. The example normalizes
library totals and applies log one plus expression. The first graph connects eight-nearest spatial neighbors and weights their edges
by both coordinate distance and patch-feature similarity. Here the patch descriptors
are synthetic; the report uses H&E features through the upstream MCGAE workflow. For the communication graph, the current code starts from spatial
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
and ReLU. This is the reconstruction's hybrid setting, rather than its optional
cross-view attention-fusion variant. The base still uses attention internally.
The report defines optional Jumping Knowledge by projecting concatenated layer
outputs, but disables it in the reference protocol. This teaching run also disables
it; the ablation does not isolate a JK contribution. We inspect one forward pass after a short training run;
the eighty-epoch loop is deliberately omitted from this presentation trace.

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

The teaching analysis separately scores the report's marker programs: IFITM1/SNHG6
for Tumor, IGFBP7/VIM/LGALS1 for F/S, and ZG16/ITLN1/CLCA1 for normal/other. Compare the cluster panel with the marker panel. They are
different outputs computed by different rules. Marker agreement can help interpret
the representation, but labels built from those same expression programs are not
independent biological validation. Evaluation against available tissue references
is a further step, and the archived report assesses that agreement.

**Action:** Inspect one row of `q`, then the `cluster_id` vector. Emphasize the
difference between an unsupervised partition and a marker-based compartment label.

## 7:35–8:45 — State 5: pseudotime and spatial interpretation

The report's downstream construction first identifies tumor states with Leiden at
resolution 0.4. It scores each state's median EMT, ECM and distance to F/S spots.
An early state has lower EMT/ECM and greater stromal distance. Within that state,
rank sums select a root spot. DPT then orders the full teaching graph relative to
that root. The reference EMT program contrasts VIM, FN1, MMP9, TWIST1 and ZEB1
with CDH1 and EPCAM; ECM uses COL1A1, COL1A2, FN1 and TNC. PAGA summarizes
state connectivity. Unreachable spots are excluded rather than assigned a fabricated rank.

The left panel shows normalized root-relative DPT. The normal/other and stromal
centroids define an anatomical axis. Spot scores are sorted along it and smoothed
with a centered 31-spot moving average, then interpolated to an 80 × 80 grid using
Gaussian weights and bandwidth 80 in the input coordinates. Up to 80 neighbors
are used (all 60 in this small example). The continuous progression intensity is P;
the complementary potential is U = 1 − P. Arrows follow F = −∇U = ∇P, calculated
using actual grid spacing. The right panel shows P, not U, so higher intensity means
a later inferred state rather than higher complementary potential.

The change is from discrete spot scores to a spatial landscape that can guide
inspection. Root choice, marker programs and smoothing influence that landscape.
There are no calibrated time, speed or force units. It is an invasion-associated
interpretation of a static section, rather than an observed movie of migrating cells.

**Action:** Point out the chosen root, the relative score scale and the gradient
arrows. Do not identify the largest synthetic value as a clinical finding.

## 8:45–10:00 — Key result and healthcare interpretation

Return to the report tables. Across eight sections, the archived mean ARI is
0.4901 for SIGMA and 0.4204 for MCGAE. SIGMA exceeds MCGAE on four sections,
falls below it on four, and ranks first among the four listed methods on three.
The average gain does not mean every tissue section improves.

The corrected ST-colon1 ablation shows a stronger result. With the upgraded
backbone, physical-only to SIGMA changes ARI from 0.6345 to 0.7328 and Macro Dice
from 0.7292 to 0.8618. Upgraded SIGMA leads all eight variants on all four metrics:
ARI 0.7328, NMI 0.6595, Macro Dice 0.8618 and IoU 0.7630. The original-backbone
SIGMA variant falls below its baseline, so the benefit depends on architecture.
This is a separate comparison from the eight-section benchmark; its ST-colon1
number should not replace the benchmark entry.
These archived metrics describe clustering or compartment agreement; they do not
measure prospective patient-level invasion prediction.

The healthcare interpretation is that communication-aware representations can
support investigation of tumor–stroma organization. The downstream field supplies
a spatial hypothesis for pathology review and biological follow-up. The execution
explains how the model produces those representations and visualizations, while
the report supplies the separate, qualified evidence about segmentation performance.

The report also discusses 1,665 tumor-core spots: the ordering correlates with EMT
at r = 0.780, ECM at r = 0.765 and nearest-CAF distance at r = −0.286. The shared-
embedding Slingshot–DPT comparison gives approximately rho = 0.85. These are
interpretation checks rather than independent validation of movement. Future work
should examine seed variability, module ablations and root/marker/smoothing
sensitivity, then compare candidate routes with independent biological evidence.

**Action:** Finish on the report benchmark and ablation figure. State one gain,
the architecture dependence, one limitation and a concrete validation direction.
The final Software 3.0 note reproduces the report's limited-assistance disclosure;
it is not an additional computational state.

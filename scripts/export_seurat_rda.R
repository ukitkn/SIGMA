args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript export_seurat_rda.R input.rda output_directory [assay_name]")
}

input_path <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- args[[2]]
requested_assay <- if (length(args) >= 3) args[[3]] else ""
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages(library(Seurat))
suppressPackageStartupMessages(library(Matrix))
environment <- new.env(parent = globalenv())
object_names <- load(input_path, envir = environment)
seurat_objects <- object_names[vapply(object_names, function(name) {
  inherits(environment[[name]], "Seurat")
}, logical(1))]
if (length(seurat_objects) == 0) {
  stop("The RDA file does not contain a Seurat object.")
}
object <- environment[[seurat_objects[[1]]]]
assay_names <- SeuratObject::Assays(object)
if (nzchar(requested_assay)) {
  if (!(requested_assay %in% assay_names)) {
    stop(paste("Requested assay is missing:", requested_assay))
  }
  assay_name <- requested_assay
} else if ("Spatial" %in% assay_names) {
  assay_name <- "Spatial"
} else if ("RNA" %in% assay_names) {
  assay_name <- "RNA"
} else {
  assay_name <- assay_names[[1]]
}

counts <- tryCatch(
  SeuratObject::LayerData(object, assay = assay_name, layer = "counts"),
  error = function(error) NULL
)
if (is.null(counts) || nrow(counts) == 0 || ncol(counts) == 0) {
  counts <- tryCatch(
    SeuratObject::GetAssayData(object, assay = assay_name, slot = "counts"),
    error = function(error) NULL
  )
}
if (is.null(counts) || nrow(counts) == 0 || ncol(counts) == 0) {
  stop(paste("A non-empty raw count layer is unavailable for assay", assay_name))
}
counts <- as(counts, "dgCMatrix")
barcodes <- colnames(counts)

image_names <- names(object@images)
if (length(image_names) == 0) {
  stop("The Seurat object has no Visium image with spot coordinates.")
}
coordinates <- Seurat::GetTissueCoordinates(object[[image_names[[1]]]])
if (!all(c("imagerow", "imagecol") %in% colnames(coordinates))) {
  stop("Visium coordinates must contain imagerow and imagecol columns.")
}
missing_coordinates <- setdiff(barcodes, rownames(coordinates))
if (length(missing_coordinates) > 0) {
  stop(paste(length(missing_coordinates), "count barcodes have no image coordinates."))
}
coordinates <- coordinates[barcodes, , drop = FALSE]
metadata <- object@meta.data[barcodes, , drop = FALSE]

Matrix::writeMM(counts, file.path(output_dir, "matrix.mtx"))
writeLines(barcodes, file.path(output_dir, "barcodes.tsv"), useBytes = TRUE)
features <- data.frame(
  feature_id = rownames(counts),
  feature_name = rownames(counts),
  feature_type = "Gene Expression",
  stringsAsFactors = FALSE
)
utils::write.table(features, file.path(output_dir, "features.tsv"), sep = "\t",
                    quote = FALSE, row.names = FALSE, col.names = FALSE)
metadata$barcode <- rownames(metadata)
utils::write.csv(metadata, file.path(output_dir, "metadata.csv"), row.names = FALSE)
spatial <- data.frame(
  barcode = barcodes,
  x = as.numeric(coordinates$imagecol),
  y = as.numeric(coordinates$imagerow),
  stringsAsFactors = FALSE
)
utils::write.csv(spatial, file.path(output_dir, "spatial.csv"), row.names = FALSE)
writeLines(c(
  paste0("object=", seurat_objects[[1]]),
  paste0("assay=", assay_name),
  paste0("image=", image_names[[1]]),
  paste0("spots=", length(barcodes)),
  paste0("genes=", nrow(counts))
), file.path(output_dir, "source_metadata.txt"))

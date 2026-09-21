#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(flowCore)
  library(PeacoQC)
  library(flowDensity)
  library(ggplot2)
  library(gridExtra)
})

folder_path <- "/storage/mezya.sezen/mphasis/LAIP29/fcs/"
output_path <- "/storage/mezya.sezen/mphasis/LAIP29/final_cleaned"

if (!dir.exists(output_path)) dir.create(output_path, recursive = TRUE)

message("=== Script started: Margin + Debris + Doublet Removal (no PeacoQC QC) ===")

# Read FCS files
fcs_files <- list.files(folder_path, pattern = "\\.fcs$", full.names = TRUE)
message("Total FCS files found: ", length(fcs_files))
ffs <- lapply(fcs_files, read.FCS, transformation = FALSE, truncate_max_range = FALSE)
sample_names <- basename(fcs_files)

# -------- Plot function (contour plots) --------
plot_contour <- function(ff, title = "") {
  df <- as.data.frame(exprs(ff)[, c("FSC-A", "SSC-A")])
  ggplot(df, aes(x = `FSC-A`, y = `SSC-A`)) +
    geom_density_2d(color = "blue") +
    labs(title = title, x = "FSC-A", y = "SSC-A") +
    theme_minimal()
}

pdf("/storage/mezya.sezen/mphasis/LAIP29/final_cleaned/cleanup_contours_final.pdf", width = 14, height = 6)

# -------- Main processing loop --------
for (i in seq_along(ffs)) {
  ff <- ffs[[i]]
  fname <- sample_names[i]
  message("\nProcessing: ", fname)

  # --- Step 0: Before cleaning
  p_before <- plot_contour(ff, title = paste(fname, "Before preprocessing"))

  # --- Step 1: Margin removal
  message("  Margin removal")
  original_cols <- colnames(ff)
  ff_margin <- PeacoQC::RemoveMargins(ff, c("FSC-A", "FSC-H", "SSC-A", "SSC-H"))
  ff_margin <- ff_margin[, original_cols]
  p_margin <- plot_contour(ff_margin, title = paste(fname, "After margin removal"))

  # --- Step 2: Debris removal
  message("  Debris removal")
  ff_debris <- nmRemove(ff_margin, c("FSC-A", "SSC-A"), verbose = TRUE)
  scatter_markers <- c("FSC-A", "SSC-A")
  scatter_exprs <- exprs(ff_debris)[, scatter_markers]
  scatter_cutoffs <- apply(scatter_exprs, 2, function(x) quantile(x, probs = 0.01))
  scatter_gate <- flowDensity(ff_debris, channels = scatter_markers,
                              position = c(TRUE, TRUE), gates = scatter_cutoffs)
  ff_debris_clean <- ff_debris[scatter_gate@index, ]
  p_debris <- plot_contour(ff_debris_clean, title = paste(fname, "After debris removal"))

  # --- Step 3: Doublet removal (no PeacoQC quality check)
  message("  Doublet removal")
  ff_final <- RemoveDoublets(ff_debris_clean)
  p_doublet <- plot_contour(ff_final, title = paste(fname, "After doublet removal"))

  # --- Combine plots for this sample
  grid.arrange(p_before, p_margin, p_debris, p_doublet, ncol = 2, top = fname)

  # --- Save cleaned FCS
  out_fcs <- file.path(output_path,
                       paste0(tools::file_path_sans_ext(fname), "_Cleaned.fcs"))
  write.FCS(ff_final, out_fcs)
  message("Saved cleaned FCS: ", out_fcs)
}

dev.off()
message("\n=== Script finished: cleanup_contours_final.pdf generated ===")

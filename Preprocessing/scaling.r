#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(flowCore))

# Min-Max Scaling Function
minmax_scale_ff <- function(ff, export_path = NULL, q_low = 0.01, q_high = 0.99) {
  cols <- c('FSC-A', 'FSC-H', 'SSC-A', 'SSC-H')
  if (!is.null(ff@description$SPILL)) {
    cols <- c(cols, colnames(ff@description$SPILL))
  }
  
  # Scale in place
  exprs(ff)[, cols] <- apply(exprs(ff)[, cols, drop=FALSE], 2, function(x) {
    (x - quantile(x, q_low)) / (quantile(x, q_high) - quantile(x, q_low))
  })
  
  if (!is.null(export_path)) {
    write.FCS(ff, export_path)
    message("Saved scaled FCS to: ", export_path)
  }
  
  return(ff)
}

# MAIN
args <- commandArgs(trailingOnly = TRUE)
if(length(args) < 2) stop("Usage: Rscript scaling.r <input_dir> <output_dir>")

input_dir <- args[1]
output_dir <- args[2]
if(!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

input_files <- list.files(input_dir, pattern="\\.fcs$", full.names=TRUE)
if(length(input_files) == 0) stop("No FCS files found in input directory!")

# Open PDF for histograms
pdf(file.path(output_dir, "scaling_plots.pdf"), width=12, height=8)
channels_to_plot <- c("FSC-A", "FSC-H", "SSC-A", "FITC-A")

# Loop through each file
for (input_fcs in input_files) {
  base <- tools::file_path_sans_ext(basename(input_fcs))
  output_fcs <- file.path(output_dir, paste0(base, "_scaled.fcs"))
  
  ff <- read.FCS(input_fcs, transformation = FALSE)
  ff_scaled <- minmax_scale_ff(ff, export_path = output_fcs)
  
  par(mfrow=c(length(channels_to_plot), 2))  # One row per channel, 2 columns (raw & scaled)
  
  for (ch in channels_to_plot) {
    hist(exprs(ff)[, ch],
         main = paste(base, "Before", ch),
         xlab = "Raw Intensity",
         col = "skyblue", breaks = 100)
    
    hist(exprs(ff_scaled)[, ch],
         main = paste(base, "Scaled", ch),
         xlab = "Scaled (0-1)",
         col = "orange", breaks = 100)
  }
  
  # Add a title between files (optional)
  #mtext(base, side = 3, line = 0, outer = TRUE)
}

dev.off()
message("All histograms saved to scaling_plots.pdf in ", output_dir)

library(flowCore)
library(flowDensity)
library(ggplot2)
library(gridExtra)

transform_path <- "/storage/mezya.sezen/mphasis/LAIP29/"
cleaned_path   <- "/storage/mezya.sezen/mphasis/LAIP29/cleaned/"
output_path    <- "/storage/mezya.sezen/mphasis/LAIP29/transformed"



plot_contour <- function(ff, marker_map, x_marker, y_marker, title = "") {
  ch_x <- marker_map[x_marker]
  ch_y <- marker_map[y_marker]

  if (is.na(ch_x) || is.na(ch_y)) {
    return(ggplot() + labs(title = paste("Missing:", x_marker, "or", y_marker)))
  }

  df <- as.data.frame(exprs(ff)[, c(ch_x, ch_y)])
  ggplot(df, aes(x = .data[[ch_x]], y = .data[[ch_y]])) +
    geom_density_2d(color = "blue") +
    labs(title = title, x = ch_x, y = ch_y) +
    theme_minimal()
}

# --- Histogram function for all channels ---
plot_histograms <- function(ff_before, ff_after, fl_channels, title = "") {
  library(tidyr)
  library(dplyr)

  df_before <- as.data.frame(exprs(ff_before)[, fl_channels, drop = FALSE])
  df_after  <- as.data.frame(exprs(ff_after)[, fl_channels, drop = FALSE])

  # Convert to long format for ggplot
  df_before_long <- df_before %>%
    pivot_longer(cols = everything(),
                 names_to = "Channel",
                 values_to = "Intensity") %>%
    mutate(Time = "Before")

  df_after_long <- df_after %>%
    pivot_longer(cols = everything(),
                 names_to = "Channel",
                 values_to = "Intensity") %>%
    mutate(Time = "After")

  df_combined <- bind_rows(df_before_long, df_after_long)

  ggplot(df_combined, aes(x = Intensity, fill = Time)) +
    geom_histogram(position = "identity", alpha = 0.5, bins = 100) +
    facet_wrap(~Channel, scales = "free") +
    scale_fill_manual(values = c("Before" = "red", "After" = "blue")) +
    labs(title = title, x = "Fluorescence Intensity", y = "Count") +
    theme_minimal()
}


message("=== Script started ===")

# Read FCS files
fcs_files <- list.files(cleaned_path, pattern = "_Cleaned.fcs$", full.names = TRUE)
message("Total FCS files found: ", length(fcs_files))
ffs <- lapply(fcs_files, read.FCS, transformation = FALSE, truncate_max_range = FALSE)
sample_names <- basename(fcs_files)

message("Collecting parameter metadata for all files …")
param_metadata_list <- lapply(fcs_files, function(f) {
  ff <- read.FCS(f, transformation = FALSE, truncate_max_range = FALSE)
  m  <- pData(parameters(ff))
  m$file <- basename(f)        # add file name for reference
  m
})
all_param_metadata <- do.call(rbind, param_metadata_list)

# Get mapping from desc -> detector



# Store the base names (file names only) for later use
sample_names <- basename(fcs_files)

plot_marker <- function(ff, marker_map, marker_x, marker_y, title = "") {
  ch_x <- marker_map[marker_x]
  ch_y <- marker_map[marker_y]

  # Check if channel exists
  if (is.na(ch_x) || is.na(ch_y)) {
    message("Skipping plot '", title, "' because missing channel(s): ",
            marker_x, "=", ch_x, ", ", marker_y, "=", ch_y)
    return(ggplot() + labs(title = paste("Missing:", marker_x, "or", marker_y)))
  }

  df <- as.data.frame(exprs(ff)[, c(ch_x, ch_y)])
  ggplot(df, aes(x = .data[[ch_x]], y = .data[[ch_y]])) +
    geom_point(alpha = 0.3, size = 0.5) +
    labs(title = title, x = ch_x, y = ch_y) +
    theme_minimal()
}



# ---------- Extract tube names (p1, p2 …) from file names ----------
get_tube <- function(name) {
  # Updated regex for LAIP_*_*_P* pattern
   matches <- regmatches(name, regexec("_(P\\d+)", name))
  if (length(matches[[1]]) > 1) {
    return(matches[[1]][2])
  } else {
    warning("Could not extract tube ID from filename: ", name)
    return(NA)
  }
}
tubes <- sapply(sample_names, get_tube)

# ---------- Group file indices by tube ----------
tube_index_map <- split(seq_along(tubes), tubes)

# ---------- Step 1: Calculate and save Logicle transform per tube ---------- 
message("Calculating/saving logicle transforms per tube …")
for (tube in names(tube_index_map)) {
  message("Calculating logicle transform for tube: ", tube)

  transform_file <- file.path(transform_path,
                              paste0(tube, "_logicleTransform.rds"))
  if (file.exists(transform_file)) next  # skip if already computed

  # flowFrames belonging to this tube
  tube_ffs <- ffs[tube_index_map[[tube]]]

  
  # ---- UPDATED: Determine fluorescence channels using metadata from all files in this tube
  tube_files <- sample_names[tube_index_map[[tube]]]
  tube_meta  <- subset(all_param_metadata, file %in% tube_files)
  # Get only valid fluorescence channels (desc not empty, name not NA)
  fl_channels <- na.omit(unique(tube_meta$name[tube_meta$desc != ""]))


  
  cat("Tube:", tube, "\n")
  cat("All channels in tube (fl_channels):", paste(fl_channels, collapse = ", "), "\n\n")
  
  for (i in seq_along(tube_ffs)) {
    ff <- tube_ffs[[i]]
    file_name <- tube_files[i]
    present_ch <- colnames(ff)
    cat("File:", file_name, "\n")
    cat("Channels present in file:", paste(present_ch, collapse = ", "), "\n\n")
  }

  # ----- Estimate logicle parameters for each sample -----
   if (exists("estimateMedianLogicle", mode = "function")) {
    # Subset each flowFrame to fluorescence channels only
    tube_ffs_sub <- lapply(tube_ffs, function(ff) {
    present_ch <- intersect(colnames(ff), fl_channels)
    ff[, present_ch, drop = FALSE]
  })
    tube_fs <- flowSet(tube_ffs_sub)

    # Returns a transformList built from median of per-sample logicle parameters
    tfList <- estimateMedianLogicle(tube_fs, channels = fl_channels)
  } else {
    # Fallback: per-sample estimateLogicle + median manually
    logicle_params_list <- lapply(tube_ffs, function(ff) estimateLogicle(ff, channels = fl_channels))

    # Median aggregation per channel
    median_params <- list()
    param_names <- c("w", "t", "m", "a")
    for (ch in fl_channels) {
      param_matrix <- do.call(rbind, lapply(logicle_params_list, function(tf) {
        unlist(lapply(tf@transforms[ch], function(tr) c(tr@w, tr@t, tr@m, tr@a)))
      }))
      median_param <- apply(param_matrix, 2, median)
      names(median_param) <- param_names
      median_params[[ch]] <- median_param
    }

    # Build transformList from median parameters
    logicle_transforms <- lapply(median_params, function(p) logicleTransform(
      w = p["w"], t = p["t"], m = p["m"], a = p["a"]
    ))
    tfList <- transformList(fl_channels, logicle_transforms)
  }

  # Save transformList for later use
  saveRDS(tfList, transform_file)
  message("Calculating/saving logicle transforms per tube …")
}


message("=== Starting preprocessing plots ===")
pdf("/storage/mezya.sezen/mphasis/transformation_contours.pdf", width = 12, height = 6)
pdf_hist <- "/storage/mezya.sezen/mphasis/transformation_histograms.pdf"
pdf(pdf_hist, width = 12, height = 8)

for (i in seq_along(ffs)) {
  ff <- ffs[[i]]
  fname <- sample_names[i]
  tube <- get_tube(fname)
  
  message("Processing file: ", fname, " (tube: ", tube, ")")

  # Build marker map (desc → name)
  param_meta <- pData(parameters(ff))
  marker_map <- setNames(param_meta$name, param_meta$desc)

  # --- Compensation ---
  if (!is.null(ff@description$SPILL)) {
    ff_comp <- compensate(ff, ff@description$SPILL)
  } else {
    warning("No SPILL matrix found in ", fname, " — skipping compensation.")
    ff_comp <- ff
  }

  # --- Logicle transformation ---
  tf_path <- file.path(transform_path, paste0(tube, "_logicleTransform.rds"))
  if (file.exists(tf_path)) {
    tfList <- readRDS(tf_path)
    ff_trans <- transform(ff_comp, tfList)
  } else {
    warning("Transform file not found for tube ", tube, " — skipping transformation.")
    ff_trans <- ff_comp
  }

  # --- Plot Before vs After side by side ---
  p_before <- plot_contour(ff_comp, marker_map, "CD34 Cy55", "CD117", "Before Transformation")
  p_after  <- plot_contour(ff_trans, marker_map, "CD34 Cy55", "CD117", "After Transformation")
  grid.arrange(p_before, p_after, ncol = 2, top = fname)
  
  # --- Histograms for all fluorescence channels ---
  fl_channels <- na.omit(unique(param_meta$name[param_meta$desc != ""]))
  if (length(fl_channels) > 0) {
    p_hist <- plot_histograms(ff_comp, ff_trans, fl_channels, title = fname)
    print(p_hist)
  }

  # --- Save transformed FCS ---
  out_fcs <- file.path(output_path,
                       paste0(tools::file_path_sans_ext(sample_names[i]), "_transformed.fcs"))
  write.FCS(ff_trans, out_fcs)
  message("Saved transformed FCS: ", out_fcs)
}

dev.off()
message("=== Script finished: transformation_plots.pdf generated ===")

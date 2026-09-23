library(flowCore)
library(ggplot2)
library(gridExtra)

# ==============================
# Input files
# ==============================
raw_file <- "/storage/pavinap.priyaa/Mphasis/dataset/after_doublet_removal/BLAST110_15_P1_cleaned_doubletRemoval.fcs"                     # before preprocessing
cleaned_file <- "/storage/pavinap.priyaa/Mphasis/dataset/transformed/BLAST110_15_P1_cleaned_doubletRemoval_transformed.fcs"         # after preprocessing
output_pdf <- "/storage/pavinap.priyaa/Mphasis/Contour_Transformation.pdf"

# ==============================
# Choose markers (use description, not name)
# ==============================
x_marker <- "CD34 Cy55"
y_marker <- "CD117"

# ==============================
# Helper function: get marker map
# ==============================
get_marker_map <- function(ff) {
  param_meta <- pData(parameters(ff))
  setNames(param_meta$name, param_meta$desc)
}

# ==============================
# Helper: generate contour plot with consistent scales
# ==============================
plot_contour_matched <- function(ff, marker_map, x_marker, y_marker, title, xlim_all = NULL, ylim_all = NULL) {
  ch_x <- marker_map[x_marker]
  ch_y <- marker_map[y_marker]

  if (is.na(ch_x) || is.na(ch_y)) {
    message("Skipping plot because missing channel(s): ", x_marker, " or ", y_marker)
    return(ggplot() + labs(title = paste("Missing:", x_marker, "or", y_marker)))
  }

  df <- as.data.frame(exprs(ff)[, c(ch_x, ch_y)])
  names(df) <- c("X", "Y")

  p <- ggplot(df, aes(x = X, y = Y)) +
    geom_density_2d(color = "black", linewidth = 0.5) +
    labs(title = title, x = paste0(x_marker, " (", ch_x, ")"),
         y = paste0(y_marker, " (", ch_y, ")")) +
    theme_minimal(base_size = 12)

  # If combined limits are provided, fix axes for both plots
  if (!is.null(xlim_all) && !is.null(ylim_all)) {
    p <- p + coord_cartesian(xlim = xlim_all, ylim = ylim_all)
  }

  return(p)
}

# ==============================
# Read both FCS files
# ==============================
ff_raw <- read.FCS(raw_file, transformation = FALSE)
ff_cleaned <- read.FCS(cleaned_file, transformation = FALSE)

marker_map_raw <- get_marker_map(ff_raw)
marker_map_cleaned <- get_marker_map(ff_cleaned)

# Ensure both files share same x/y channel for comparison
ch_x_raw <- marker_map_raw[x_marker]
ch_y_raw <- marker_map_raw[y_marker]
ch_x_cleaned <- marker_map_cleaned[x_marker]
ch_y_cleaned <- marker_map_cleaned[y_marker]

if (any(is.na(c(ch_x_raw, ch_y_raw, ch_x_cleaned, ch_y_cleaned)))) {
  stop("Missing one or more markers in either FCS file.")
}

# Compute global axis limits for fair comparison
x_all <- c(exprs(ff_raw)[, ch_x_raw], exprs(ff_cleaned)[, ch_x_cleaned])
y_all <- c(exprs(ff_raw)[, ch_y_raw], exprs(ff_cleaned)[, ch_y_cleaned])
xlim_all <- range(x_all, na.rm = TRUE)
ylim_all <- range(y_all, na.rm = TRUE)

# ==============================
# Generate and save plots
# ==============================
p_before <- plot_contour_matched(ff_raw, marker_map_raw, x_marker, y_marker, "Before Transformation", xlim_all, ylim_all)
p_after  <- plot_contour_matched(ff_cleaned, marker_map_cleaned, x_marker, y_marker, "After Transformation", xlim_all, ylim_all)

pdf(output_pdf, width = 10, height = 5)
grid.arrange(p_before, p_after, ncol = 2, top = paste0("Comparison: ", x_marker, " vs ", y_marker))
dev.off()

message("✅ PDF saved successfully: ", output_pdf)

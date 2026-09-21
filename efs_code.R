##############################################
# SETTINGS: EDIT THESE PATHS
##############################################

fcs_dir   <- "/storage/mezya.sezen/mphasis/dataset/BLAST110"   # BLAST110_*.fcs
label_dir <- "/storage/mezya.sezen/mphasis/dataset/labels"     # BLAST110_*.csv (labels)
out_dir   <- "/storage/mezya.sezen/mphasis/efs"                       # where to write EFS table

##############################################
# 1. Load packages
##############################################

if (!require("flowCore")) install.packages("flowCore", repos = "https://cloud.r-project.org")

library(flowCore)

##############################################
# 2. List FCS and label files
##############################################

fcs_files <- list.files(fcs_dir, pattern = "^BLAST110_.*\\.fcs$", full.names = TRUE)
cat("Found", length(fcs_files), "FCS files\n")
if (length(fcs_files) == 0) stop("No BLAST110_*.fcs files found in fcs_dir")

##############################################
# 3. Determine common FCS markers (intersection)
##############################################

get_fcs_cols <- function(path) {
  ff <- read.FCS(path, transformation = FALSE, which.lines = 1)
  colnames(exprs(ff))
}

all_fcs_cols_list <- lapply(fcs_files, get_fcs_cols)
common_markers <- Reduce(intersect, all_fcs_cols_list)

cat("Common markers present in ALL FCS files:\n")
print(common_markers)

# Optionally, drop non-informative channels like Time/event_ID from common set
common_markers <- setdiff(common_markers, c("Time", "event_ID"))
cat("Using these common markers for EFS:\n")
print(common_markers)

##############################################
# 4. Process each sample independently
##############################################

summaries <- list()

for (k in seq_along(fcs_files)) {
  fcs_path <- fcs_files[k]
  fcs_name <- basename(fcs_path)
  base     <- sub("\\.fcs$", "", fcs_name)
  label_path <- file.path(label_dir, paste0(base, ".csv"))

  cat("(", k, "/", length(fcs_files), ") Processing:", fcs_name, "\n")

  if (!file.exists(label_path)) {
    stop(paste("Label file not found for", fcs_name, "expected:", label_path))
  }

  # Read FCS
  ff   <- read.FCS(fcs_path, transformation = FALSE)
  expr <- as.data.frame(exprs(ff))

  # Read label CSV
  lab  <- read.csv(label_path, header = TRUE, stringsAsFactors = FALSE)

  if (!("event_ID" %in% colnames(expr))) stop(paste("FCS", fcs_name, "no event_ID"))
  if (!("event_ID" %in% colnames(lab)))  stop(paste("Label", basename(label_path), "no event_ID"))
  if (!("Blast" %in% colnames(lab)))     stop(paste("Label", basename(label_path), "no Blast"))

  merged <- merge(expr, lab, by = "event_ID", all.x = TRUE)

  # restrict to COMMON markers only
  marker_cols <- intersect(common_markers, colnames(merged))

  s <- data.frame(source_file = fcs_name, stringsAsFactors = FALSE)

  for (m in marker_cols) {
    x <- merged[[m]]
    s[[paste0(m, "_med")]] <- median(x, na.rm = TRUE)
    s[[paste0(m, "_q05")]] <- as.numeric(quantile(x, 0.05, na.rm = TRUE))
    s[[paste0(m, "_q95")]] <- as.numeric(quantile(x, 0.95, na.rm = TRUE))
  }

  # sample-level label: any Blast event in this file?
  s$class <- as.integer(any(merged$Blast == 1, na.rm = TRUE))

  summaries[[k]] <- s
}

##############################################
# 5. Align columns and stack
##############################################

all_cols <- Reduce(union, lapply(summaries, colnames))

summaries_aligned <- lapply(summaries, function(df) {
  missing <- setdiff(all_cols, colnames(df))
  if (length(missing) > 0) df[missing] <- NA
  df <- df[all_cols]
  df
})

sample_table <- do.call(rbind, summaries_aligned)

##############################################
# 6. Clean names and save for EFS
##############################################

colnames(sample_table) <- make.names(colnames(sample_table))

if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_file <- file.path(out_dir, "BLAST110_efs_ready.csv")
write.csv(sample_table, out_file, row.names = FALSE)

cat("Wrote:", out_file, "\n")
##############################################
# END
##############################################

##############################################
# SETTINGS
##############################################

fcs_dir   <- "/storage/mezya.sezen/mphasis/dataset/BLAST110"
label_dir <- "/storage/mezya.sezen/mphasis/dataset/labels"

##############################################
# 1. Check FCS headers
##############################################

if (!require("flowCore")) install.packages("flowCore", repos = "https://cloud.r-project.org")
library(flowCore)

fcs_files <- list.files(fcs_dir, pattern = "^BLAST110_.*\\.fcs$", full.names = TRUE)

get_fcs_cols <- function(path) {
  ff <- read.FCS(path, transformation = FALSE, which.lines = 1)  # 1 event only
  data.frame(
    file = basename(path),
    ncol = ncol(exprs(ff)),
    cols = paste(colnames(exprs(ff)), collapse = ";"),
    stringsAsFactors = FALSE
  )
}

fcs_info <- do.call(rbind, lapply(fcs_files, get_fcs_cols))

cat("FCS column count distribution:\n")
print(table(fcs_info$ncol))

cat("\nExample FCS header from first file:\n")
cat(fcs_info$cols[1], "\n")

# If multiple ncol values, show an example of each
if (length(unique(fcs_info$ncol)) > 1) {
  cat("\nDifferent FCS structures detected; examples:\n")
  print(fcs_info[!duplicated(fcs_info$ncol), c("file", "ncol", "cols")])
}

##############################################
# 2. Check label CSV headers
##############################################

label_files <- list.files(label_dir, pattern = "^BLAST110_.*\\.csv$", full.names = TRUE)

get_label_cols <- function(path) {
  df <- read.csv(path, nrows = 1, header = TRUE, stringsAsFactors = FALSE)
  data.frame(
    file = basename(path),
    ncol = ncol(df),
    cols = paste(colnames(df), collapse = ";"),
    stringsAsFactors = FALSE
  )
}

label_info <- do.call(rbind, lapply(label_files, get_label_cols))

cat("\nLabel CSV column count distribution:\n")
print(table(label_info$ncol))

cat("\nExample label header from first file:\n")
cat(label_info$cols[1], "\n")

if (length(unique(label_info$ncol)) > 1) {
  cat("\nDifferent label structures detected; examples:\n")
  print(label_info[!duplicated(label_info$ncol), c("file", "ncol", "cols")])
}

##############################################
# 3. Test merge on a few samples
##############################################

test_files <- fcs_files[1:min(5, length(fcs_files))]

process_one_sample <- function(fcs_path) {
  fcs_name <- basename(fcs_path)
  base     <- sub("\\.fcs$", "", fcs_name)
  label_path <- file.path(label_dir, paste0(base, ".csv"))

  ff   <- read.FCS(fcs_path, transformation = FALSE, which.lines = 1000)
  expr <- as.data.frame(exprs(ff))
  lab  <- read.csv(label_path, header = TRUE, stringsAsFactors = FALSE)

  if (!("event_ID" %in% colnames(expr))) stop(paste("FCS", fcs_name, "no event_ID"))
  if (!("event_ID" %in% colnames(lab)))  stop(paste("Label", basename(label_path), "no event_ID"))
  if (!("Blast" %in% colnames(lab)))     stop(paste("Label", basename(label_path), "no Blast"))

  merged <- merge(expr, lab, by = "event_ID", all.x = TRUE)
  merged$source_file <- fcs_name
  merged
}

cat("\nTesting merge on up to 5 samples...\n")
test_list <- lapply(test_files, process_one_sample)

cat("Merged column counts for test files:\n")
print(sapply(test_list, ncol))

cat("\nMerged column names for first test file:\n")
print(colnames(test_list[[1]]))
##############################################
# END
##############################################

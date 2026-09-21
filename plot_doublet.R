#!/usr/bin/env Rscript


suppressPackageStartupMessages({
  library(flowCore)
  library(ggplot2)
  library(gridExtra)
})


options(bitmapType = "cairo")

# 1. Read the original (raw) FCS file
fcs_raw <- read.FCS("/storage/mezya.sezen/mphasis/dataset/BLAST110/BLAST110_1_P1.fcs")
dat_raw <- exprs(fcs_raw)
df_raw <- data.frame(FSC.A = dat_raw[, "FSC-A"], FSC.H = dat_raw[, "FSC-H"])

# 2. Read the doublet-removed (cleaned) FCS file
fcs_clean <- read.FCS("/storage/mezya.sezen/mphasis/dataset/final_cleaned/BLAST110_1_P1_Cleaned.fcs")
dat_clean <- exprs(fcs_clean)
df_clean <- data.frame(FSC.A = dat_clean[, "FSC-A"], FSC.H = dat_clean[, "FSC-H"])

# 3. Plot for the original file (before doublet removal)
p_raw <- ggplot(df_raw, aes(x = FSC.A, y = FSC.H)) +
  stat_density_2d(aes(fill = after_stat(level)), geom = "polygon", color = "blue", bins = 20, alpha = 0.2) +
  theme_minimal() +
  labs(title = "FSC-A vs FSC-H (Before Doublet Exclusion)",
       x = "FSC-A (Area)", y = "FSC-H (Height)")

# 4. Plot for the cleaned file (after doublet removal)
p_clean <- ggplot(df_clean, aes(x = FSC.A, y = FSC.H)) +
  stat_density_2d(aes(fill = after_stat(level)), geom = "polygon", color = "blue", bins = 20, alpha = 0.2) +
  theme_minimal() +
  labs(title = "FSC-A vs FSC-H (After Doublet Exclusion)",
       x = "FSC-A (Area)", y = "FSC-H (Height)")

# 5. Save both plots side by side in a single PDF
pdf("/storage/mezya.sezen/mphasis/dataset/final_cleaned/fscah_doublet_density.pdf", width = 12, height = 6)
gridExtra::grid.arrange(p_raw, p_clean, ncol = 2)
dev.off()


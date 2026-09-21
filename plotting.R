#!/usr/bin/env Rscript


suppressPackageStartupMessages({
  library(flowCore)
  library(ggplot2)
})
# --- Step 1: Read FCS file
fcs_file <- "/storage/mezya.sezen/mphasis/dataset/final_cleaned/BLAST110_99_P4_Cleaned.fcs"   # change this
ff <- read.FCS(fcs_file, transformation = FALSE)

# --- Step 2: Convert to dataframe
df <- as.data.frame(exprs(ff))

pdf("FSC_scatter_contour_afterplots.pdf", width = 8, height = 6)

# Scatter plot
print(
  ggplot(df, aes(x = `FSC-A`, y = `FSC-H`)) +
    geom_point(alpha = 0.2, size = 0.5) +
    labs(title = "FSC-A vs FSC-H Scatter Plot",
         x = "FSC-A (Area)",
         y = "FSC-H (Height)") +
    theme_minimal()
)

# Contour (density) plot
print(
  ggplot(df, aes(x = `FSC-A`, y = `FSC-H`)) +
    stat_density_2d(aes(fill = ..level..), geom = "polygon", color = "blue", alpha = 0.3) +
    scale_fill_gradient(low = "lightblue", high = "darkblue") +
    labs(title = "FSC-A vs FSC-H Contour Plot",
         x = "FSC-A (Area)",
         y = "FSC-H (Height)") +
    theme_minimal()
)

# Close PDF device
dev.off()
message("Plots saved to FSC_scatter_contour_afterplots.pdf")

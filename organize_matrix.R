# Load the data
df <- read.csv("labeled_exprMatrix.tsv", stringsAsFactors = FALSE)

# Add a new column 'cell_type' based on the first column content
df$cell_type <- ifelse(grepl("DU145", df[,1]), "DU145", 
                       ifelse(grepl("PC3", df[,1]), "PC3", NA))

# Add a new column 'label' based on the first column content
df$label <- ifelse(grepl("DU145", df[,1]), "Res", 
                   ifelse(grepl("PC3", df[,1]), "Sen", NA))

# Add 'group' column with all values as 1 to potentially merge it with a second group of cells
df$group <- 1

# Add 'batch' column with all values as "X" since we do not know that information
df$batch <- "X"

# Save the updated dataframe
write.csv(df, "final_exprMatrix.tsv", row.names = FALSE)


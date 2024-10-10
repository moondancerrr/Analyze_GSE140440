# Transcriptome Analysis Pipeline of cancer single cell dataset - GSE140440

This pipeline builds an expression matrix (exprMatrix.tsv) by downloading and processing ~38 GB of sequence data using Docker. It performs transcriptome indexing, quantification, filtering, and analysis of the gene expression data. 

## Requirements

- Docker: Ensure that Docker is installed on your machine. This pipeline runs all commands inside Docker containers for consistency.
- Data: This pipeline will download sequence data (~38 GB) from publicly available repositories.
- Scripts: The required Python and R scripts (tx2gene.py, mapLabels.py, map.py, organize_matrix.R, blackbox.py, extract.py) should be available in the scripts folder.

## Pipeline Overview

1. Download Sequence Data
2. Create Decoy Sequence
3. Transcriptome Indexing
4. Salmon Quantification
5. Generate Expression Matrix
6. Gene Filtering and Transposition
7. Labeling
8. Final Matrix Formatting
9. Bayesian Model Analysis



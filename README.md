This folder contains the scripts and data to reproduce the result in the paper "A framework for performing single-cell spatial metabolomics with cell-type specific protein profiling for tissue systems biology"

You can find the raw data here: https://doi.org/10.5281/zenodo.6784251.

# Organization

## Notebooks 
"notebooks" folder contains jupyter notebook script used:
- 01 Processing of IMC (protein) and SIMS (metabolite) images
- 02 Registration of IMC and SIMS images for different imaging regions
- 03 Single-cell level segmentation and visualization of segmentation masks
- 04 Single-cell level intensity extraction and single-cell proteomics clustering
- 05 VAE joint embedding of protein and metabolite modalities 
- 06 Metabolite analysis in different regions (metabolite difference, distance analysis, competition analysis)
- 07 VAE comparison at patient level from lung cancer
- 08 Trajectory analysis 
- 09 Protein metabolite correlation analysis 
- 10 Large FOV analysis of metabolite expression 
- 11 Segmentation of single-cell with Mesmer pipeline
- 12 Comparison of registration from IMC and SIMS modalities

## Source code
"src" folder contains customs scripts used:
- affine transformation
- "utils.py" contains plotting and io custom functions
- "spatial" folder contains custom code for spatial interaction functions 
- "scSpaMet" folder contains keras code used for VAE analysis
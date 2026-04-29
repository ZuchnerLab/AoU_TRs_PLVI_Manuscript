# All of Us CDR V8 Tandem Repeat Analyses
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19239224.svg)](https://doi.org/10.5281/zenodo.19239224)

This repository contains the code necessary to re-create the figures the Danzi, Xu, et al manuscript introducing the PLVI metric for STR prioritization. 

## Project Structure

### Main Figure Notebooks

#### `Figure1.ipynb`
Cohort overview and technical validation analyses:
- **Panel A**: Sunburst plot showing cohort composition by technology (HiFi/ONT), cohort type, and ancestry
- **Panel B**: Correlation analysis of allele length percentiles across cohorts demonstrating technical consistency
- **Panel C**: Discovery saturation curves showing unique allele sequences and lengths observed as sample size increases
- **Panel D**: Distribution of allele sizes per individual across different length bins

#### `Figure2.ipynb`
Tandem repeat sequence composition and motif analysis:
- **Panels A & B**: Waterfall plots showing k-mer segmentation of FGF14 alleles, with and without LPS masking
- **Panel C**: Ridge plot displaying distribution of longest pure segment (LPS) lengths per motif type
- **Panel D**: Saturation curve of novel (non-reference) motifs observed as LPS
- **Panel E**: Relationship between locus length variance and number of novel LPS motifs

#### `Figure3.ipynb`
Pathogenic TR enrichment and variant prioritization:
- **Panel A**: Cumulative distribution function comparing allele length standard deviation between all loci and disease-associated loci
- **Panel B**: Forest plot showing odds ratios for multiple TR variation metrics and their association with pathogenic TRs
- **Panel C**: Strip plots showing pathogenic TR enrichment by specific motifs (CAG, CCG, etc.)
- **Panel D**: Replication of enrichment analysis across different population groups (AFR, AMR, EUR, EAS, SAS)

#### `Figure4.ipynb`
Genomic context and case studies:
- **Panel A**: Scatter plots of TR variation within linkage regions for FAME1, SCA3, and CANVAS
- **Panel B**: GWAS locus analysis for Fuchs Corneal Dystrophy, Intellectual Disability, and Downbeat Nystagmus
- **Panel C**: Prioritization swimlane plot for HD case (HTT CAG expansion) without filtering
- **Panel D**: Prioritization swimlane plot for HD case (HTT CAG expansion) with filtering
- **Panel E**: HPRC spike-in validation showing rank distribution of pathogenic TRs by genic region

#### `Figure5.ipynb`
EP400 case study (novel CAG expansion disorder):
- **Panel A**: Scatter plot of coding CAG repeats showing EP400 among high-variance loci
- **Panel B**: Prioritization swimlane plot for EP400 proband showing the pathogenic TR ranking
- **Panel C**: Pedigrees 
- **Panel D**: Distribution of pure CAG repeat units at EP400 locus across the cohort

#### `Figure6.ipynb`
FAM193B case study (novel CGG expansion disorder):
- **Panel A**: Scatter plot of 5'-UTR CGG repeats showing FAM193B among high-variance loci
- **Panel B**: Prioritization swimlane plot for FAM193B proband
- **Panel C**: Pedigree
- **Panel D**: Distribution of pure CGG repeat units at FAM193B locus across the cohort
- **Panel E**: Expression analysis

### Supplementary Figure Notebooks

#### `SupplementaryFigures_Part1.ipynb`
- Supplementary Figures 1-4

#### `SupplementaryFigures_Part2.ipynb`
- Supplementary Figures 5-14


#### `SupplementaryFigures_Part3.ipynb`
- Supplementary Figures 15-23

#### `SupplementaryFiles.ipynb`
Generation of supplementary data files 1-3 


## Subfolders

### `assets/`
Contains reference data files and input datasets (download from Zenodo: 10.5281/zenodo.19239224)

### `scripts/`
Python utility scripts

### `prioritizeTRWithinSample/`
Sample-level TR prioritization pipeline (download from Zenodo: 10.5281/zenodo.19239224)

### `statsTables/`
Compressed statistics tables for all analyses (download from Zenodo: 10.5281/zenodo.19239224)


### Cohort Structure
- **Discovery Cohort** (n=543): High-coverage HiFi sequencing, median depth 32x, from AoU CDR v8
- **Validation Cohort** (n=2,102): Lower-coverage HiFi sequencing, median depth 12x, from AoU CDR v7 and v8
- **Replication Cohort** (n=500): ONT long-read sequencing, median depth 35x, from 1KGP Long Read Sequencing Consortium

## Environment

### Python Version
- **Python 3.14.0**

### Key Packages

**Data Science & ML Libraries:**
- pandas: 2.3.3
- numpy: 1.23.5
- scipy: 1.10.1
- scikit-learn: 1.2.2
- statsmodels: 0.14.1
- matplotlib: 3.7.1
- seaborn: 0.12.2
- Levenshtein: 0.27.3

**Deep Learning:**
- torch: 2.5.0+cu118
- transformers: 4.40.1
- ligntning: 2.6.1
- huggingface-hub: 0.26.1

**Notebook Environment:**
- jupyter: 3.5.3
- ipython: 8.10.0
- ipykernel: 6.19.2

**Other Notable Libraries:**
- polars: 1.31.0
- numba: 0.65.0
- liftover: 1.1.16
- pyliftover: 0.4
- adjustText: 1.3.0

## Citation

Danzi, M. C., Xu, I. R. L., Fazal, S., Dolzhenko, E., Pellerin, D., Weisburd, B., Reuter, C., Sampson, J., Folland, C., Wheeler, M., O'Donnell-Luria, A., Wuchty, S., Ravenscroft, G., Eberle, M. A., and the All of Us Research Program Long Read Working Group, & Zuchner, S. (2025). Detailed tandem repeat allele profiling in 1,027 long-read genomes reveals genome-wide patterns of pathogenicity. *bioRxiv*. https://doi.org/10.1101/2025.01.06.631535

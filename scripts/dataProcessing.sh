#!/bin/bash

## This script shows the commands used to
# 1) run TRGT on the bam files, 
# 2) process the TRGT output VCFs to generate the longest pure segment (LPS) data,  
# 3) generate the quantiles and other stats files used in the manuscript, 
# 4) Annotate each TR to a genic region, 
# 5) run ExpansionHunter on the short-read WGS bam files,
# 6) assemble the set of statistics used to train the TRConstraint model and prepare the input bed files, and
# 7) train the TRConstraint model and use it to make predictions on all TRs within 10kb of genes.


# This script is not intended to be run as-is, but rather to document the commands used for data processing (because the input bam files 
# are stored within the All of Us Researcher Workbench and should ideally be processed in parallel per sample using WDL workflows or dsub 
# or similar approaches rather than for loops as shown here).

# Assume you have a list of sampleID, sequencing center names, and their corresponding bam file paths in a file called 'sample_bam_list.txt' with tab-separated values
# as well as a file called 'sample_list.txt' with the sampleIDs, sequencing center names, and whether the sample belongs in the high or low coverage groups, one set per line.

# Step 1: Run TRGT on each bam file
# Note this was done using TRGT version 5.0.0
while read sampleID centerName bamPath; do
    trgt --genome hg38.no_alt.fa --repeats adotto_TRregions_TRGTFormatWithFlankingSeq_v1.0_under1kb.bed --reads ${bamPath} --output ${sampleID}_trgt --threads $(nproc)
done < sample_bam_list.txt

# Step 2: Process TRGT output VCFs to generate LPS data
while read sampleID centerName highOrLowCoverage; do
    bcftools query -f "${sampleID}\t${centerName}\t%INFO/TRID\t%INFO/MOTIFS\t[%TGT\t%AL{0}\t%AL{1}\t%SD{0}\t%SD{1}\t%AM{0}\t%AM{1}]" ${sampleID}_trgt.TR_Explorer_1.01.vcf.gz > ${sampleID}.txt
    python scripts/calculateLPS.py --sample=${sampleID}
    for CHROM in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 X Y; do
        if [[ $highOrLowCoverage == "high" ]]; then 
            grep -P "^${sampleID}\t${centerName}\t${CHROM}-" ~{sample_id}_trgt.TR_Explorer_1.0.1_LPS.txt | gzip -c >> discoveryCohort/alleleInfo_chr${CHROM}_trgt.TR_Explorer_1.0.1.txt.gz
        else 
            grep -P "^${sampleID}\t${centerName}\t${CHROM}-" ~{sample_id}_trgt.TR_Explorer_1.0.1_LPS.txt | gzip -c >> validationCohort/alleleInfo_chr${CHROM}_trgt.TR_Explorer_1.0.1.txt.gz
        fi
done < sample_list.txt

# Step 3: Generate quantiles files used in manuscript (alleleLengthStats, lpsPerMotifStats, and lpsPerLocusStats)
cd discoveryCohort/
for CHROM in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 X Y; do
    python scripts/calculateLPSStats.py --chrom=${CHROM} --catalog_name="TR_Explorer_1.0.1.txt.gz"
cd ../validationCohort/
for CHROM in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 X Y; do
    python scripts/calculateLPSStats.py --chrom=${CHROM} --catalog_name="TR_Explorer_1.0.1.txt.gz"
cd ../

# Step 4: Annotate each TR to a genic region
scripts/generateGenicRegionAnnotation.sh

# Step 5: Run ExpansionHunter on the short-read WGS bam files
# Note: this was done using ExpansionHunter version 5.0.0
# Assume you have a file called 'srWGS_bam_list.txt' with the sampleIDs, and their corresponding bam file paths for the short-read data. 
while read sampleID bamPath; do
    ExpansionHunter \
    --reference GRCh38_full_analysis_set_plus_decoy_hla.fa \
    --reads ${bamPath} \
    --variant-catalog assets/EH_variant_catalog.json \
    --analysis-mode seeking \
    --cache-mates \
    --output-prefix EH_results/wgs_${sampleID}
done < srWGS_bam_list.txt
# aggregate the results
python scripts/aggregateEHResults.py

# Step 6: Create inputs for TRConstraint model training
# 6.1 create bed files of the genic and intergenic TRIDs (within 10kb of any gene annotated in Gencode basic or not)
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.basic.annotation.gtf.gz
gunzip gencode.v49.basic.annotation.gtf.gz
gunzip -c assets/variation_clusters_and_isolated_TRs_v1.0.1.hg38.TRGT.bed.gz > variation_clusters_and_isolated_TRs_v1.0.1.hg38.TRGT.bed
# Get genic and intergenic regions
awk -F'\t' -v OFS='\t' '{if($3=="gene"){print $0}}' gencode.v49.basic.annotation.gtf > GRCh38.gencode.v49.basic.genes.gtf
cut -f 1,4,5 GRCh38.gencode.v49.basic.genes.gtf | bedtools slop -i stdin -g GRCh38.chromSizes -b 10000 > GRCh38.gencode.v49.basic.genes_10kbSlopped.bed
bedtools sort -i GRCh38.gencode.v49.basic.genes_10kbSlopped.bed | bedtools merge -i stdin > GRCh38.gencode.v49.basic.genes_10kbSloppedMerged.bed
bedtools intersect -v -a variation_clusters_and_isolated_TRs_v1.0.1.hg38.TRGT.bed -b GRCh38.gencode.v49.basic.genes_10kbSloppedMerged.bed > TR-Explorer_v1.0.1_intergenic_10kb.bed
bedtools intersect -u -a variation_clusters_and_isolated_TRs_v1.0.1.hg38.TRGT.bed -b GRCh38.gencode.v49.basic.genes_10kbSloppedMerged.bed > TR-Explorer_v1.0.1_nonIntergenic_10kb.bed
# 6.2 create TRConstraint input files (named training_10kbFromGenes.bed, validaiton_10kbFromGenes.bed, and testing_within10kbOfGenes.bed)
python createTRConstraintInputs.py --lpsPerMotif statsTables/DiscoveryCohort_lpsPerMotifStats.TR_Explorer_1.0.1.txt.gz \
--lpsMA statsTables/DiscoveryCohort_lpsPerLocusStats.TR_Explorer_1.0.1.txt.gz --cps statsTables/DiscoveryCohort_lpsAllMotifStats.TR_Explorer_1.0.1.txt.gz \
--alleleLen statsTables/DiscoveryCohort_alleleLengthStats.TR_Explorer_1.0.1.txt.gz \
--genicBed TR-Explorer_v1.0.1_nonIntergenic_10kb.bed --intergenicBed TR-Explorer_v1.0.1_intergenic_10kb.bed
# 6.3 create TRConstraint input fasta files
tail -n +2 training_10kbFromGenes.bed | bedtools slop -i stdin -g GRCh38.chromSizes -b 75 | bedtools getfasta -fi GRCh38_full_analysis_set_plus_decoy_hla.fa -bed stdin -tab -nameOnly > training_10kbFromGenes_seqs_75bp.txt
tail -n +2 validation_10kbFromGenes.bed | bedtools slop -i stdin -g GRCh38.chromSizes -b 75 | bedtools getfasta -fi GRCh38_full_analysis_set_plus_decoy_hla.fa -bed stdin -tab -nameOnly > validation_10kbFromGenes_seqs_75bp.txt
tail -n +2 testing_within10kbOfGenes.bed | bedtools slop -i stdin -g GRCh38.chromSizes -b 75 | bedtools getfasta -fi GRCh38_full_analysis_set_plus_decoy_hla.fa -bed stdin -tab -nameOnly > testing_within10kbOfGenes_seqs_75bp.txt

# Step 7: Train TRConstraint model and use it to make predictions on all TRs within 10kb of genes
# Note that this step requires pytorch and the transformers and lightning libraries to be installed. 
# The use of a GPU is strongly recommended. Batch size recalibration will be needed to obtain good performance. 
# I did this training on an NVIDIA L40S GPU.
# 7.1 Train the model:
python scripts/trainTRConstraintModel.py
# 7.2 Run inference on loci:
python scripts/inferenceWithTRConstraintModel.py

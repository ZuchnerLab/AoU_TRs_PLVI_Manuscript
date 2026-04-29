#!/bin/bash
wget https://ftp.ebi.ac.uk/pub/ensembl/release-115/gtf/homo_sapiens/Homo_sapiens.GRCh38.115.gtf.gz
gunzip Homo_sapiens.GRCh38.115.gtf.gz
grep 'gencode_basic' Homo_sapiens.GRCh38.115.gtf > GRCh38.115.gencodeBasic.gtf

# get gencodeBasic transcripts, exons, CDSs, and UTRs
awk -F'\t' -v OFS='\t' '{if($3=="transcript"){print $0}}' GRCh38.115.gencodeBasic.gtf | grep -v '^KI' | grep -v '^GL' | sed 's/^/chr/' | sed 's/^chrMT/chrM/' > GRCh38.115.gencodeBasic.transcripts.gtf
awk -F'\t' -v OFS='\t' '{if($3=="exon"){print $0}}' GRCh38.115.gencodeBasic.gtf | grep -v '^KI' | grep -v '^GL' | sed 's/^/chr/' | sed 's/^chrMT/chrM/' > GRCh38.115.gencodeBasic.exons.gtf
awk -F'\t' -v OFS='\t' '{if($3=="CDS"){print $0}}' GRCh38.115.gencodeBasic.gtf | grep -v '^KI' | grep -v '^GL' | sed 's/^/chr/' | sed 's/^chrMT/chrM/' > GRCh38.115.gencodeBasic.CDSs.gtf
awk -F'\t' -v OFS='\t' '{if($3=="five_prime_utr"){print $0}}' GRCh38.115.gencodeBasic.gtf | grep -v '^KI' | grep -v '^GL' | sed 's/^/chr/' | sed 's/^chrMT/chrM/' > GRCh38.115.gencodeBasic.UTR5.gtf
awk -F'\t' -v OFS='\t' '{if($3=="three_prime_utr"){print $0}}' GRCh38.115.gencodeBasic.gtf | grep -v '^KI' | grep -v '^GL' | sed 's/^/chr/' | sed 's/^chrMT/chrM/' > GRCh38.115.gencodeBasic.UTR3.gtf

# get protein-coding subset of gencodeBasic transcripts, exons, CDSs, and UTRs
grep 'protein_coding' GRCh38.115.gencodeBasic.transcripts.gtf | cut -f 9 | cut -d'"' -f 6 > proteinCodingTranscripts.txt
grep -wFf proteinCodingTranscripts.txt GRCh38.115.gencodeBasic.transcripts.gtf > GRCh38.115.gencodeBasic.coding.transcripts.gtf
grep -wFf proteinCodingTranscripts.txt GRCh38.115.gencodeBasic.exons.gtf > GRCh38.115.gencodeBasic.coding.exons.gtf
grep -wFf proteinCodingTranscripts.txt GRCh38.115.gencodeBasic.CDSs.gtf > GRCh38.115.gencodeBasic.coding.CDSs.gtf
grep -wFf proteinCodingTranscripts.txt GRCh38.115.gencodeBasic.UTR5.gtf > GRCh38.115.gencodeBasic.coding.UTR5.gtf
grep -wFf proteinCodingTranscripts.txt GRCh38.115.gencodeBasic.UTR3.gtf > GRCh38.115.gencodeBasic.coding.UTR3.gtf

# sort GTF files
bedtools sort -i GRCh38.115.gencodeBasic.coding.transcripts.gtf -g hg38.chrom_sizes > GRCh38.115.gencodeBasic.coding.sorted.transcripts.gtf
bedtools sort -i GRCh38.115.gencodeBasic.coding.exons.gtf -g hg38.chrom_sizes > GRCh38.115.gencodeBasic.coding.sorted.exons.gtf
bedtools sort -i GRCh38.115.gencodeBasic.coding.CDSs.gtf -g hg38.chrom_sizes > GRCh38.115.gencodeBasic.coding.sorted.CDSs.gtf
bedtools sort -i GRCh38.115.gencodeBasic.coding.UTR5.gtf -g hg38.chrom_sizes > GRCh38.115.gencodeBasic.coding.sorted.UTR5.gtf
bedtools sort -i GRCh38.115.gencodeBasic.coding.UTR3.gtf -g hg38.chrom_sizes > GRCh38.115.gencodeBasic.coding.sorted.UTR3.gtf

# get intergenic regions (everything not in the transcript spans)
bedtools complement -i GRCh38.115.gencodeBasic.coding.sorted.transcripts.gtf -g hg38.chrom_sizes > intergenic.bed

# get intronic regions (everything not in the exonic or intergenic spans)
cut -f 1,4,5 GRCh38.115.gencodeBasic.coding.exons.gtf | cat - intergenic.bed | bedtools sort -i stdin -g hg38.chrom_sizes | bedtools complement -i stdin -g hg38.chrom_sizes > intronic.bed

# get coding regions (the CDS regions)
cut -f 1,4,5 GRCh38.115.gencodeBasic.coding.CDSs.gtf > coding.bed

# get 5'UTRs and 3'UTRs
cut -f 1,4,5 GRCh38.115.gencodeBasic.coding.sorted.UTR5.gtf > utr5.bed
cut -f 1,4,5 GRCh38.115.gencodeBasic.coding.sorted.UTR3.gtf > utr3.bed


# compile bed regions as an annotation set
awk '{print $0"\tcoding"}' coding.bed > annotations.labeled.bed
awk '{print $0"\tintron"}' intronic.bed >> annotations.labeled.bed
awk '{print $0"\tintergenic"}' intergenic.bed >> annotations.labeled.bed
awk '{print $0"\tUTR_5"}' utr5.bed >> annotations.labeled.bed
awk '{print $0"\tUTR_3"}' utr3.bed >> annotations.labeled.bed
bedtools sort -i annotations.labeled.bed -g hg38.chrom_sizes > annotations.labeled.sorted.bed

# match up each TR with all regions that it overlaps
zcat variation_clusters_and_isolated_TRs_v1.0.1.hg38.TRGT.bed.gz | cut -d';' -f 1 | awk -F'\t' -v OFS='\t' '{print $1,$2,$3,substr($4,4)}' > catalog.bed
bedtools intersect -wo -a catalog.bed -b annotations.labeled.sorted.bed > overlaps_with_lengths.txt 

# pick the annotation with the most overlap
awk '
{
  key = $4;
  if (olap[key] < $9) {
    olap[key] = $9;
    type[key] = $8;
  }
}
END {
  for (k in olap) {
    print k"\t"type[k];
  }
}' overlaps_with_lengths.txt > genicRegionAnnotations.txt
gzip genicRegionAnnotations.txt

# if there were any non-matched TRIDs, we would assign them an intergenic annotation, but all matched up for me when running this
# note that I had to adjust a few of the known pathogenic ones to correct some coding vs UTR vs intron distinctions where it overlapped multiple. 

# now get gene name to TRID assignments 
awk -F'\t' -v OFS='\t' '{if($3=="gene"){print $0}}' Homo_sapiens.GRCh38.115.gtf | grep -v '^KI' | grep -v '^GL' | sed 's/^/chr/' | sed 's/^chrMT/chrM/' > GRCh38.115.genes.gtf
grep 'protein_coding' GRCh38.115.genes.gtf | tr '"' '\t' | cut -f 1,4,5,14 | grep -v 'havana' > proteinCodingGenes.bed

cat GRCh38.115.gencodeBasic.coding.sorted.transcripts.gtf | tr '"' '\t' | cut -f 1,4,5,18 | grep -v 'havana' > proteinCodingGenes.bed
bedtools complement -i proteinCodingGenes.bed -g hg38.chrom_sizes | awk '{print $0"\tintergenic"}' > intergenic.bed
cat intergenic.bed proteinCodingGenes.bed | bedtools sort -i stdin -g hg38.chrom_sizes > proteinCodingGenesAndIntergenicRegions.bed
bedtools intersect -wo -a catalog.bed -b proteinCodingGenesAndIntergenicRegions.bed  > overlaps_with_lengths2.txt

awk '
{
  key = $4;
  if (olap[key] < $9) {
    olap[key] = $9;
    type[key] = $8;
  }
}
END {
  for (k in olap) {
    print k"\t"type[k];
  }
}' overlaps_with_lengths2.txt > TR_to_gene_name.txt
gzip TR_to_gene_name.txt


#!/bin/bash
set -euo pipefail

## Lift the TR-Explorer v1.0.1 TRGT catalog from hg38 to hg19, keeping the
## original TRIDs (which encode hg38-based coordinates) unchanged so that
## downstream analyses keyed on TRID remain consistent with the hg38 catalog.
##
## Requirements on PATH: liftOver (UCSC), bedtools, bgzip, awk, sort, wget.
## liftOver: https://hgdownload.soe.ucsc.edu/admin/exe/
##
## Inputs:
##   assets/variation_clusters_and_isolated_TRs_v1.0.1.hg38.TRGT.bed.gz
## Outputs (written to assets/):
##   variation_clusters_and_isolated_TRs_v1.0.1.hg19.TRGT.bed.gz  -- lifted catalog
##   variation_clusters_and_isolated_TRs_v1.0.1.hg38_to_hg19.unmapped.txt  -- entries that failed to lift

CATALOG_URL="https://github.com/broadinstitute/trexplorer-catalog/releases/download/v1.0.1/variation_clusters_and_isolated_TRs_v1.0.1.hg38.TRGT.bed.gz"
CHAIN_URL="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/hg38ToHg19.over.chain.gz"

ASSETS_DIR="assets"
WORK_DIR="liftover_work"
mkdir -p "${ASSETS_DIR}" "${WORK_DIR}"

HG38_CATALOG="${ASSETS_DIR}/variation_clusters_and_isolated_TRs_v1.0.1.hg38.TRGT.bed.gz"
CHAIN_FILE="${WORK_DIR}/hg38ToHg19.over.chain.gz"
HG38_BED="${WORK_DIR}/variation_clusters_and_isolated_TRs_v1.0.1.hg38.TRGT.bed"
HG19_MAPPED="${WORK_DIR}/variation_clusters_and_isolated_TRs_v1.0.1.hg19.TRGT.mapped.bed"
HG19_UNMAPPED="${ASSETS_DIR}/variation_clusters_and_isolated_TRs_v1.0.1.hg38_to_hg19.unmapped.txt"
HG19_SORTED="${ASSETS_DIR}/variation_clusters_and_isolated_TRs_v1.0.1.hg19.TRGT.bed"

# Step 1: fetch catalog if not already cached
if [[ ! -s "${HG38_CATALOG}" ]]; then
    echo "Downloading hg38 catalog..."
    wget -O "${HG38_CATALOG}" "${CATALOG_URL}"
fi

# Step 2: fetch UCSC hg38 -> hg19 chain
if [[ ! -s "${CHAIN_FILE}" ]]; then
    echo "Downloading hg38ToHg19 chain..."
    wget -O "${CHAIN_FILE}" "${CHAIN_URL}"
fi

# Step 3: decompress catalog (liftOver reads plain BED)
gunzip -c "${HG38_CATALOG}" > "${HG38_BED}"

# Step 4: run liftOver.
# liftOver preserves column 4 verbatim, so the ID=...;MOTIFS=...;STRUC=... field
# (which encodes the original hg38 coordinates in the TRID) is carried through unchanged.
# -minMatch=0.95 is the default; loci that cannot be uniquely re-mapped land in the unmapped file.
echo "Running liftOver..."
liftOver \
    "${HG38_BED}" \
    "${CHAIN_FILE}" \
    "${HG19_MAPPED}" \
    "${HG19_UNMAPPED}"

# Step 5: sort by hg19 chrom/start (TRGT expects sorted input) and bgzip.
# Chromosomes are kept in the UCSC "chr" naming that liftOver emits, matching the input catalog style.
echo "Sorting lifted catalog..."
sort -k1,1 -k2,2n "${HG19_MAPPED}" > "${HG19_SORTED}"

echo "Compressing with bgzip..."
bgzip -f "${HG19_SORTED}"

# Step 6: quick sanity report
N_IN=$(wc -l < "${HG38_BED}")
N_OUT=$(zcat "${HG19_SORTED}.gz" | wc -l)
N_UNMAPPED=$(grep -cv '^#' "${HG19_UNMAPPED}" || true)
echo "hg38 input loci:      ${N_IN}"
echo "hg19 lifted loci:     ${N_OUT}"
echo "unmapped loci:        ${N_UNMAPPED}"
echo "Wrote: ${HG19_SORTED}.gz"
echo "Wrote: ${HG19_UNMAPPED}"

# Step 7: cleanup intermediate files (comment out to keep them for debugging)
rm -f "${HG38_BED}" "${HG19_MAPPED}"

import polars as pl
import sys, getopt
from typing import Optional

def simplest_motif(motif):
    i = (motif+motif).find(motif, 1, -1)
    if i != -1:
        motif = motif[:i]
    motifs = []
    b = len(motif)
    for i in range(b):
        c = motif[i:]+motif[:i]
        motifs.append(c)
    motifs = [x.upper() for x in motifs]
    return(min(motifs))

def get_quantile_names():
    quantiles = [0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999, 1.0]
    names = {q: f"{int(q*100) if q*100 in [0,1,5] else int(q*100)}{'th' if int(q*100) not in [1,2,3] else ('st' if int(q*100)==1 else ('nd' if int(q*100)==2 else 'rd'))}Percentile" for q in quantiles}
    names[0.999] = '99.9thPercentile'
    names[0.0] = '0thPercentile'
    names[1.0] = '100thPercentile'
    return names

def get_stats(df: pl.DataFrame, group_by_cols: list[str], value_col: str, extra_aggs: Optional[list] = None) -> pl.DataFrame:
    quantile_names = get_quantile_names()

    aggs = [
        pl.mean(value_col).alias('mean'),
        pl.std(value_col).alias('std'),
        pl.col(value_col).mode().first().alias('mode'),
        pl.col(value_col).sub(pl.col(value_col).median()).abs().median().alias("mad"),
        pl.n_unique(value_col).alias('uniqueLengths'),
        pl.len().alias('N'),
        *[pl.quantile(value_col, q).alias(name) for q, name in quantile_names.items()]
    ]

    if extra_aggs:
        aggs.extend(extra_aggs)

    stats_df = df.group_by(group_by_cols).agg(aggs)

    column_order = group_by_cols + ['N'] + list(quantile_names.values()) + ['mean', 'std', 'mad', 'mode', 'uniqueLengths']
    if extra_aggs:
        column_order += [agg.meta.output_name() for agg in extra_aggs]

    return stats_df.select(column_order)

def calculateCPS_vectorized(lps_stats: pl.DataFrame) -> pl.DataFrame:
    # Self-join to create all pairs of motifs within each TRID
    pairs = lps_stats.join(lps_stats, on="TRID", how="inner", suffix="_right")

    # Filter for unique pairs
    pairs = pairs.filter(pl.col("LPS_Motif") < pl.col("LPS_Motif_right"))

    # Calculate intersection and union sizes
    pairs = pairs.with_columns([
        pl.struct(["LPS_Motif", "LPS_Motif_right"]).map_elements(
            lambda x: len(set(x["LPS_Motif"]).intersection(set(x["LPS_Motif_right"]))),
            return_dtype=pl.Int64
        ).alias("intersection_len"),
        pl.struct(["LPS_Motif", "LPS_Motif_right"]).map_elements(
            lambda x: len(set(x["LPS_Motif"]).union(set(x["LPS_Motif_right"]))),
            return_dtype=pl.Int64
        ).alias("union_len")
    ])

    # Calculate CDS part for each pair
    pairs = pairs.with_columns(
        ((1 - pl.col("intersection_len") / pl.col("union_len")) * pl.col("N_motif") * pl.col("N_motif_right")).alias("cds_part")
    )

    # Sum CDS parts for each TRID
    cds_sum = pairs.group_by("TRID").agg(pl.sum("cds_part").alias("cds"))

    # Get total counts for each TRID
    total_counts = lps_stats.group_by("TRID").agg(pl.sum("N_motif").alias("total_count"))

    # Join and calculate final CPS
    cps_df = total_counts.join(cds_sum, on="TRID", how="left").fill_null(0)
    cps_df = cps_df.with_columns(
        (pl.col("cds") / ((pl.col("total_count") * (pl.col("total_count") - 1)) / 2)).alias("CPS")
    )

    return cps_df.select(["TRID", "CPS"])

def main ( argv ):
    CHROM=""
    CatalogName=""
    try:
        opts, args = getopt.getopt(argv,"h",["chrom=","catalog_name=","help"])
    except getopt.GetoptError:
        print('calculateLPSStats_downsampled.py --chrom=<Chromosome> --catalog_name=<CatalogName>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('--chrom'):
            CHROM=arg
        elif opt in ('--catalog_name'):
            CatalogName=arg
        elif opt in ('h','--help'):
            print('calculateLPSStats_downsampled.py --chrom=<Chromosome> --catalog_name=<CatalogName>')
            sys.exit()
        else:
            print('calculateLPSStats_downsampled.py --chrom=<Chromosome> --catalog_name=<CatalogName>')
            sys.exit()
    # Load metadata from pickle and convert to polars
    metadata = pl.read_csv('V8_metadata.txt',separator='\t')

    # read in the alleleInfo file
    alleleInfo = pl.read_csv(f'alleleInfo_chr{CHROM}_trgt.{CatalogName}.txt.gz', separator='\t',infer_schema=False, has_header=True, schema={
        'SampleID': pl.Int64,'CenterName':pl.String,'TRID': pl.String, 'Allele1': pl.String, 'Allele2': pl.String,
        'Allele1_LPS_Length': pl.Int64, 'Allele1_LPS_Motif': pl.String,
        'Allele2_LPS_Length': pl.Int64, 'Allele2_LPS_Motif': pl.String,
        'Allele1_Len': pl.String, 'Allele2_Len': pl.String,
        'Allele1_DP': pl.String, 'Allele2_DP': pl.String,
        'Allele1_Methylation': pl.String, 'Allele2_Methylation': pl.String})
    # drop duplicate samples (keeping whichever one had greater depth in metadata table)
    alleleInfo = alleleInfo.join(metadata.select(['participant', 'GC']).rename({'participant': 'SampleID', 'GC': 'CenterName'}).with_columns(pl.col('SampleID').cast(pl.Int64)), on=['SampleID', 'CenterName'], how='inner')

    # Get the sorted unique sample IDs
    sorted_unique_samples = sorted(alleleInfo['SampleID'].unique().to_list())

    # Iterate over downsampled subset sizes
    for subset_size in range(20, 541, 20):
        if subset_size > len(sorted_unique_samples):
            print(f'Skipping subset_size={subset_size}: only {len(sorted_unique_samples)} unique samples available')
            continue

        subset_samples = sorted_unique_samples[:subset_size]
        alleleInfo_subset = alleleInfo.filter(pl.col('SampleID').is_in(subset_samples))

        ### LPS analyses
        # melt dataframe to get alleles 1 and 2 on separate rows
        alleles1 = alleleInfo_subset.select(['SampleID', 'TRID', 'Allele1_LPS_Length', 'Allele1_LPS_Motif']).rename({'Allele1_LPS_Length': 'LPS_Length', 'Allele1_LPS_Motif': 'LPS_Motif'})
        alleles2 = alleleInfo_subset.select(['SampleID', 'TRID', 'Allele2_LPS_Length', 'Allele2_LPS_Motif']).rename({'Allele2_LPS_Length': 'LPS_Length', 'Allele2_LPS_Motif': 'LPS_Motif'})
        allelesMelted = pl.concat([alleles1, alleles2])
        # filter out rows with missing values
        allelesMelted2 = allelesMelted.drop_nulls()
        allelesMelted2 = allelesMelted2.filter(pl.col('LPS_Length') > 0)

        # Apply simplest_motif to LPS_Motif column
        allelesMelted2 = allelesMelted2.with_columns([
            pl.col('LPS_Motif').map_elements(simplest_motif, return_dtype=pl.String).alias('LPS_Motif')
        ])

        # Stats per LPS Motif
        lpsStats = get_stats(allelesMelted2, group_by_cols=['TRID', 'LPS_Motif'], value_col='LPS_Length')
        lpsStats = lpsStats.rename({"N": "N_motif"})
        lpsStats.write_csv(f'lpsPerMotifStats_chr{CHROM}_trgt.{CatalogName}_subset_n{subset_size}.txt', separator='\t')

        # Stats combined for all LPS Motifs
        lpsStatsCombined = get_stats(allelesMelted2, group_by_cols=['TRID'], value_col='LPS_Length')
        # calculate CPS scores
        cpsValues = calculateCPS_vectorized(lpsStats)
        lpsStatsCombined = lpsStatsCombined.join(cpsValues, on='TRID', how='left')
        lpsStatsCombined.write_csv(f'lpsPerLocusStats_chr{CHROM}_trgt.{CatalogName}_subset_n{subset_size}.txt', separator='\t')

        # Stats for allele lengths
        alleles1 = alleleInfo_subset.select(['SampleID', 'TRID', 'Allele1', 'Allele1_Len']).rename({'Allele1_Len': 'Allele_Length', 'Allele1': 'Allele'})
        alleles2 = alleleInfo_subset.select(['SampleID', 'TRID', 'Allele2', 'Allele2_Len']).rename({'Allele2_Len': 'Allele_Length', 'Allele2': 'Allele'})
        allelesMeltedNonLPS = pl.concat([alleles1, alleles2])
        allelesMeltedNonLPS = allelesMeltedNonLPS.with_columns(pl.col('Allele_Length').cast(pl.Int64, strict=False))
        allelesMeltedNonLPS = allelesMeltedNonLPS.drop_nulls(subset=['Allele_Length'])

        alleleLengthStats = get_stats(allelesMeltedNonLPS, group_by_cols=['TRID'], value_col='Allele_Length', extra_aggs=[pl.col('Allele').unique().count().alias('uniqueAlleleSeqs')])
        alleleLengthStats.write_csv(f'alleleLengthStats_chr{CHROM}_trgt.{CatalogName}_subset_n{subset_size}.txt', separator='\t')

if __name__ == "__main__":
    main(sys.argv[1:])

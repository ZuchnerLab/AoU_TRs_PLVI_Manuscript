import polars as pl
import pandas as pd
from tqdm import tqdm

# Configuration
CHROMS = ["22"]  # Will be expanded to all chromosomes

# ============================================================================
# STEP 1: Convert lenStats to Polars
# ============================================================================
print("Preprocessing lenStats for fast lookup...")

# Convert to Polars DataFrame
lenStats_pl = pl.DataFrame({
    'TRID': lenStats['TRID'].values,
    'LPS_Motif': lenStats['LPS_Motif'].values,
    '99thPercentile': lenStats['99thPercentile'].values,
    '99.9thPercentile': lenStats['99.9thPercentile'].values,
    '100thPercentile': lenStats['100thPercentile'].values
})

print(f"Loaded thresholds for {len(lenStats_pl)} TRIDs")

# ============================================================================
# STEP 2: Get participant list
# ============================================================================
print("Setting up participant list...")
participant_list = combinedMetadata.participant.values
print(f"Tracking {len(participant_list)} participants")

# ============================================================================
# STEP 3: Process chromosomes with Polars STREAMING mode
# ============================================================================
# Two-pass approach per allele to reduce memory usage:
# Pass 1: Inner join for known TRID+Motif combinations
# Pass 2: Anti-join for novel TRID+Motif combinations

all_results = []

for chrom in tqdm(CHROMS, desc="Chromosomes"):
    print(f'\nProcessing Chromosome {chrom} with streaming engine...')

    # Download file (uncomment when ready)
    #! gcloud storage cp --no-clobber gs://fc-secure-a3225c5b-3828-4657-8a82-98874f6652a8/alleleInfo_v8/alleleInfo_chr{chrom}_trgt.TR_Explorer_1.0.1.txt.gz .

    file_path = f'alleleInfo_chr{chrom}_trgt.TR_Explorer_1.0.1.txt.gz'

    # ========================================================================
    # ALLELE 1 - PASS 1: Inner join for known combinations
    # ========================================================================
    allele1_matched = (
        pl.scan_csv(
            file_path,
            separator='\t',
            has_header=True,
            infer_schema_length=0
        )
        # Inner join - only matched TRID+Motif combinations
        .join(
            lenStats_pl.lazy(),
            left_on=['TRID', 'Allele1_LPS_Motif'],
            right_on=['TRID', 'LPS_Motif'],
            how='inner'
        )
        # Convert allele length: handle '.' and convert to int
        .with_columns([
            pl.when(pl.col('Allele1_LPS_Length') == '.')
              .then(pl.lit(0))
              .otherwise(
                  pl.col('Allele1_LPS_Length')
                    .str.to_integer(strict=False)
                    .fill_null(0)
              )
              .alias('Allele1_LPS_Length_Int'),
        ])
        # Score based on threshold comparison (no nulls to handle!)
        .with_columns([
            pl.when(pl.col('Allele1_LPS_Length_Int') > pl.col('99thPercentile'))
              .then(pl.lit(1))
              .otherwise(pl.lit(0))
              .cast(pl.Int8)
              .alias('allele1_count_99'),

            pl.when(pl.col('Allele1_LPS_Length_Int') > pl.col('99.9thPercentile'))
              .then(pl.lit(1))
              .otherwise(pl.lit(0))
              .cast(pl.Int8)
              .alias('allele1_count_999'),

            pl.when(pl.col('Allele1_LPS_Length_Int') > pl.col('100thPercentile'))
              .then(pl.lit(1))
              .otherwise(pl.lit(0))
              .cast(pl.Int8)
              .alias('allele1_count_100'),
        ])
        .select(['SampleID', 'TRID', 'Allele1_LPS_Motif', 'allele1_count_99', 'allele1_count_999', 'allele1_count_100'])
        .collect(engine="streaming")
    )

    # ========================================================================
    # ALLELE 1 - PASS 2: Anti-join for novel combinations
    # ========================================================================
    allele1_novel = (
        pl.scan_csv(
            file_path,
            separator='\t',
            has_header=True,
            infer_schema_length=0
        )
        # Anti-join - only unmatched TRID+Motif combinations
        .join(
            lenStats_pl.lazy(),
            left_on=['TRID', 'Allele1_LPS_Motif'],
            right_on=['TRID', 'LPS_Motif'],
            how='anti'
        )
        # Convert allele length
        .with_columns([
            pl.when(pl.col('Allele1_LPS_Length') == '.')
              .then(pl.lit(0))
              .otherwise(
                  pl.col('Allele1_LPS_Length')
                    .str.to_integer(strict=False)
                    .fill_null(0)
              )
              .alias('Allele1_LPS_Length_Int'),
        ])
        # Novel motif gets 1 point if length > 0
        .with_columns([
            pl.when(pl.col('Allele1_LPS_Length_Int') > 0)
              .then(pl.lit(1))
              .otherwise(pl.lit(0))
              .cast(pl.Int8)
              .alias('allele1_count_99'),

            pl.when(pl.col('Allele1_LPS_Length_Int') > 0)
              .then(pl.lit(1))
              .otherwise(pl.lit(0))
              .cast(pl.Int8)
              .alias('allele1_count_999'),

            pl.when(pl.col('Allele1_LPS_Length_Int') > 0)
              .then(pl.lit(1))
              .otherwise(pl.lit(0))
              .cast(pl.Int8)
              .alias('allele1_count_100'),
        ])
        .select(['SampleID', 'allele1_count_99', 'allele1_count_999', 'allele1_count_100'])
        .collect(engine="streaming")
    )

    # Combine Allele1 matched and novel
    allele1_result = pl.concat([
        allele1_matched.select(['SampleID', 'allele1_count_99', 'allele1_count_999', 'allele1_count_100']),
        allele1_novel
    ])

    # ========================================================================
    # ALLELE 2 - PASS 1: Inner join for known combinations
    # ========================================================================
    allele2_matched = (
        pl.scan_csv(
            file_path,
            separator='\t',
            has_header=True,
            infer_schema_length=0
        )
        # Inner join - only matched TRID+Motif combinations
        .join(
            lenStats_pl.lazy(),
            left_on=['TRID', 'Allele2_LPS_Motif'],
            right_on=['TRID', 'LPS_Motif'],
            how='inner'
        )
        # Convert allele length: handle '.' and convert to int
        .with_columns([
            pl.when(pl.col('Allele2_LPS_Length') == '.')
              .then(pl.lit(0))
              .otherwise(
                  pl.col('Allele2_LPS_Length')
                    .str.to_integer(strict=False)
                    .fill_null(0)
              )
              .alias('Allele2_LPS_Length_Int'),
        ])
        # Score based on threshold comparison (no nulls to handle!)
        .with_columns([
            pl.when(pl.col('Allele2_LPS_Length_Int') > pl.col('99thPercentile'))
              .then(pl.lit(1))
              .otherwise(pl.lit(0))
              .cast(pl.Int8)
              .alias('allele2_count_99'),

            pl.when(pl.col('Allele2_LPS_Length_Int') > pl.col('99.9thPercentile'))
              .then(pl.lit(1))
              .otherwise(pl.lit(0))
              .cast(pl.Int8)
              .alias('allele2_count_999'),

            pl.when(pl.col('Allele2_LPS_Length_Int') > pl.col('100thPercentile'))
              .then(pl.lit(1))
              .otherwise(pl.lit(0))
              .cast(pl.Int8)
              .alias('allele2_count_100'),
        ])
        .select(['SampleID', 'TRID', 'Allele2_LPS_Motif', 'allele2_count_99', 'allele2_count_999', 'allele2_count_100'])
        .collect(engine="streaming")
    )

    # ========================================================================
    # ALLELE 2 - PASS 2: Anti-join for novel combinations
    # ========================================================================
    allele2_novel = (
        pl.scan_csv(
            file_path,
            separator='\t',
            has_header=True,
            infer_schema_length=0
        )
        # Anti-join - only unmatched TRID+Motif combinations
        .join(
            lenStats_pl.lazy(),
            left_on=['TRID', 'Allele2_LPS_Motif'],
            right_on=['TRID', 'LPS_Motif'],
            how='anti'
        )
        # Convert allele length
        .with_columns([
            pl.when(pl.col('Allele2_LPS_Length') == '.')
              .then(pl.lit(0))
              .otherwise(
                  pl.col('Allele2_LPS_Length')
                    .str.to_integer(strict=False)
                    .fill_null(0)
              )
              .alias('Allele2_LPS_Length_Int'),
        ])
        # Novel motif gets 1 point if length > 0
        .with_columns([
            pl.when(pl.col('Allele2_LPS_Length_Int') > 0)
              .then(pl.lit(1))
              .otherwise(pl.lit(0))
              .cast(pl.Int8)
              .alias('allele2_count_99'),

            pl.when(pl.col('Allele2_LPS_Length_Int') > 0)
              .then(pl.lit(1))
              .otherwise(pl.lit(0))
              .cast(pl.Int8)
              .alias('allele2_count_999'),

            pl.when(pl.col('Allele2_LPS_Length_Int') > 0)
              .then(pl.lit(1))
              .otherwise(pl.lit(0))
              .cast(pl.Int8)
              .alias('allele2_count_100'),
        ])
        .select(['SampleID', 'allele2_count_99', 'allele2_count_999', 'allele2_count_100'])
        .collect(engine="streaming")
    )

    # Combine Allele2 matched and novel
    allele2_result = pl.concat([
        allele2_matched.select(['SampleID', 'allele2_count_99', 'allele2_count_999', 'allele2_count_100']),
        allele2_novel
    ])

    # ========================================================================
    # Combine both alleles and aggregate by participant
    # ========================================================================
    result = (
        allele1_result
        .join(allele2_result, on='SampleID', how='outer')
        .with_columns([
            (pl.col('allele1_count_99').fill_null(0) + pl.col('allele2_count_99').fill_null(0)).alias('99thPercentile'),
            (pl.col('allele1_count_999').fill_null(0) + pl.col('allele2_count_999').fill_null(0)).alias('99.9thPercentile'),
            (pl.col('allele1_count_100').fill_null(0) + pl.col('allele2_count_100').fill_null(0)).alias('100thPercentile'),
        ])
        .group_by('SampleID')
        .agg([
            pl.col('99thPercentile').sum(),
            pl.col('99.9thPercentile').sum(),
            pl.col('100thPercentile').sum(),
        ])
    )

    print(f"  Chr {chrom}: Found outliers for {len(result)} participants")
    all_results.append(result)


# ============================================================================
# STEP 4: Combine results and create final DataFrame
# ============================================================================
print("\nCombining results across chromosomes...")

# Concatenate and aggregate
final_counts = (
    pl.concat(all_results)
    .group_by('SampleID')
    .agg([
        pl.col('99thPercentile').sum(),
        pl.col('99.9thPercentile').sum(),
        pl.col('100thPercentile').sum(),
    ])
)

# Convert to pandas for compatibility with your existing code
print("\nConverting to pandas DataFrame...")
outlierCounts = final_counts.to_pandas().set_index('SampleID')

print(f"\nCounts before reindex - non-zero entries: {(outlierCounts > 0).any(axis=1).sum()}")

# Reindex to include all participants (fill missing with 0)
outlierCounts = outlierCounts.reindex(participant_list, fill_value=0)
outlierCounts.index.name = 'participant'

print(f"\nCounts after reindex - non-zero entries: {(outlierCounts > 0).any(axis=1).sum()}")

# Ensure integer types
outlierCounts = outlierCounts.astype('int64')

print(f"\nComplete! Processed data for {len(outlierCounts)} participants")
print(f"Memory usage: {outlierCounts.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\nSummary statistics:")
print(outlierCounts.describe())

print(f"\nTotal outliers identified:")
print(f"  99th percentile: {outlierCounts['99thPercentile'].sum():,}")
print(f"  99.9th percentile: {outlierCounts['99.9thPercentile'].sum():,}")
print(f"  100th percentile: {outlierCounts['100thPercentile'].sum():,}")

outlierCounts

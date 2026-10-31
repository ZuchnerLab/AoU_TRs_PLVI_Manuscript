import polars as pl
import sys, getopt, re

def gc_content(seq):
    if seq is None:
        return None
    seq = seq.upper()
    if seq == '' or seq == '.':
        return None
    return (seq.count('G') + seq.count('C')) / len(seq)

def repeat_purity(seq, motif):
    if seq is None or motif is None:
        return None
    seq = seq.upper()
    motif = motif.upper()
    if seq == '' or seq == '.' or motif == '' or motif == '.' or motif == 'NONE':
        return None
    n = len(seq)
    L = len(motif)
    if L > n:
        return 0.0
    # re.findall on a literal pattern already scans left-to-right and skips
    # past each match, giving the same disjoint matches as a manual greedy scan
    matched = len(re.findall(re.escape(motif), seq)) * L
    return matched / n

def main ( argv ):
    CHROM=""
    CatalogName=""
    try:
        opts, args = getopt.getopt(argv,"h",["chrom=","catalog_name=","help"])
    except getopt.GetoptError:
        print('calculateAlleleSeqStats.py --chrom=<Chromosome> --catalog_name=<CatalogName>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('--chrom'):
            CHROM=arg
        elif opt in ('--catalog_name'):
            CatalogName=arg
        elif opt in ('h','--help'):
            print('calculateAlleleSeqStats.py --chrom=<Chromosome> --catalog_name=<CatalogName>')
            sys.exit()
        else:
            print('calculateAlleleSeqStats.py --chrom=<Chromosome> --catalog_name=<CatalogName>')
            sys.exit()

    # Load metadata
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

    # calculate GC content and repeat purity for each allele
    alleleSeqStats = alleleInfo.select([
        'SampleID', 'TRID',
        pl.col('Allele1').map_elements(gc_content, return_dtype=pl.Float64).alias('allele1GCContent'),
        pl.col('Allele2').map_elements(gc_content, return_dtype=pl.Float64).alias('allele2GCContent'),
        pl.struct(['Allele1', 'Allele1_LPS_Motif']).map_elements(
            lambda x: repeat_purity(x['Allele1'], x['Allele1_LPS_Motif']), return_dtype=pl.Float64
        ).alias('allele1RepeatPurity'),
        pl.struct(['Allele2', 'Allele2_LPS_Motif']).map_elements(
            lambda x: repeat_purity(x['Allele2'], x['Allele2_LPS_Motif']), return_dtype=pl.Float64
        ).alias('allele2RepeatPurity'),
    ])
    alleleSeqStats.write_csv(f'GCContentAndRepeatPuritySeqStats_chr{CHROM}_trgt.{CatalogName}.txt', separator='\t')

    # melt alleles 1 and 2 onto separate rows, then average per TRID
    allele1Long = alleleSeqStats.select(['TRID', pl.col('allele1GCContent').alias('GCContent'), pl.col('allele1RepeatPurity').alias('RepeatPurity')])
    allele2Long = alleleSeqStats.select(['TRID', pl.col('allele2GCContent').alias('GCContent'), pl.col('allele2RepeatPurity').alias('RepeatPurity')])
    alleleSeqStatsLong = pl.concat([allele1Long, allele2Long])

    alleleSeqStatsPerLocus = alleleSeqStatsLong.group_by('TRID').agg([
        pl.mean('GCContent').alias('meanGCContent'),
        pl.mean('RepeatPurity').alias('meanRepeatPurity'),
        pl.median('GCContent').alias('medianGCContent'),
        pl.median('RepeatPurity').alias('medianRepeatPurity')
    ])
    alleleSeqStatsPerLocus.write_csv(f'GCContentAndRepeatPurityStatsPerLocus_chr{CHROM}_trgt.{CatalogName}.txt', separator='\t')

if __name__ == "__main__":
    main(sys.argv[1:])

import sys, getopt, os
import pandas
import regex as re

# Pre-compile regex patterns that are constant for performance.
# This pattern finds k-mers with at least 3 consecutive matches.
THREE_CONSECUTIVE_KMER_RE = re.compile(r"\b(\w+)\s+\1\b\s\1\b")

def _find_top_lps_for_allele(allele, refMotifs):
    """Helper function to find the Longest Perfect Sequence (LPS) for a single allele."""
    if not allele or allele == '.' or allele == '':
        return -1, 'None'

    spanCovered = 0
    topLPSLength = 0
    topLPSMotif = ''

    # 1. Check reference motifs first
    for k in refMotifs:
        # re.escape(k) is important in case a motif contains special regex characters
        matches = re.findall('(?:' + re.escape(k) + ')+', allele)
        if not matches:
            continue

        # Find the longest perfect repeat of this motif
        current_max_match = max(matches, key=len)
        if len(current_max_match) > 0.5 * len(allele):
            # If any one motif covers >50% of the allele, it's the dominant LPS. We're done.
            return len(current_max_match), k

        spanCovered += sum(len(x) for x in matches)
        if len(current_max_match) > topLPSLength:
            topLPSLength = len(current_max_match)
            topLPSMotif = k

    # 2. If no dominant LPS and there's enough space left, search for novel motifs
    if (len(allele) - spanCovered) > topLPSLength:
        # Set max k-mer length to search for; capped at 20bp or 1/3 of allele length
        maxKmerLen = min(20, len(allele) // 3)

        for k in range(1, maxKmerLen + 1):
            # Find k-mers that repeat at least 3 times consecutively
            words_str = ' '.join(re.findall('.' * k, allele))
            novel_motifs = THREE_CONSECUTIVE_KMER_RE.findall(words_str)

            for motif in set(novel_motifs):
                # Use re.escape on the discovered motif to treat it as a literal string
                matches = re.findall('(?:' + re.escape(motif) + ')+', allele)
                if not matches:
                    continue

                current_max_match = max(matches, key=len)
                if len(current_max_match) > 0.5 * len(allele):
                    # Found a dominant novel LPS, we are done.
                    return len(current_max_match), motif

                if len(current_max_match) > topLPSLength:
                    topLPSLength = len(current_max_match)
                    topLPSMotif = motif

    return topLPSLength, topLPSMotif

def calculateLPS(row):
    """Calculates the Longest Perfect Sequence (LPS) for alleles in a row."""
    allele1 = row.Allele1
    allele2 = row.Allele2
    motifs = row.MOTIFS.split(',')
    alleles = [allele1, allele2]
    if allele2 == '.':
        alleles = [allele1]
    lps_results = []
    if len(alleles) == 2:
        if alleles[0] == alleles[1]:
            # If both alleles are the same, just run LPS once and apply result to both
            lps_length, lps_motif = _find_top_lps_for_allele(alleles[0], motifs)
            return pandas.Series([lps_length, lps_motif, lps_length, lps_motif],
                                 index=['Allele1_LPS_Length', 'Allele1_LPS_Motif', 'Allele2_LPS_Length', 'Allele2_LPS_Motif'])
    for allele in alleles:
        lps_length, lps_motif = _find_top_lps_for_allele(allele, motifs)
        lps_results.append((lps_length, lps_motif))

    if len(lps_results) >= 2:
        return pandas.Series([lps_results[0][0], lps_results[0][1], lps_results[1][0], lps_results[1][1]],
                             index=['Allele1_LPS_Length', 'Allele1_LPS_Motif', 'Allele2_LPS_Length', 'Allele2_LPS_Motif'])
    elif len(lps_results) == 1:
        return pandas.Series([lps_results[0][0], lps_results[0][1], -1, "None"],
                             index=['Allele1_LPS_Length', 'Allele1_LPS_Motif', 'Allele2_LPS_Length', 'Allele2_LPS_Motif'])
    else:
        # Return a default empty series in case of no alleles
        return pandas.Series([-1, "None", -1, "None"],
                             index=['Allele1_LPS_Length', 'Allele1_LPS_Motif', 'Allele2_LPS_Length', 'Allele2_LPS_Motif'])

def main ( argv ):
    sampleID=""
    try:
        opts, args = getopt.getopt(argv,"h",["sample=","help"])
    except getopt.GetoptError:
        print('calculateLPS.py --sample=<SampleID>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('--sample'):
            sampleID=arg
        elif opt in ('h','--help'):
            print('calculateLPS.py --sample=<SampleID>')
            sys.exit()
        else:
            print('calculateLPS.py --sample=<SampleID>')
            sys.exit()

    # load TRGT output file
    sample=pandas.read_csv(str(sampleID) + '.txt',sep='\t',low_memory=True,header=None,
                        names=['SampleID','CenterName','TRID','MOTIFS','Alleles','Allele1_Len','Allele2_Len','Allele1_DP','Allele2_DP','Allele1_Methylation','Allele2_Methylation'],
                        dtype={'SampleID':int,'CenterName':str,'TRID':str,'MOTIFS':str,'Alleles':str,'Allele1_Len':str,'Allele2_Len':str,
                                'Allele1_DP':str,'Allele2_DP':str,'Allele1_Methylation':str,'Allele2_Methylation':str})

    # separate out alleles
    sample['Alleles']=sample['Alleles'].str.replace('|','/',regex=False)
    sample.loc[~(sample['Alleles'].str.contains('/')),'Alleles']=sample.loc[~(sample['Alleles'].str.contains('/')),'Alleles'] + '/.'
    sample[['Allele1','Allele2']]=sample['Alleles'].str.split('/',expand=True)

    # calculate LPS for each allele
    sample[['Allele1_LPS_Length','Allele1_LPS_Motif','Allele2_LPS_Length','Allele2_LPS_Motif']]=sample.apply(calculateLPS,axis=1)

    # save results to file 
    sample.loc[:,['SampleID','CenterName','TRID','Allele1','Allele2','Allele1_LPS_Length','Allele1_LPS_Motif',
                'Allele2_LPS_Length','Allele2_LPS_Motif','Allele1_Len','Allele2_Len',
                'Allele1_DP','Allele2_DP','Allele1_Methylation','Allele2_Methylation'
                ]].to_csv(str(sampleID) + '_trgt.TR_Explorer_1.0.1_LPS.txt', sep='\t', index=False, header=False)
    return

if __name__ == "__main__":
    main(sys.argv[1:])

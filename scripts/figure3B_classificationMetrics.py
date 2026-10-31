"""
Repeat the Figure 3B comparisons from Figure3.ipynb but score them with
sensitivity, specificity, F1, and area under the ROC curve instead of
generalized odds ratios. Thresholds required for sens/spec/F1 are drawn from
the 90th, 95th, 99th, and 99.9th percentiles of each score's distribution.

Score direction handling: allele-length and LPS-length statistics assume
higher = more pathogenic (matches the mannwhitneyu alternative='greater'
setup in Figure 3B). Distance-to-feature scores assume lower = more
pathogenic (matches the flipped group order used for those columns in
Figure 3B). AUROC uses the raw score with the appropriate sign so its
value is directly comparable across features.
"""

# Report sensitivity and specificity of PLVI for the disease-associated STRs. Do the same for each genomic feature and RExPRT. 
import os
import pickle

import numpy as np
import pandas as pd
import scipy.stats


BASE_DIR = os.path.dirname('.')
STATS_DIR = os.path.join(BASE_DIR, 'statsTables')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
ENRICH_DIR = os.path.join(ASSETS_DIR, 'pathoEnrichment')
PERCENTILE_THRESHOLDS = [90, 95, 99, 99.9]

def roc_auc_score(y_true, y_score):
    """AUROC via the Mann-Whitney U identity: U / (n_pos * n_neg).

    Ties contribute 0.5. Matches sklearn.metrics.roc_auc_score for
    binary labels.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return float('nan')
    u, _ = scipy.stats.mannwhitneyu(pos, neg, alternative='greater')
    return u / (pos.size * neg.size)


def confusion_counts(y_true, y_pred):
    """Return (tn, fp, fn, tp) for binary 0/1 arrays."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tn, fp, fn, tp


def f1_from_counts(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def evaluate_score(scores, labels, direction, fixed_thresholds=None):
    """Return AUROC plus per-threshold sensitivity / specificity / F1.

    scores           -- 1-D numeric array of the score being evaluated
    labels           -- 1-D binary array (1 = pathogenic, 0 = other)
    direction        -- 'higher' if larger scores predict pathogenic,
                        'lower'  if smaller scores predict pathogenic
    fixed_thresholds -- optional dict mapping tag -> raw threshold value.
                        Scores at/beyond the threshold (in the pathogenic
                        direction) are called positive.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    mask = ~np.isnan(scores)
    s = scores[mask]
    y = labels[mask]

    signed = s if direction == 'higher' else -s
    auroc = roc_auc_score(y, signed)

    metrics = {
        'n_total': int(len(y)),
        'n_pathogenic': int(y.sum()),
        'direction': direction,
        'auroc': auroc,
    }

    def _record(tag, thresh):
        if direction == 'higher':
            preds = (s >= thresh).astype(int)
        else:
            preds = (s <= thresh).astype(int)

        tn, fp, fn, tp = confusion_counts(y, preds)
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        f1 = f1_from_counts(tp, fp, fn)

        metrics[f'threshold_{tag}'] = thresh
        metrics[f'sensitivity_{tag}'] = sens
        metrics[f'specificity_{tag}'] = spec
        metrics[f'f1_{tag}'] = f1
        metrics[f'tp_{tag}'] = int(tp)
        metrics[f'fp_{tag}'] = int(fp)
        metrics[f'fn_{tag}'] = int(fn)
        metrics[f'tn_{tag}'] = int(tn)

    for p in PERCENTILE_THRESHOLDS:
        if direction == 'higher':
            thresh = np.percentile(s, p)
        else:
            thresh = np.percentile(s, 100 - p)
        _record(f'p{p}'.replace('.', '_'), thresh)

    if fixed_thresholds:
        for tag, thresh in fixed_thresholds.items():
            _record(tag, thresh)

    return metrics


def complement(seq):
    comp = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join(comp[b] for b in seq)


def reverse_complement(seq):
    return complement(seq[::-1])


def simplest_motif(motif):
    i = (motif + motif).find(motif, 1, -1)
    if i != -1:
        motif = motif[:i]
    motifs = [(motif[i:] + motif[:i]).upper() for i in range(len(motif))]
    motif_rc = reverse_complement(motif.upper())
    i = (motif_rc + motif_rc).find(motif_rc, 1, -1)
    if i != -1:
        motif_rc = motif_rc[:i]
    motifs += [motif_rc[i:] + motif_rc[:i] for i in range(len(motif_rc))]
    return min(motifs)


def load_pathogenic_ids():
    pathogenic = pd.read_csv(
        os.path.join(ASSETS_DIR, 'pathogenicTRIDs_TRExplorer_v1.0.1_STRchiveFiltered_noVNTRs_withAnno.txt'),
        sep='\t',
    )
    poly_alanine_names = ['ARX', 'NIPA1', 'HOXA13', 'PHOX2B', 'SOX3',
                          'HOXD13', 'ZIC2', 'RUNX2', 'PABPN1', 'FOXL2']
    single_shift_names = ['MIR7-2', 'COMP']
    excluded = set(poly_alanine_names) | set(single_shift_names)
    other_pathogenic = pathogenic.loc[~pathogenic['geneName'].isin(excluded), 'TRID'].values
    return other_pathogenic


def run_allele_length_block(other_pathogenic_ids, results):
    total_len = pd.read_csv(
        os.path.join(STATS_DIR, 'DiscoveryCohort_alleleLengthStats.TR_Explorer_1.0.1.txt.gz'),
        sep='\t',
    )
    total_len['patho'] = total_len['TRID'].isin(other_pathogenic_ids).astype(int)

    columns = ['std', '50thPercentile', '99thPercentile', '99.9thPercentile',
               '100thPercentile', 'mean', 'mad', 'mode',
               'uniqueLengths', 'uniqueAlleleSeqs']
    for column in columns:
        sub = total_len[[column, 'patho']].dropna(subset=[column])
        metrics = evaluate_score(sub[column].values, sub['patho'].values, direction='higher')
        metrics['condition'] = f'Allele Length {column}'
        results.append(metrics)

    total_len[['chr', 'start', 'end', 'refMotif']] = (
        total_len['TRID'].str.split(',', expand=True).iloc[:, 0].str.split('-', expand=True)
    )
    total_len['motifLen'] = total_len['refMotif'].str.len()
    total_len['percentile_stdev_ref_norm'] = (
        total_len.groupby('motifLen', observed=True)['std'].rank(pct=True)
    )
    sub = total_len[['percentile_stdev_ref_norm', 'patho']].dropna(subset=['percentile_stdev_ref_norm'])
    metrics = evaluate_score(sub['percentile_stdev_ref_norm'].values, sub['patho'].values, direction='higher')
    metrics['condition'] = 'Allele Length (Ref Motif Normalized)'
    results.append(metrics)

    return total_len


def run_lps_block(other_pathogenic_ids, results):
    lps = pd.read_csv(
        os.path.join(STATS_DIR, 'DiscoveryCohort_lpsPerMotifStats.TR_Explorer_1.0.1.txt.gz'),
        sep='\t',
    )
    lps = (lps.sort_values(by=['TRID', 'N_motif'], ascending=False)
              .drop_duplicates(subset='TRID', keep='first')
              .reset_index(drop=True))
    lps['simpleMotif'] = lps['LPS_Motif'].apply(simplest_motif)
    lps['motifLen'] = lps['simpleMotif'].str.len()
    lps['percentile_stdev'] = lps['std'].rank(pct=True)
    lps['patho'] = lps['TRID'].isin(other_pathogenic_ids).astype(int)

    for column in ['percentile_stdev', 'mean', 'mad', 'mode']:
        sub = lps[[column, 'patho']].dropna(subset=[column])
        metrics = evaluate_score(sub[column].values, sub['patho'].values, direction='higher')
        metrics['condition'] = f'LPS Length {column}'
        results.append(metrics)

    lps['percentile_stdev_motif_norm'] = (
        lps.groupby('motifLen', observed=True)['std'].rank(pct=True)
    )
    sub = lps[['percentile_stdev_motif_norm', 'patho']].dropna(subset=['percentile_stdev_motif_norm'])
    metrics = evaluate_score(sub['percentile_stdev_motif_norm'].values, sub['patho'].values, direction='higher')
    metrics['condition'] = 'LPS Length (LPS Motif Normalized)'
    results.append(metrics)

    # add on the LPS per locus stats
def run_rexprt_block(other_pathogenic_ids, results):
    rexprt=pd.read_csv(os.path.join(ENRICH_DIR, 'TR_Explorer_v1.0.1.hg19.RExPRTscores.txt.gz'),sep='\t')
    rexprt=rexprt.rename(columns={'sampleID':'TRID'})
    rexprt['patho']=rexprt['TRID'].isin(other_pathogenic_ids).astype(int)
    for column in ['ensembleScore','ensembleMax']:
        sub = rexprt[[column, 'patho']].dropna(subset=[column])
        metrics = evaluate_score(sub[column].values, sub['patho'].values,
                                 direction='higher',
                                 fixed_thresholds={'binary': 0.5})
        metrics['condition'] = f'RExPRT {column}'
        results.append(metrics)


def run_distance_block(total_len, results):
    tad = pd.read_csv(os.path.join(ENRICH_DIR, 'distanceToClosestTADBoundary.txt.gz'),
                      sep='\t', header=None, names=['TRID', 'TADDistance'])
    rad21 = pd.read_csv(os.path.join(ENRICH_DIR, 'distanceToClosestRAD21Site.txt.gz'),
                        sep='\t', header=None, names=['TRID', 'RAD21Distance'])
    smc3 = pd.read_csv(os.path.join(ENRICH_DIR, 'distanceToClosestSMC3Site.txt.gz'),
                       sep='\t', header=None, names=['TRID', 'SMC3Distance'])
    oreganno = pd.read_csv(os.path.join(ENRICH_DIR, 'distanceToClosestORegAnnoFeature.txt.gz'),
                           sep='\t', header=None, names=['TRID', 'ORegAnnoDistance'])
    alu = pd.read_csv(os.path.join(ENRICH_DIR, 'distanceToClosestAlu.txt.gz'),
                      sep='\t', header=None, names=['TRID', 'AluDistance'])
    estr = pd.read_csv(os.path.join(ENRICH_DIR, 'distanceToClosestESTR.txt.gz'),
                       sep='\t', header=None, names=['TRID', 'eSTRDistance'])

    combined = (total_len.merge(tad, how='inner', on='TRID')
                          .merge(rad21, how='inner', on='TRID')
                          .merge(smc3, how='inner', on='TRID')
                          .merge(oreganno, how='inner', on='TRID')
                          .merge(alu, how='inner', on='TRID')
                          .merge(estr, how='inner', on='TRID'))

    for column in ['TADDistance', 'RAD21Distance', 'SMC3Distance',
                   'ORegAnnoDistance', 'AluDistance', 'eSTRDistance']:
        sub = combined[[column, 'patho']].dropna(subset=[column])
        metrics = evaluate_score(sub[column].values, sub['patho'].values, direction='lower')
        metrics['condition'] = column
        results.append(metrics)

def main():
    other_pathogenic_ids = load_pathogenic_ids()
    results = []

    total_len = run_allele_length_block(other_pathogenic_ids, results)
    run_lps_block(other_pathogenic_ids, results)
    run_rexprt_block(other_pathogenic_ids, results)
    run_distance_block(total_len, results)

    results_df = pd.DataFrame(results)
    front = ['condition', 'direction', 'n_total', 'n_pathogenic', 'auroc']
    ordered_cols = front + [c for c in results_df.columns if c not in front]
    results_df = results_df[ordered_cols]

    out_csv = os.path.join(ENRICH_DIR, 'results_collection_figure3B_classificationMetrics.csv')
    out_pkl = os.path.join(ENRICH_DIR, 'results_collection_figure3B_classificationMetrics.pkl')
    results_df.to_csv(out_csv, index=False)
    with open(out_pkl, 'wb') as f:
        pickle.dump(results, f)

    print('=' * 80)
    print('CLASSIFICATION METRICS FOR FIGURE 3B COMPARISONS')
    print('=' * 80)
    summary_cols = (['condition', 'direction', 'auroc']
                    + [f'sensitivity_p{str(p).replace(".", "_")}' for p in PERCENTILE_THRESHOLDS]
                    + [f'specificity_p{str(p).replace(".", "_")}' for p in PERCENTILE_THRESHOLDS]
                    + [f'f1_p{str(p).replace(".", "_")}' for p in PERCENTILE_THRESHOLDS])
    with pd.option_context('display.max_columns', None, 'display.width', 240,
                           'display.float_format', lambda x: f'{x:.4f}'):
        print(results_df[summary_cols].to_string(index=False))
    print(f'\nWrote {len(results_df)} rows to:\n  {out_csv}\n  {out_pkl}')


if __name__ == '__main__':
    main()

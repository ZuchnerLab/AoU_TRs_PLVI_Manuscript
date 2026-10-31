#!/usr/bin/env python3
"""
skewnessAnalyses.py
===================

Per-locus (per-TRID) distribution-shape analysis of tandem-repeat calls.

For every TRID we pool both alleles of every individual (each allele is one
observation, i.e. "melt then concat", matching scripts/calculateLPSStats.py) and,
for TWO separate length metrics -- the LPS (Longest Perfect Sequence) length and
the total allele length -- we compute:

  * Hartigan's dip test  (dip statistic + p-value; the formal test of unimodality)
  * skewness             (adjusted Fisher-Pearson)
  * kurtosis             (excess / Fisher)
  * summary stats        (N, unique lengths, mean, std, median, min, max)
  * histograms           (multi-page PDF, for flagged loci + a random QC sample)

Loci are then classified unimodal vs. multimodal from the dip test
(Benjamini-Hochberg FDR across all loci). For the multimodal loci we estimate how
much of the extra structure is explained by:

  (a) alternative LPS-Motif values  -- different pure repeat units at the locus, and
  (b) genetic ancestry              -- different modal lengths across populations,

using, per locus and per factor, (i) eta^2 = fraction of length variance explained
by the factor, and (ii) a residual dip test: if removing the factor's group means
makes the residual distribution unimodal -- judged with the same significance and
--min-dip effect-size criterion used for the primary modality call -- that factor
"explains" the multimodality.
A two-factor (motif x ancestry) partition and a primary-driver call are also emitted.

INPUT
-----
A gzipped, headered, tab-separated table (default selectLoci_alleleInfo_withAncestry.txt.gz)
with the columns:
    person_id, centerName, TRID,
    allele1_LPS_Length, allele1_LPS_Motif,
    allele2_LPS_Length, allele2_LPS_Motif,
    allele1_Length,     allele2_Length,
    ancestry

Missing / no-call values may appear as '.', 'None', 'NA', '' or -1 in any length
column; these are dropped, and only strictly positive lengths are retained
(matching the `LPS_Length > 0` filter used elsewhere in the project).

OUTPUT (written under --outdir, default ./skewnessResults)
----------------------------------------------------------
    locusStats_LPS.tsv                 per-TRID stats + modality call  (all loci)
    locusStats_alleleLength.tsv
    multimodal_drivers_LPS.tsv         motif/ancestry attribution      (multimodal loci)
    multimodal_drivers_alleleLength.tsv
    histograms_multimodal_LPS.pdf      one page/locus: pooled + by-motif + by-ancestry
    histograms_multimodal_alleleLength.pdf
    histograms_qcSample_LPS.pdf        random QC sample of loci
    histograms_qcSample_alleleLength.pdf
    summary.txt                        run summary + counts

Usage:
    python skewnessAnalyses.py \
        --alleles selectLoci_alleleInfo.txt.gz \
        --outdir  skewnessResults
"""

import argparse
import os
import warnings
from collections import namedtuple

import numpy as np
import pandas as pd
import scipy.stats as ss
import diptest

import matplotlib
matplotlib.use("Agg")               # headless: write files, never open a window
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
BASE_COLUMNS = [
    "person_id", "centerName", "TRID",
    "allele1_LPS_Length", "allele1_LPS_Motif",
    "allele2_LPS_Length", "allele2_LPS_Motif",
    "allele1_Length", "allele2_Length",
]
ANCESTRY_LABELS = {"AFR", "AMR", "EUR", "EAS", "SAS", "MID"}
MISSING_TOKENS = {".", "none", "na", "nan", "null", "", "-1"}

# The two length metrics we analyse independently. (out_tag, melted column name)
METRICS = [("LPS", "LPS_Length"), ("alleleLength", "Allele_Length")]

# Configuration for the dip test. Tandem-repeat lengths are integer base pairs, so
# the raw data is heavily tied; ties inflate Hartigan's dip statistic and make the
# test call almost everything multimodal at large N. We therefore "dither" the data
# (add uniform jitter of +/- `jitter` bp, the standard remedy for applying the dip
# test to discretised data) and report the median dip / p over `reps` jitterings.
DipCfg = namedtuple("DipCfg", "min_n reps jitter seed")


# --------------------------------------------------------------------------- #
# Motif canonicalisation -- verbatim from scripts/calculateLPSStats.py
# (lexicographically-minimal rotation of the smallest repeating unit) so that
# e.g. GGC / GCG / CGG all collapse to the same canonical motif CGG.
# --------------------------------------------------------------------------- #
def simplest_motif(motif):
    i = (motif + motif).find(motif, 1, -1)
    if i != -1:
        motif = motif[:i]
    motifs = []
    b = len(motif)
    for i in range(b):
        c = motif[i:] + motif[:i]
        motifs.append(c)
    motifs = [x.upper() for x in motifs]
    return min(motifs)


def canon_motif(m):
    """Canonicalise a motif string, returning None for missing/invalid motifs."""
    if not isinstance(m, str):
        return None
    m = m.strip().upper()
    if m in ("", "NONE", "NA", ".") or not set(m) <= set("ACGTN"):
        return None
    return simplest_motif(m)


# --------------------------------------------------------------------------- #
# Loading & cleaning
# --------------------------------------------------------------------------- #
def _num_or_nan(series):
    """Coerce a column to float, mapping the project's missing sentinels to NaN."""
    s = series.astype(str).str.strip()
    s = s.mask(s.str.lower().isin(MISSING_TOKENS))
    return pd.to_numeric(s, errors="coerce")


def load_alleles(path):
    """
    Read the allele table
    """
    df = pd.read_csv(path, sep="\t", dtype=str)
    return df


def melt_alleles(df):
    """
    Stack allele1 / allele2 into one long table (one row per allele) carrying the
    LPS length, total allele length, canonical LPS motif, and (person-level) ancestry.
    """
    keep = ["person_id", "centerName", "TRID", "ancestry"]
    a1 = df[keep + ["allele1_LPS_Length", "allele1_Length", "allele1_LPS_Motif"]].rename(
        columns={"allele1_LPS_Length": "LPS_Length",
                 "allele1_Length": "Allele_Length",
                 "allele1_LPS_Motif": "LPS_Motif"})
    a2 = df[keep + ["allele2_LPS_Length", "allele2_Length", "allele2_LPS_Motif"]].rename(
        columns={"allele2_LPS_Length": "LPS_Length",
                 "allele2_Length": "Allele_Length",
                 "allele2_LPS_Motif": "LPS_Motif"})
    long = pd.concat([a1, a2], ignore_index=True)

    long["LPS_Length"] = _num_or_nan(long["LPS_Length"])
    long["Allele_Length"] = _num_or_nan(long["Allele_Length"])
    long["motif_canon"] = long["LPS_Motif"].map(canon_motif)
    return long


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #
def dip_test(x, dc):
    """
    Dithered Hartigan's dip statistic + p-value (median over `dc.reps` jitterings),
    with guards for tiny / constant input. See DipCfg for why dithering is needed.
    """
    n = x.size
    if n < 2 or np.unique(x).size < 2:
        return 0.0, 1.0            # constant -> perfectly unimodal
    if n <= 3 or n < dc.min_n:
        return np.nan, np.nan      # not enough data to call modality reliably
    rng = np.random.default_rng(dc.seed)
    dips = np.empty(dc.reps)
    pvals = np.empty(dc.reps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(dc.reps):
            xd = x + rng.uniform(-dc.jitter, dc.jitter, n) if dc.jitter > 0 else x
            dips[i], pvals[i] = diptest.diptest(xd)
    return float(np.median(dips)), float(np.median(pvals))


def locus_stat_row(trid, x, dc):
    """Summary + shape statistics for one locus (one length array)."""
    x = x[np.isfinite(x)]
    n = x.size
    n_uniq = np.unique(x).size
    dip, dip_p = dip_test(x, dc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skew = ss.skew(x, bias=False) if (n >= 3 and n_uniq >= 2) else np.nan
        kurt = ss.kurtosis(x, fisher=True, bias=False) if (n >= 4 and n_uniq >= 2) else np.nan
    return {
        "TRID": trid,
        "N": n,
        "n_unique": n_uniq,
        "mean": x.mean() if n else np.nan,
        "std": x.std(ddof=1) if n >= 2 else np.nan,
        "median": np.median(x) if n else np.nan,
        "min": x.min() if n else np.nan,
        "max": x.max() if n else np.nan,
        "skewness": skew,
        "kurtosis": kurt,
        "dip": dip,
        "dip_pvalue": dip_p,
    }


def compute_locus_stats(long, value_col, dc):
    """Per-TRID stats for one metric, over all loci."""
    sub = long.loc[long[value_col].notna() & (long[value_col] > 0), ["TRID", value_col]]
    rows = [locus_stat_row(trid, g[value_col].to_numpy(float), dc)
            for trid, g in sub.groupby("TRID", sort=False)]
    return pd.DataFrame(rows)


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR q-values (scipy 1.10 has no false_discovery_control)."""
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    mask = np.isfinite(p)
    pv = p[mask]
    m = pv.size
    if m == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * m / (np.arange(1, m + 1))
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]      # enforce monotonicity
    out = np.empty(m)
    out[order] = np.clip(ranked, 0, 1)
    q[mask] = out
    return q


def classify_modality(stats, alpha, min_n, min_dip):
    """
    Add dip_qvalue + modality label (unimodal / multimodal / insufficient_data).

    A locus is multimodal if its BH-FDR dip q-value < alpha AND its dip statistic is
    at least `min_dip` (an optional effect-size floor: at N~1000 the dip test detects
    even tiny real departures from unimodality, so `min_dip` lets you require a
    minimum bump size; 0 disables it).
    """
    stats = stats.copy()
    stats["dip_qvalue"] = bh_fdr(stats["dip_pvalue"].to_numpy())
    is_multi = (stats["dip_qvalue"] < alpha) & (stats["dip"] >= min_dip)
    label = np.where(
        (stats["N"] < min_n) | stats["dip_pvalue"].isna(), "insufficient_data",
        np.where(is_multi, "multimodal", "unimodal"))
    stats["modality"] = label
    return stats


# --------------------------------------------------------------------------- #
# Driver analysis: motif- and ancestry-attribution of multimodality
# --------------------------------------------------------------------------- #
def collapse_rare(labels, min_group_n, other="__other__"):
    """Fold categories with < min_group_n members into a single 'other' bucket."""
    labels = pd.Series(labels, dtype=object)
    counts = labels.value_counts(dropna=False)
    keep = set(counts[counts >= min_group_n].index)
    collapsed = labels.where(labels.isin(keep), other).to_numpy()
    return collapsed, len(keep)


def eta_sq_and_resid(x, groups):
    """eta^2 (fraction of variance explained by `groups`) and the within-group residuals."""
    grand = x.mean()
    ss_tot = np.sum((x - grand) ** 2)
    gmean = pd.Series(x).groupby(pd.Series(groups)).transform("mean").to_numpy()
    if ss_tot == 0:
        return np.nan, x - gmean
    ss_between = np.sum((gmean - grand) ** 2)
    return ss_between / ss_tot, x - gmean


def driver_row(trid, x, motif, anc, alpha, dc, min_group_n, min_dip):
    """
    Attribute a multimodal locus's structure to LPS-Motif and/or ancestry.

    For each factor: eta^2 (extent of length variance explained) and a residual dip
    test -- if de-meaning by the factor's groups makes the residuals unimodal, that
    factor "removes" (explains) the multimodality. "Unimodal" here uses the SAME
    criterion as classify_modality(): the residual counts as still-multimodal only if
    its dip p-value is < alpha AND its dip statistic is at least `min_dip`, so a
    statistically-significant-but-trivially-small residual bump does not stop a factor
    from being credited (the modality call and the driver call stay consistent).
    """
    motif_c, n_motif = collapse_rare(motif, min_group_n)
    anc_c, n_anc = collapse_rare(anc, min_group_n)
    both = np.char.add(np.char.add(motif_c.astype(str), "|"), anc_c.astype(str))

    eta_m, res_m = eta_sq_and_resid(x, motif_c)
    eta_a, res_a = eta_sq_and_resid(x, anc_c)
    eta_b, res_b = eta_sq_and_resid(x, both)

    d_m, p_m = dip_test(res_m, dc)
    d_a, p_a = dip_test(res_a, dc)
    d_b, p_b = dip_test(res_b, dc)

    def _removes(dip, p):
        """True if the residual is no longer multimodal by the main analysis's rule."""
        if not np.isfinite(p):
            return False                       # could not test -> no credit
        return (p >= alpha) or not (np.isfinite(dip) and dip >= min_dip)

    rm = _removes(d_m, p_m)
    ra = _removes(d_a, p_a)
    rb = _removes(d_b, p_b)

    if rm and not ra:
        driver = "motif"
    elif ra and not rm:
        driver = "ancestry"
    elif rm and ra:
        driver = "motif_and_ancestry"       # each factor alone removes it (often correlated)
    elif rb:
        driver = "motif+ancestry_joint"      # only the combination removes it
    else:
        driver = "unexplained"

    # Quantitative primary driver = factor explaining the most variance.
    primary_var = "motif" if (np.nan_to_num(eta_m) >= np.nan_to_num(eta_a)) else "ancestry"

    return {
        "TRID": trid,
        "N": x.size,
        "n_distinct_motifs": pd.Series(motif).nunique(),
        "n_common_motifs": n_motif,
        "n_common_ancestries": n_anc,
        "eta2_motif": eta_m,
        "resid_dip_motif": d_m,
        "resid_dip_p_motif": p_m,
        "motif_removes_multimodality": rm,
        "eta2_ancestry": eta_a,
        "resid_dip_ancestry": d_a,
        "resid_dip_p_ancestry": p_a,
        "ancestry_removes_multimodality": ra,
        "eta2_motif_and_ancestry": eta_b,
        "resid_dip_both": d_b,
        "resid_dip_p_both": p_b,
        "both_remove_multimodality": rb,
        "driver": driver,
        "primary_driver_by_variance": primary_var,
    }


def compute_drivers(long, value_col, multimodal_trids, alpha, dc, min_group_n, min_dip):
    """Run driver_row over every multimodal TRID for one metric."""
    sub = long.loc[
        long[value_col].notna() & (long[value_col] > 0)
        & long["motif_canon"].notna() & long["ancestry"].notna()
        & long["TRID"].isin(multimodal_trids),
        ["TRID", value_col, "motif_canon", "ancestry"],
    ]
    rows = []
    for trid, g in sub.groupby("TRID", sort=False):
        if len(g) < dc.min_n:
            continue
        rows.append(driver_row(
            trid, g[value_col].to_numpy(float),
            g["motif_canon"].to_numpy(), g["ancestry"].to_numpy(),
            alpha, dc, min_group_n, min_dip))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Histograms
# --------------------------------------------------------------------------- #
def _bins(x):
    n_uniq = np.unique(x).size
    return int(min(60, max(10, n_uniq)))


def _plot_by_category(ax, x, cats, title, max_cats=6):
    """Stacked histogram split by the most common categories (+ 'other')."""
    s = pd.Series(cats)
    top = list(s.value_counts().head(max_cats).index)
    groups, labels = [], []
    for c in top:
        groups.append(x[s.to_numpy() == c])
        labels.append(str(c))
    other = x[~s.isin(top).to_numpy()]
    if other.size:
        groups.append(other)
        labels.append("other")
    ax.hist(groups, bins=_bins(x), stacked=True, label=labels)
    ax.set_title(title)
    ax.legend(fontsize=6, ncol=2)


def render_multimodal_pdf(pdf_path, drivers, long, value_col, stats_idx,
                          max_pages, sort_by="dip"):
    """One page per multimodal locus: pooled hist + by-motif + by-ancestry."""
    if drivers.empty:
        print(f"  (no multimodal loci for {value_col}; skipping {os.path.basename(pdf_path)})")
        return
    ranked = drivers.merge(stats_idx, on="TRID", how="left") \
                    .sort_values(sort_by, ascending=False)
    n_total = len(ranked)
    if n_total > max_pages:
        print(f"  NOTE: {n_total} multimodal loci for {value_col}; plotting the "
              f"top {max_pages} by {sort_by}. {n_total - max_pages} not plotted "
              f"(all are still in the drivers table).")
        ranked = ranked.head(max_pages)

    long_v = long.loc[long[value_col].notna() & (long[value_col] > 0)]
    grouped = long_v.groupby("TRID", sort=False)      # index once, not per page
    with PdfPages(pdf_path) as pdf:
        for _, r in ranked.iterrows():
            g = grouped.get_group(r["TRID"])
            x = g[value_col].to_numpy(float)
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
            axes[0].hist(x, bins=_bins(x), color="0.4")
            axes[0].set_title("pooled")
            _plot_by_category(axes[1], x, g["motif_canon"].fillna("NA").to_numpy(),
                              "by LPS motif")
            _plot_by_category(axes[2], x, g["ancestry"].fillna("NA").to_numpy(),
                              "by ancestry")
            fig.suptitle(
                f"{r['TRID']}   N={int(r['N'])}   dip_q={_fmt(r.get('dip_qvalue'))}   "
                f"eta2_motif={_fmt(r['eta2_motif'])}  eta2_anc={_fmt(r['eta2_ancestry'])}  "
                f"driver={r['driver']}", fontsize=9)
            for ax in axes:
                ax.set_xlabel(value_col)
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig)
            plt.close(fig)
    print(f"  wrote {pdf_path} ({len(ranked)} pages)")


def render_qc_pdf(pdf_path, stats, long, value_col, n_sample, seed):
    """Random QC sample of loci (pooled histogram only)."""
    usable = stats.loc[stats["modality"] != "insufficient_data", "TRID"].to_numpy()
    if usable.size == 0:
        print(f"  (no usable loci for QC sample of {value_col})")
        return
    rng = np.random.default_rng(seed)
    pick = rng.choice(usable, size=int(min(n_sample, usable.size)), replace=False)
    idx = stats.set_index("TRID")
    long_v = long.loc[long[value_col].notna() & (long[value_col] > 0)]
    grouped = long_v.groupby("TRID", sort=False)
    with PdfPages(pdf_path) as pdf:
        for trid in pick:
            x = grouped.get_group(trid)[value_col].to_numpy(float)
            fig, ax = plt.subplots(figsize=(5, 3.4))
            ax.hist(x, bins=_bins(x), color="0.4")
            row = idx.loc[trid]
            ax.set_title(f"{trid}  [{row['modality']}]\n"
                         f"skew={_fmt(row['skewness'])} kurt={_fmt(row['kurtosis'])} "
                         f"dip_q={_fmt(row['dip_qvalue'])}", fontsize=9)
            ax.set_xlabel(value_col)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"  wrote {pdf_path} ({len(pick)} pages)")


def _fmt(v):
    try:
        return f"{float(v):.3g}"
    except (TypeError, ValueError):
        return "NA"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alleles", default="selectLoci_alleleInfo_withAncestry.txt.gz",
                    help="Input gzipped TSV of allele info (default: %(default)s)")
    ap.add_argument("--outdir", default="skewnessResults",
                    help="Output directory (default: %(default)s)")
    ap.add_argument("--dip-alpha", type=float, default=0.05,
                    help="FDR threshold on the dip-test q-value for calling a locus "
                         "multimodal, and alpha for residual dip tests (default: %(default)s)")
    ap.add_argument("--min-n", type=int, default=20,
                    help="Minimum pooled alleles for a reliable modality call "
                         "(default: %(default)s)")
    ap.add_argument("--min-dip", type=float, default=0.0,
                    help="Effect-size floor on the dip statistic for calling a "
                         "distribution multimodal; applied both to the per-locus "
                         "modality call and to the residual dip tests in the driver "
                         "analysis (0 disables; try ~0.02 to ignore trivially small "
                         "bumps) (default: %(default)s)")
    ap.add_argument("--dip-reps", type=int, default=10,
                    help="Number of dither (jitter) replicates for the dip test; the "
                         "median dip/p is reported (default: %(default)s)")
    ap.add_argument("--jitter", type=float, default=0.5,
                    help="Uniform dither half-width in bp added before the dip test to "
                         "break integer-length ties (default: %(default)s)")
    ap.add_argument("--min-group-n", type=int, default=20,
                    help="Minimum members for a motif/ancestry group to be kept "
                         "separate (rarer groups are pooled into 'other') "
                         "(default: %(default)s)")
    ap.add_argument("--max-hist", type=int, default=500,
                    help="Max multimodal-locus histogram pages per metric, ranked by "
                         "dip statistic (default: %(default)s)")
    ap.add_argument("--qc-sample", type=int, default=50,
                    help="Number of random loci in the QC histogram PDF (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for the QC sample (default: %(default)s)")
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    dc = DipCfg(min_n=args.min_n, reps=args.dip_reps, jitter=args.jitter, seed=args.seed)

    print(f"[1/4] Loading {args.alleles} ...")
    df = load_alleles(args.alleles)
    long = melt_alleles(df)
    n_people = df["person_id"].nunique()
    n_loci = df["TRID"].nunique()
    anc_counts = df.drop_duplicates("person_id")["ancestry"].value_counts(dropna=False)
    print(f"      {n_people} individuals, {n_loci} TRIDs, {len(long)} pooled alleles")
    print(f"      ancestry (per individual): "
          f"{', '.join(f'{k}={v}' for k, v in anc_counts.items())}")

    summary_lines = [
        f"Input: {args.alleles}",
        f"Individuals: {n_people}    TRIDs: {n_loci}    pooled alleles: {len(long)}",
        f"Ancestry counts (per individual): {dict(anc_counts)}",
        f"Params: dip_alpha={args.dip_alpha} min_n={args.min_n} "
        f"min_dip={args.min_dip} min_group_n={args.min_group_n} "
        f"max_hist={args.max_hist}",
        "",
    ]

    for tag, col in METRICS:
        print(f"[2/4] {tag}: per-locus dip / skewness / kurtosis ...")
        stats = compute_locus_stats(long, col, dc)
        stats = classify_modality(stats, args.dip_alpha, args.min_n, args.min_dip)
        stats_path = os.path.join(args.outdir, f"locusStats_{tag}.tsv")
        stats.to_csv(stats_path, sep="\t", index=False)

        counts = stats["modality"].value_counts()
        n_multi = int(counts.get("multimodal", 0))
        print(f"      {stats_path}: "
              f"{int(counts.get('unimodal', 0))} unimodal, {n_multi} multimodal, "
              f"{int(counts.get('insufficient_data', 0))} insufficient")

        print(f"[3/4] {tag}: attributing multimodality to motif / ancestry ...")
        multi = set(stats.loc[stats["modality"] == "multimodal", "TRID"])
        drivers = compute_drivers(long, col, multi, args.dip_alpha,
                                  dc, args.min_group_n, args.min_dip)
        drivers_path = os.path.join(args.outdir, f"multimodal_drivers_{tag}.tsv")
        drivers.to_csv(drivers_path, sep="\t", index=False)

        drv_counts = drivers["driver"].value_counts() if not drivers.empty else pd.Series(dtype=int)
        print(f"      {drivers_path}: " +
              (", ".join(f"{k}={v}" for k, v in drv_counts.items()) if len(drv_counts) else "no loci"))

        print(f"[4/4] {tag}: histograms ...")
        render_multimodal_pdf(
            os.path.join(args.outdir, f"histograms_multimodal_{tag}.pdf"),
            drivers, long, col, stats[["TRID", "dip", "dip_qvalue"]], args.max_hist)
        render_qc_pdf(
            os.path.join(args.outdir, f"histograms_qcSample_{tag}.pdf"),
            stats, long, col, args.qc_sample, args.seed)

        summary_lines += [
            f"== {tag} ({col}) ==",
            f"  unimodal={int(counts.get('unimodal', 0))} "
            f"multimodal={n_multi} insufficient={int(counts.get('insufficient_data', 0))}",
            f"  multimodal drivers: " +
            (", ".join(f"{k}={v}" for k, v in drv_counts.items()) if len(drv_counts) else "none"),
            f"  median eta2_motif={_fmt(drivers['eta2_motif'].median()) if not drivers.empty else 'NA'}, "
            f"median eta2_ancestry={_fmt(drivers['eta2_ancestry'].median()) if not drivers.empty else 'NA'}",
            "",
        ]

    with open(os.path.join(args.outdir, "summary.txt"), "w") as fh:
        fh.write("\n".join(summary_lines))
    print(f"Done. Results in {args.outdir}/")


if __name__ == "__main__":
    main()

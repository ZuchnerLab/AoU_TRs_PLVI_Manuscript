#!/usr/bin/env python3
"""
ridgePlotLocus.py
=================

Ridge plot of the LPS-length (or total-allele-length) distribution at one or more
tandem-repeat loci, split into one ridge per genetic-ancestry group or per LPS motif.

Reads the same allele table as skewnessAnalyses.py (gzipped, headered TSV):
    person_id, centerName, TRID,
    allele1_LPS_Length, allele1_LPS_Motif,
    allele2_LPS_Length, allele2_LPS_Motif,
    allele1_Length,     allele2_Length,
    ancestry
and applies the same cleaning: both alleles of every individual are pooled (one
observation per allele), missing tokens ('.', 'None', 'NA', '', -1) are dropped,
only strictly positive lengths are kept, and LPS motifs are canonicalised to the
lexicographically-minimal rotation of the smallest repeating unit (GGC/GCG/CGG -> CGG).

Small-group masking (optional, OFF by default): pass --min-group-participants N
(e.g. 20, the All of Us dissemination threshold) to pool groups represented by
fewer than N individuals into "Other", and to suppress "Other" entirely if it is
still below N. A footnote on the figure records how many groups were pooled or
suppressed. With the default (0) every group is shown at full resolution.

Examples
--------
  # one locus, LPS length by ancestry -> PNG
  python ridgePlotLocus.py --alleles selectLoci_alleleInfo_withAncestry.txt.gz \
      --trid 4-3074876-3074940-CAG --out HTT_LPS_byAncestry.png

  # every TRID in a list, allele length by LPS motif, one PDF page per locus
  python ridgePlotLocus.py --alleles selectLoci_alleleInfo_withAncestry.txt.gz \
      --trid-file pathogenic42.txt --metric alleleLength --group-by motif \
      --out pathogenic42_alleleLength_byMotif.pdf

  # mark a pathogenic threshold (in the units of the plotted metric, bp)
  python ridgePlotLocus.py --alleles ... --trid 4-3074876-3074940-CAG \
      --vline 120 --vline-label "pathogenic (40 CAG)" --out HTT.png
"""

import argparse
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import scipy.stats as ss

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
MISSING_TOKENS = {".", "none", "na", "nan", "null", "", "-1"}
METRIC_COLS = {"LPS": "LPS_Length", "alleleLength": "Allele_Length"}
METRIC_LABEL = {"LPS": "LPS length (bp)", "alleleLength": "Allele length (bp)"}

# Fixed ancestry order + fixed colour per ancestry, so a group keeps its colour on
# every locus/page regardless of which groups survive filtering.
ANCESTRY_ORDER = ["AFR", "AMR", "EAS", "EUR", "MID", "SAS"]

# Validated categorical palette (fixed slot order; never cycled). Slots 1-8.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER_COLOR = "#a3a29d"        # neutral grey for the pooled "Other" ridge
POOLED_COLOR = "#52514e"       # dark neutral for the "All" ridge

INK = "#0b0b0b"                # primary text
INK_2 = "#52514e"              # secondary text
INK_3 = "#8a8984"              # muted text / reference lines
GRID = "#e4e3df"
SURFACE = "#ffffff"


# --------------------------------------------------------------------------- #
# Motif canonicalisation -- same as skewnessAnalyses.py / calculateLPSStats.py
# --------------------------------------------------------------------------- #
def simplest_motif(motif):
    i = (motif + motif).find(motif, 1, -1)
    if i != -1:
        motif = motif[:i]
    rots = [(motif[i:] + motif[:i]).upper() for i in range(len(motif))]
    return min(rots)


def canon_motif(m):
    if not isinstance(m, str):
        return None
    m = m.strip().upper()
    if m in ("", "NONE", "NA", ".") or not set(m) <= set("ACGTN"):
        return None
    return simplest_motif(m)


def _num_or_nan(series):
    s = series.astype(str).str.strip()
    s = s.mask(s.str.lower().isin(MISSING_TOKENS))
    return pd.to_numeric(s, errors="coerce")


def motif_len_from_trid(trid):
    """TRIDs look like '4-3074876-3074940-CAG'; return the motif length (or None)."""
    last = str(trid).split("-")[-1]
    parts = [p for p in re.split(r"[^ACGTNacgtn]+", last) if p]
    if not parts:
        return None
    return len(simplest_motif(min(parts, key=len).upper()))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_trids(path, trids, chunksize):
    """Stream the (potentially very large) allele table, keeping only the requested TRIDs."""
    want = set(trids)
    keep = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=chunksize):
        sub = chunk[chunk["TRID"].isin(want)]
        if len(sub):
            keep.append(sub)
    if not keep:
        return pd.DataFrame()
    return pd.concat(keep, ignore_index=True)


def melt_alleles(df):
    """One row per allele: LPS length, allele length, canonical LPS motif, ancestry."""
    base = ["person_id", "TRID"] + (["ancestry"] if "ancestry" in df.columns else [])
    out = []
    for k in ("1", "2"):
        a = df[base + [f"allele{k}_LPS_Length", f"allele{k}_Length", f"allele{k}_LPS_Motif"]]
        out.append(a.rename(columns={f"allele{k}_LPS_Length": "LPS_Length",
                                     f"allele{k}_Length": "Allele_Length",
                                     f"allele{k}_LPS_Motif": "LPS_Motif"}))
    long = pd.concat(out, ignore_index=True)
    long["LPS_Length"] = _num_or_nan(long["LPS_Length"])
    long["Allele_Length"] = _num_or_nan(long["Allele_Length"])
    long["motif_canon"] = long["LPS_Motif"].map(canon_motif)
    if "ancestry" not in long.columns:
        long["ancestry"] = np.nan
    anc = long["ancestry"].astype(str).str.strip()
    long["ancestry"] = anc.mask(anc.str.lower().isin(MISSING_TOKENS)).str.upper()
    return long


# --------------------------------------------------------------------------- #
# Grouping (with the participant-count floor)
# --------------------------------------------------------------------------- #
def build_groups(g, value_col, group_by, min_participants, max_motifs):
    """
    Returns (groups, notes) where groups is an ordered list of
    (label, values ndarray, n_alleles, n_participants, colour).
    """
    key = "ancestry" if group_by == "ancestry" else "motif_canon"
    g = g.loc[g[value_col].notna() & (g[value_col] > 0)].copy()
    n_missing_group = int(g[key].isna().sum())
    g = g.loc[g[key].notna()]
    g[key] = g[key].astype(str).str.strip()

    part = g.groupby(key)["person_id"].nunique()
    if group_by == "ancestry":
        order = [a for a in ANCESTRY_ORDER if a in part.index] + \
                sorted(a for a in part.index if a not in ANCESTRY_ORDER)
        colour = {a: PALETTE[i] for i, a in enumerate(ANCESTRY_ORDER)}
    else:
        # motifs: most common first (by alleles); colour by that rank on THIS locus
        order = list(g[key].value_counts().index)
        colour = {}

    big = [k for k in order if part[k] >= min_participants]
    if group_by == "motif":
        big = big[:max_motifs]
        colour = {m: PALETTE[i] for i, m in enumerate(big)}
    small = [k for k in order if k not in big]

    groups, notes = [], []
    for k in big:
        x = g.loc[g[key] == k, value_col].to_numpy(float)
        groups.append((k, x, x.size, int(part[k]), colour.get(k, OTHER_COLOR)))
    if small:
        sub = g.loc[g[key].isin(small)]
        n_p = sub["person_id"].nunique()
        if n_p >= min_participants:
            x = sub[value_col].to_numpy(float)
            groups.append((f"Other ({len(small)})", x, x.size, int(n_p), OTHER_COLOR))
            notes.append(f"{len(small)} group(s) pooled into Other")
        else:
            notes.append(f"{len(small)} group(s) with <{min_participants} participants suppressed")
    if n_missing_group:
        notes.append(f"{n_missing_group} alleles without {'an ancestry' if key == 'ancestry' else 'a motif'} label omitted")
    return groups, g, notes


# --------------------------------------------------------------------------- #
# Density estimation
# --------------------------------------------------------------------------- #
def hist_curve(x, edges):
    counts, _ = np.histogram(x, bins=edges)
    return counts.astype(float)


def kde_curve(x, grid, bw):
    if x.size < 2 or np.std(x) == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k = ss.gaussian_kde(x, bw_method=bw / np.std(x, ddof=1))
    return k(grid)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def ridge_figure(trid, groups, pooled, metric, group_by, style, binwidth, scale,
                 overlap, xlim, vlines, vline_labels, notes):
    rows = ([("All", pooled[0], pooled[0].size, pooled[1], POOLED_COLOR)] if pooled else []) + groups
    n = len(rows)
    fig_h = 1.35 + 0.62 * n
    fig, ax = plt.subplots(figsize=(8.2, fig_h))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    allx = np.concatenate([r[1] for r in rows])
    if xlim:
        lo, hi = xlim
    else:
        lo, hi = allx.min(), allx.max()
        for v in (vlines or []):           # keep reference lines on the axis
            lo, hi = min(lo, v), max(hi, v)
    # bins centred on the observed length grid (lengths step by motif length)
    start = allx.min() - binwidth / 2.0
    edges = np.arange(start, allx.max() + binwidth, binwidth)
    grid = np.linspace(lo - 2 * binwidth, hi + 2 * binwidth, 800)

    curves = []
    for _, x, *_ in rows:
        if style == "kde":
            y = kde_curve(x, grid, binwidth)
            if y is None:                      # constant group -> fall back to a bar
                curves.append(("hist", edges, hist_curve(x, edges)))
            else:
                curves.append(("kde", grid, y))
        else:
            curves.append(("hist", edges, hist_curve(x, edges)))

    # scaling: 'peak' -> each ridge to its own max; 'density' -> area 1, shared max
    if scale == "density":
        dens = []
        for kind, xs, y in curves:
            area = y.sum() * (binwidth if kind == "hist" else (xs[1] - xs[0]))
            dens.append(y / area if area > 0 else y)
        gmax = max(d.max() for d in dens) or 1.0
        heights = [d / gmax for d in dens]
    else:
        heights = [y / (y.max() or 1.0) for _, _, y in curves]

    for i, ((label, x, n_all, n_part, col), (kind, xs, _), h) in enumerate(zip(rows, curves, heights)):
        base = (n - 1 - i)
        z = 10 + i * 3                          # lower ridges drawn in front
        yy = base + overlap * h
        if kind == "hist":
            xs_step = np.append(xs[:-1], xs[-1])
            ys_step = np.append(yy, yy[-1])
            ax.fill_between(xs_step, base, ys_step, step="post", color=col,
                            alpha=0.9, lw=0, zorder=z)
            ax.step(xs_step, ys_step, where="post", color=SURFACE, lw=1.6, zorder=z + 1)
        else:
            ax.fill_between(xs, base, yy, color=col, alpha=0.9, lw=0, zorder=z)
            ax.plot(xs, yy, color=SURFACE, lw=1.6, zorder=z + 1)
        ax.plot([lo - 2 * binwidth, hi + 2 * binwidth], [base, base], color=INK_3, lw=0.6, zorder=z + 1.5)
        # median marker
        med = float(np.median(x))
        ax.plot([med, med], [base, base + 0.28 * overlap], color=INK, lw=1.4,
                solid_capstyle="butt", zorder=z + 2)
        # labels in ink, never series colour
        ax.text(-0.012, base + 0.12, label, transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=10, color=INK, fontweight="bold")
        ax.text(-0.012, base + 0.12, f"n={n_part:,} people, {n_all:,} alleles",
                transform=ax.get_yaxis_transform(), ha="right", va="top",
                fontsize=7.5, color=INK_2)
        ax.text(1.012, base + 0.12, f"median {med:g}", transform=ax.get_yaxis_transform(),
                ha="left", va="bottom", fontsize=7.5, color=INK_2)

    for j, v in enumerate(vlines or []):
        ax.axvline(v, color=INK_3, lw=1.1, ls=(0, (4, 3)), zorder=100)
        lab = vline_labels[j] if vline_labels and j < len(vline_labels) else f"{v:g}"
        ax.text(v, n - 1 + overlap + 0.15, lab, ha="center", va="bottom",
                fontsize=8, color=INK_2)

    ax.set_xlim(lo - binwidth, hi + binwidth)
    ax.set_ylim(-0.25, n - 1 + overlap + 0.6)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK_3)
    ax.tick_params(axis="x", colors=INK_2, labelsize=9)
    ax.xaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel(METRIC_LABEL[metric], color=INK_2, fontsize=10)

    by = "genetic ancestry" if group_by == "ancestry" else "LPS motif"
    fig.suptitle(f"{trid}", x=0.02, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    sub = (f"{METRIC_LABEL[metric].split(' (')[0]} by {by}  ·  "
           f"{'histogram, ' + format(binwidth, 'g') + ' bp bins' if style == 'hist' else 'KDE, bw ' + format(binwidth, 'g') + ' bp'}"
           f"  ·  {'each ridge scaled to its own peak' if scale == 'peak' else 'shared density scale'}"
           f"  ·  tick = median")
    fig.text(0.02, 1 - 0.42 / fig_h, sub, ha="left", va="top", fontsize=8.5, color=INK_2)
    if notes:
        fig.text(0.02, 0.012, "Note: " + "; ".join(notes) + ".", ha="left", va="bottom",
                 fontsize=7.5, color=INK_3)
    fig.subplots_adjust(left=0.24, right=0.86, top=1 - 0.75 / fig_h, bottom=0.62 / fig_h + 0.06)
    return fig


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alleles", default="selectLoci_alleleInfo_withAncestry.txt.gz",
                    help="Gzipped TSV of allele info (default: %(default)s)")
    ap.add_argument("--trid", action="append", default=[],
                    help="TRID to plot; repeat the flag for several loci")
    ap.add_argument("--trid-file", help="Text file with one TRID per line")
    ap.add_argument("--metric", choices=["LPS", "alleleLength"], default="LPS",
                    help="Length metric to plot (default: %(default)s)")
    ap.add_argument("--group-by", choices=["ancestry", "motif"], default="ancestry",
                    help="One ridge per ancestry group or per canonical LPS motif "
                         "(default: %(default)s)")
    ap.add_argument("--style", choices=["hist", "kde"], default="hist",
                    help="Stepped histogram (honest for integer lengths) or Gaussian KDE "
                         "(default: %(default)s)")
    ap.add_argument("--binwidth", type=float, default=None,
                    help="Bin width / KDE bandwidth in bp (default: the motif length "
                         "parsed from the TRID, else 1)")
    ap.add_argument("--max-bins", type=int, default=80,
                    help="When --binwidth is not given, widen the default bin to a whole "
                         "number of motif units so no more than this many bins span the "
                         "data (default: %(default)s)")
    ap.add_argument("--scale", choices=["peak", "density"], default="peak",
                    help="'peak': each ridge scaled to its own maximum (shape comparison); "
                         "'density': area-normalised on a shared scale (default: %(default)s)")
    ap.add_argument("--overlap", type=float, default=1.5,
                    help="Ridge height in units of row spacing; >1 overlaps (default: %(default)s)")
    ap.add_argument("--no-pooled", action="store_true",
                    help="Omit the 'All' ridge of the pooled distribution")
    ap.add_argument("--min-group-participants", type=int, default=0, metavar="N",
                    help="Optional small-group masking: groups with fewer than N "
                         "participants are pooled into 'Other' (and 'Other' is suppressed "
                         "if still < N). Use 20 for the All of Us dissemination rule. "
                         "Default 0 = off, every group shown")
    ap.add_argument("--max-motifs", type=int, default=8,
                    help="Max motif ridges before the rest fold into 'Other' (default: %(default)s)")
    ap.add_argument("--xlim", type=float, nargs=2, metavar=("MIN", "MAX"),
                    help="x-axis range in bp (default: data range)")
    ap.add_argument("--vline", type=float, action="append",
                    help="Draw a dashed reference line at this length (repeatable)")
    ap.add_argument("--vline-label", action="append",
                    help="Label for the matching --vline (repeatable, same order)")
    ap.add_argument("--out", default="ridgePlot.pdf",
                    help="Output file. .pdf -> one page per TRID in a single file; "
                         ".png/.svg -> one file per TRID (TRID appended when >1) "
                         "(default: %(default)s)")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--chunksize", type=int, default=2_000_000,
                    help="Rows per chunk when streaming the allele table")
    args = ap.parse_args(argv)

    trids = list(args.trid)
    if args.trid_file:
        with open(args.trid_file) as fh:
            trids += [ln.strip().split()[0] for ln in fh if ln.strip() and not ln.startswith("#")]
    trids = list(dict.fromkeys(trids))
    if not trids:
        ap.error("give at least one --trid or a --trid-file")

    print(f"Reading {args.alleles} for {len(trids)} TRID(s) ...", file=sys.stderr)
    df = load_trids(args.alleles, trids, args.chunksize)
    if df.empty:
        sys.exit("None of the requested TRIDs were found in the allele table.")
    if args.group_by == "ancestry" and "ancestry" not in df.columns:
        sys.exit("--group-by ancestry needs an 'ancestry' column in the allele table.")
    long = melt_alleles(df)
    value_col = METRIC_COLS[args.metric]

    ext = os.path.splitext(args.out)[1].lower()
    multi_pdf = ext == ".pdf"
    pdf = PdfPages(args.out) if multi_pdf else None
    written = 0
    by_trid = dict(tuple(long.groupby("TRID", sort=False)))
    for trid in trids:
        if trid not in by_trid:
            print(f"  WARNING: {trid} not found; skipped", file=sys.stderr)
            continue
        groups, kept, notes = build_groups(by_trid[trid], value_col, args.group_by,
                                           args.min_group_participants, args.max_motifs)
        if not groups:
            why = (f"no group with >= {args.min_group_participants} participants"
                   if args.min_group_participants > 0 else "no alleles with a usable length/label")
            print(f"  WARNING: {trid} has {why}; skipped", file=sys.stderr)
            continue
        allv = by_trid[trid]
        allv = allv.loc[allv[value_col].notna() & (allv[value_col] > 0)]
        pooled = None if args.no_pooled else (
            allv[value_col].to_numpy(float), int(allv["person_id"].nunique()))
        if args.binwidth:
            bw = args.binwidth
        else:
            # default: one motif unit; widen to whole multiples of it when the range
            # would give more than --max-bins bins
            unit = motif_len_from_trid(trid) or 1.0
            span = float(allv[value_col].max() - allv[value_col].min()) if len(allv) else 0.0
            bw = unit * max(1, int(np.ceil(span / (unit * args.max_bins))))
        fig = ridge_figure(trid, groups, pooled, args.metric, args.group_by, args.style,
                           float(bw), args.scale, args.overlap, args.xlim,
                           args.vline, args.vline_label, notes)
        if multi_pdf:
            pdf.savefig(fig, facecolor=SURFACE)
        else:
            path = args.out if len(trids) == 1 else \
                f"{os.path.splitext(args.out)[0]}_{trid}{ext}"
            fig.savefig(path, dpi=args.dpi, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        written += 1
        print(f"  {trid}: {len(groups)} ridge(s)" + (f"  [{'; '.join(notes)}]" if notes else ""),
              file=sys.stderr)
    if pdf is not None:
        pdf.close()
    print(f"Done: {written} locus plot(s) -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

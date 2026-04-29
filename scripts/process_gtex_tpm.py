"""
Process GTEx gene TPM matrix to compute per-gene, per-tissue statistics.

Usage:
    python process_gtex_tpm.py <tpm_file.gct.gz> <sample_attrs_file> [output_file]

Arguments:
    tpm_file        GTEx_Analysis_*_gene_tpm.gct.gz
    sample_attrs    GTEx sample attributes file (e.g. GTEx_Analysis_*_Annotations_SampleAttributesDS.txt)
                    Required to map sample IDs to tissue names. Available from the same GTEx download page.
    output_file     Output TSV path (default: gtex_gene_tissue_stats.tsv)

Strategy:
    Rather than melting the full matrix (~56K genes x ~17K samples = ~1B rows), this script
    groups sample columns by tissue and computes horizontal statistics row-wise for each tissue.
    This keeps memory usage proportional to (n_genes x n_samples_per_tissue) per step.
"""

import gzip
import sys
import time
import polars as pl
from pathlib import Path
from tqdm import tqdm


def _log(msg: str, t0: float | None = None) -> float:
    """Print a timestamped message and return the current time."""
    elapsed = f"  [{time.time() - t0:.1f}s]" if t0 is not None else ""
    print(f"[{time.strftime('%H:%M:%S')}]{elapsed} {msg}", flush=True)
    return time.time()


def read_gct_header(tpm_file: str) -> list[str]:
    """Read column names from a GCT file (skips the 2-line GCT header).

    Uses gzip directly to read only the 3rd line, avoiding full decompression.
    """
    opener = gzip.open if tpm_file.endswith(".gz") else open
    with opener(tpm_file, "rt") as f:
        f.readline()  # line 1: GCT version (#1.2)
        f.readline()  # line 2: dimensions (n_genes\tn_samples)
        header_line = f.readline()  # line 3: column names
    return header_line.rstrip("\n").split("\t")


def build_tissue_map(sample_attrs_file: str) -> dict[str, str]:
    """
    Load sample attributes and return a dict mapping SAMPID -> SMTSD (tissue name).
    SMTSD is the detailed tissue site description (e.g. 'Brain - Cerebellum').
    """
    attrs = pl.read_csv(
        sample_attrs_file,
        separator="\t",
        columns=["SAMPID", "SMTSD"],
        infer_schema_length=10000,
    )
    return dict(zip(attrs["SAMPID"].to_list(), attrs["SMTSD"].to_list()))


def group_samples_by_tissue(
    sample_cols: list[str], tissue_map: dict[str, str]
) -> dict[str, list[str]]:
    """Group sample column names by their tissue type."""
    tissue_to_samples: dict[str, list[str]] = {}
    unmapped = []
    for sample in sample_cols:
        tissue = tissue_map.get(sample)
        if tissue is None:
            unmapped.append(sample)
            tissue = "unknown"
        tissue_to_samples.setdefault(tissue, []).append(sample)

    if unmapped:
        print(
            f"  Warning: {len(unmapped)} sample(s) not found in attributes file "
            f"(assigned to 'unknown'). First few: {unmapped[:5]}"
        )
    return tissue_to_samples


def compute_stats(
    tpm_file: str,
    sample_attrs_file: str,
    output_file: str = "gtex_gene_tissue_stats.tsv",
) -> pl.DataFrame:

    script_start = time.time()

    # --- 1. Read header -------------------------------------------------------
    t = _log("Reading GCT header...")
    all_cols = read_gct_header(tpm_file)
    gene_id_col = all_cols[0]    # "Name" — Ensembl gene ID (e.g. ENSG00000...)
    gene_name_col = all_cols[1]  # "Description" — gene symbol
    sample_cols = all_cols[2:]
    _log(f"{len(sample_cols):,} samples found in TPM matrix", t)

    # --- 2. Tissue mapping ----------------------------------------------------
    t = _log("Loading sample attributes...")
    tissue_map = build_tissue_map(sample_attrs_file)
    tissue_to_samples = group_samples_by_tissue(sample_cols, tissue_map)
    _log(f"{len(tissue_to_samples)} tissues identified", t)

    # --- 3. Load full TPM matrix (Float32 to reduce memory) -------------------
    t = _log("Reading TPM matrix (this may take a minute)...")
    schema_overrides = {col: pl.Float32 for col in sample_cols}
    df = pl.read_csv(
        tpm_file,
        separator="\t",
        skip_rows=2,
        schema_overrides=schema_overrides,
    )
    _log(f"Matrix loaded: {df.shape[0]:,} genes x {df.shape[1]:,} columns", t)

    # --- 4. Compute per-tissue horizontal statistics --------------------------
    # For each tissue, build a List column across that tissue's sample columns,
    # then use list aggregation functions — avoiding a full melt of the matrix.
    _log(f"Computing per-tissue statistics ({len(tissue_to_samples)} tissues)...")
    result_parts: list[pl.DataFrame] = []

    tissues_sorted = sorted(tissue_to_samples.items())
    with tqdm(tissues_sorted, unit="tissue", ncols=80) as pbar:
        for tissue, samples in pbar:
            pbar.set_description(tissue[:40])

            tissue_df = (
                df.select([gene_id_col, gene_name_col] + samples)
                .with_columns(
                    pl.concat_list(samples).alias("_vals")
                )
                .select(
                    pl.col(gene_id_col).alias("gene_id"),
                    pl.col(gene_name_col).alias("gene_name"),
                    pl.lit(tissue).alias("tissue"),
                    pl.col("_vals").list.mean().alias("mean_tpm"),
                    pl.col("_vals").list.median().alias("median_tpm"),
                    # ddof=1 (sample std); set to 0 for population std
                    pl.col("_vals").list.std(ddof=1).alias("std_tpm"),
                    pl.lit(len(samples)).cast(pl.Int32).alias("n_samples"),
                )
            )
            result_parts.append(tissue_df)

    # --- 5. Combine and write -------------------------------------------------
    t = _log("Combining and sorting results...")
    result = pl.concat(result_parts).sort(["gene_id", "tissue"])
    _log(f"Result shape: {result.shape[0]:,} rows x {result.shape[1]} columns", t)

    t = _log(f"Writing results to {output_file}...")
    result.write_csv(output_file, separator="\t")
    _log(f"Saved to: {Path(output_file).resolve()}", t)

    total = time.time() - script_start
    _log(f"All done. Total elapsed: {total:.1f}s ({total/60:.1f} min)")
    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    tpm_file = sys.argv[1]
    sample_attrs_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "gtex_gene_tissue_stats.tsv"

    compute_stats(tpm_file, sample_attrs_file, output_file)

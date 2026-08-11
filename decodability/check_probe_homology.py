"""
check_probe_homology.py
-----------------------
Assess within-split sequence-identity leakage for the 486-protein probe eval set.

For each pair of proteins, computes % sequence identity using Biopython's
global pairwise aligner (BLOSUM62, gap penalties matching BLAST defaults).
Then simulates the 5-fold GroupKFold assignment and reports what fraction of
test proteins have a near-homolog (>30% identity) in their corresponding
training fold.

Usage:
    python check_probe_homology.py [--identity-threshold 0.30] [--n-folds 5]

Outputs:
    probe_homology_report.txt   — human-readable summary
    probe_homology_matrix.npy   — (N x N) float32 pairwise identity matrix
    probe_homology_pairs.csv    — all pairs exceeding the threshold
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
from Bio import Align, SeqIO
from sklearn.model_selection import GroupKFold

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent
COMMON_PROTEINS = REPO / "inputs" / "common_activation_proteins.txt"
# Sequences come from the processed shards when they have been built
# (swissprot_annotation_pipeline.py), else from the committed flat FASTA of the
# same 5,000-protein SwissProt-A5 pool.
SHARD_DIRS = [REPO / "processed_swissprot_a5" / f"shard_{i}" for i in range(8)]
FALLBACK_FASTA = REPO / "inputs" / "swissprot_a5.fasta"
OUT_DIR = REPO / "data" / "homology"
OUT_REPORT = OUT_DIR / "probe_homology_report.txt"
OUT_MATRIX = OUT_DIR / "probe_homology_matrix.npy"
OUT_PAIRS = OUT_DIR / "probe_homology_pairs.csv"


# ---------------------------------------------------------------------------
# 1. Load sequences
# ---------------------------------------------------------------------------
def load_eval_sequences(protein_ids: list[str]) -> dict[str, str]:
    id_set = set(protein_ids)
    seqs: dict[str, str] = {}

    sources = [s / "sequences.fasta" for s in SHARD_DIRS]
    if not any(f.exists() for f in sources):
        if not FALLBACK_FASTA.exists():
            raise SystemExit(
                "No sequences found. Build the processed shards with\n"
                "  python swissprot_annotation_pipeline.py \\\n"
                "      --input_ids inputs/swissprot_a5_ids.txt \\\n"
                "      --output_dir processed_swissprot_a5\n"
                f"or restore {FALLBACK_FASTA}."
            )
        sources = [FALLBACK_FASTA]

    for fasta in sources:
        if not fasta.exists():
            continue
        for rec in SeqIO.parse(fasta, "fasta"):
            pid = rec.id.split()[0]
            if pid in id_set:
                seqs[pid] = str(rec.seq)
        if len(seqs) == len(id_set):
            break

    missing = id_set - set(seqs)
    if missing:
        print(f"WARNING: {len(missing)} of {len(id_set)} eval proteins have no sequence "
              f"(e.g. {sorted(missing)[:5]})")
    return seqs


# ---------------------------------------------------------------------------
# 2. Pairwise % identity  (global alignment, BLOSUM62)
# ---------------------------------------------------------------------------
def pct_identity(seq_a: str, seq_b: str, aligner: Align.PairwiseAligner) -> float:
    """Global alignment identity = identical positions / alignment length."""
    if seq_a == seq_b:
        return 1.0
    alignments = aligner.align(seq_a, seq_b)
    best = next(iter(alignments))
    aligned_a, aligned_b = best[0], best[1]
    matches = sum(a == b and a != "-" for a, b in zip(aligned_a, aligned_b))
    length = max(len(seq_a), len(seq_b))  # denominator: longer sequence
    return matches / length if length > 0 else 0.0


def build_identity_matrix(
    ordered_ids: list[str], seqs: dict[str, str], aligner: Align.PairwiseAligner
) -> np.ndarray:
    n = len(ordered_ids)
    mat = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(mat, 1.0)
    total_pairs = n * (n - 1) // 2
    done = 0
    t0 = time.time()
    for i in range(n):
        for j in range(i + 1, n):
            identity = pct_identity(seqs[ordered_ids[i]], seqs[ordered_ids[j]], aligner)
            mat[i, j] = mat[j, i] = identity
            done += 1
            if done % 500 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (total_pairs - done) / rate
                print(
                    f"  {done}/{total_pairs} pairs  ({rate:.0f} pairs/s, "
                    f"ETA {eta/60:.1f} min)",
                    end="\r",
                )
    print()
    return mat


# ---------------------------------------------------------------------------
# 3. Fold-aware leakage stats
# ---------------------------------------------------------------------------
def fold_leakage_stats(
    ordered_ids: list[str],
    mat: np.ndarray,
    threshold: float,
    n_folds: int,
) -> dict:
    n = len(ordered_ids)
    groups = np.arange(n)  # one group per protein (GroupKFold groups by protein)
    # dummy X and y — only groups matter
    X_dummy = np.zeros((n, 1))
    y_dummy = np.zeros(n)

    leaked_test_proteins: set[int] = set()
    fold_summaries = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        GroupKFold(n_splits=n_folds).split(X_dummy, y_dummy, groups)
    ):
        leaked_in_fold = []
        for ti in test_idx:
            # does this test protein have any training protein > threshold?
            max_id_to_train = mat[ti, train_idx].max()
            if max_id_to_train > threshold:
                leaked_in_fold.append((ti, float(max_id_to_train)))
                leaked_test_proteins.add(ti)
        fold_summaries.append(
            {
                "fold": fold_idx,
                "n_test": len(test_idx),
                "n_leaked": len(leaked_in_fold),
                "details": leaked_in_fold,
            }
        )

    # Global stats
    above_threshold = mat > threshold
    np.fill_diagonal(above_threshold, False)
    has_homolog = above_threshold.any(axis=1)  # any partner > threshold

    return {
        "fold_summaries": fold_summaries,
        "n_with_any_homolog": int(has_homolog.sum()),
        "n_leaked_test_proteins": len(leaked_test_proteins),
        "n_total": n,
    }


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity-threshold", type=float, default=0.30)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()
    threshold = args.identity_threshold
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading protein IDs...")
    protein_ids = COMMON_PROTEINS.read_text().strip().splitlines()
    print(f"  {len(protein_ids)} proteins in common_activation_proteins.txt")

    print("Loading sequences from shards...")
    seqs = load_eval_sequences(protein_ids)
    missing = [p for p in protein_ids if p not in seqs]
    if missing:
        print(f"  WARNING: {len(missing)} proteins not found in shards: {missing[:5]}...")
    ordered_ids = [p for p in protein_ids if p in seqs]
    print(f"  {len(ordered_ids)} sequences loaded")

    # Check for cached matrix
    if OUT_MATRIX.exists():
        print(f"Loading cached identity matrix from {OUT_MATRIX}...")
        mat = np.load(OUT_MATRIX)
        assert mat.shape == (len(ordered_ids), len(ordered_ids)), (
            f"Cached matrix shape {mat.shape} doesn't match {len(ordered_ids)} proteins. "
            "Delete probe_homology_matrix.npy to recompute."
        )
    else:
        aligner = Align.PairwiseAligner()
        aligner.mode = "global"
        aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
        aligner.open_gap_score = -11
        aligner.extend_gap_score = -1

        n = len(ordered_ids)
        total_pairs = n * (n - 1) // 2
        print(
            f"Computing {total_pairs:,} pairwise alignments "
            f"({n} proteins, this may take ~30-60 min)..."
        )
        mat = build_identity_matrix(ordered_ids, seqs, aligner)
        np.save(OUT_MATRIX, mat)
        print(f"Saved matrix to {OUT_MATRIX}")

    # Threshold stats
    print(f"\nAnalysing at {threshold*100:.0f}% identity threshold, "
          f"{args.n_folds}-fold GroupKFold...")
    stats = fold_leakage_stats(ordered_ids, mat, threshold, args.n_folds)

    # Pairs above threshold
    rows_i, rows_j = np.where((mat > threshold) & (np.tri(len(ordered_ids), k=-1).T.astype(bool)))
    pairs_above = [
        (ordered_ids[i], ordered_ids[j], float(mat[i, j]))
        for i, j in zip(rows_i, rows_j)
    ]

    # Write pairs CSV
    with open(OUT_PAIRS, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["protein_a", "protein_b", "pct_identity"])
        for pa, pb, pid in sorted(pairs_above, key=lambda x: -x[2]):
            writer.writerow([pa, pb, f"{pid:.4f}"])

    # Write report
    lines = []
    lines.append("=" * 60)
    lines.append("Probe eval-set homology leakage report")
    lines.append("=" * 60)
    lines.append(f"Proteins: {stats['n_total']}")
    lines.append(f"Identity threshold: {threshold*100:.0f}%")
    lines.append(f"Folds (GroupKFold by protein): {args.n_folds}")
    lines.append("")
    lines.append(f"Proteins with >=1 near-homolog (>{threshold*100:.0f}% ID) in the set:")
    lines.append(
        f"  {stats['n_with_any_homolog']} / {stats['n_total']} "
        f"({100*stats['n_with_any_homolog']/stats['n_total']:.1f}%)"
    )
    lines.append("")
    lines.append(f"Total pairs above threshold: {len(pairs_above)}")
    lines.append("")
    lines.append("Per-fold leakage (test proteins with a training-fold homolog):")
    total_leaked = 0
    total_test = 0
    for fs in stats["fold_summaries"]:
        pct = 100 * fs["n_leaked"] / fs["n_test"] if fs["n_test"] else 0
        lines.append(
            f"  Fold {fs['fold']}: {fs['n_leaked']}/{fs['n_test']} test proteins "
            f"({pct:.1f}%) have a >{threshold*100:.0f}% ID neighbour in train"
        )
        total_leaked += fs["n_leaked"]
        total_test += fs["n_test"]

    lines.append("")
    lines.append(
        f"Overall: {total_leaked}/{total_test} test-protein appearances "
        f"({100*total_leaked/total_test:.1f}%) have a training-fold near-homolog"
    )
    lines.append("")
    lines.append(
        "Interpretation: because both stacks (pairformer and diffusion) use "
        "identical protein sets and identical fold assignments, any inflation of "
        "absolute probe-raw F1 due to leakage is common-mode and cannot explain "
        "the differential attenuation of sequence-chemistry signals in the "
        "diffusion module."
    )
    lines.append("")
    lines.append(f"Top 10 highest-identity pairs:")
    for pa, pb, pid in sorted(pairs_above, key=lambda x: -x[2])[:10]:
        lines.append(f"  {pa}  {pb}  {pid*100:.1f}%")

    report_text = "\n".join(lines)
    OUT_REPORT.write_text(report_text, encoding="utf-8")
    print("\n" + report_text)
    print(f"\nReport saved to {OUT_REPORT}")
    print(f"Pairs CSV saved to {OUT_PAIRS}")


if __name__ == "__main__":
    main()

import bisect
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, TypedDict

class TERecord(TypedDict):
    chrom:  str
    start:  int
    end:    int
    name:   str
    family: str

K9Dict = Dict[str, Tuple[List[int], List[int], List[int]]]

def load_K9(filepath: str) -> K9Dict:
    """
    Load a bedgraph file into a dict with format:
        chrom   start   end   count
        2L      435878  435924  9
    Returns sorted {chrom: ([starts], [ends], [counts])}
    """
    chrom_data: K9Dict = {}

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            chrom, start, end, count = parts[0], int(parts[1]), int(parts[2]), int(parts[3])

            if chrom not in chrom_data:
                chrom_data[chrom] = ([], [], [])
            chrom_data[chrom][0].append(start)
            chrom_data[chrom][1].append(end)
            chrom_data[chrom][2].append(count)

    for chrom, (starts, ends, counts) in chrom_data.items():
        if starts != sorted(starts):
            triples = sorted(zip(starts, ends, counts))
            s, e, c = zip(*triples)
            chrom_data[chrom] = (list(s), list(e), list(c))

    return chrom_data

def load_TE_ann(filepath: str) -> List[TERecord]:
    """
    Load a TE annotation file with format:
        chrom  start  end  name  family  ...
        2L     489573 491304 roo LTR/Bel-Pao ...
    Returns a list with one dict per TE. 
    """
    records: List[TERecord] = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            records.append({
                "chrom":  parts[0],
                "start":  int(parts[1]),
                "end":    int(parts[2]),
                "name":   parts[3],
                "family": parts[4],
            })
    return records

def extract_K9_window(
    k9: K9Dict,
    chrom: str,
    center: int,
    window: int = 20_000,
    n_bins: int = 400,
) -> Optional[np.ndarray]:
    """
    Extract binned K9 signal in [center-window, center+window].
    Counts are weighted by bp overlap per bin.
    Returns np.ndarray of shape (n_bins,), or None if chrom not in k9.
    """
    if chrom not in k9:
        return None

    starts, ends, counts = k9[chrom]
    query_start = center - window
    query_end   = center + window
    bin_size    = (2 * window) / n_bins

    idx_right = bisect.bisect_left(starts, query_end)
    idx_left  = max(0, bisect.bisect_left(starts, query_start) - 1)

    result = np.zeros(n_bins, dtype=np.float32)

    for i in range(idx_left, idx_right):
        s, e, c = starts[i], ends[i], counts[i]
        if e <= query_start or s >= query_end:
            continue

        overlap_start = max(s, query_start)
        overlap_end   = min(e, query_end)

        bin_start_idx = int((overlap_start - query_start) / bin_size)
        bin_end_idx   = min(int((overlap_end - query_start) / bin_size), n_bins - 1)

        for b in range(bin_start_idx, bin_end_idx + 1):
            bin_bp_start = query_start + b * bin_size
            bin_bp_end   = bin_bp_start + bin_size
            bp_in_bin    = min(overlap_end, bin_bp_end) - max(overlap_start, bin_bp_start)
            result[b]   += c * (bp_in_bin / bin_size)   # K9 count normalized by bp (although usually should be 1)

    return result


def build_te_df(
    A4_k9:  K9Dict,
    A7_k9:  K9Dict,
    A4_ann: List[TERecord],
    A7_ann: List[TERecord],
    window: int = 20_000,
    n_bins: int = 800,
) -> pd.DataFrame:
    """
    Build a DataFrame with one row per TE locus from either strain.
    K9 windows are extracted from both strains at every locus.
    Returns DataFrame: chrom, locus_start, locus_end, center, A4_K9, A7_K9, label_A4, label_A7
    """
    rows = []

    def process(ann, label_A4, label_A7):
        for te in ann:
            chrom  = te["chrom"]
            center = (te["start"] + te["end"]) // 2
            A4_sig = extract_K9_window(A4_k9, chrom, center, window, n_bins)
            A7_sig = extract_K9_window(A7_k9, chrom, center, window, n_bins)
            if A4_sig is None or A7_sig is None:
                continue
            rows.append({
                "chrom":       chrom,
                "locus_start": te["start"],
                "locus_end":   te["end"],
                "center":      center,
                "A4_K9":       A4_sig,
                "A7_K9":       A7_sig,
                "label_A4":    label_A4,
                "label_A7":    label_A7,
            })

    process(A4_ann, label_A4=1, label_A7=0)
    process(A7_ann, label_A4=0, label_A7=1)

    return pd.DataFrame(rows)

def sample_negatives(
    te_df:     pd.DataFrame,
    A4_k9:     K9Dict,
    A7_k9:     K9Dict,
    neg_ratio: float = 0.10,
    window:    int   = 20_000,
    n_bins:    int   = 800,
    seed:      int   = 42,
) -> pd.DataFrame:
    """
    Sample random genomic positions as negative samples (no TE in either strain).
    Candidates are midpoints of A4 K9 intervals that don't overlap a TE window.
    Returns DataFrame with same columns as te_df, all labels 0.
    """
    rng   = np.random.default_rng(seed)
    n_neg = int(len(te_df) * neg_ratio)

    # Build sorted te_windows per chrom
    te_regions: Dict[str, List[Tuple[int, int]]] = {}
    for _, row in te_df.iterrows():
        chrom, c = row["chrom"], row["center"]
        te_regions.setdefault(chrom, []).append((c - window, c + window))
    for chrom in te_regions:
        te_regions[chrom].sort()

    def overlaps_te(chrom: str, center: int) -> bool:
        if chrom not in te_regions:
            return False
        regions = te_regions[chrom]
        q_start, q_end = center - window, center + window
        idx = bisect.bisect_left(regions, (q_start,))
        for r_start, r_end in regions[max(0, idx - 1): idx + 2]:
            if r_start < q_end and r_end > q_start:
                # candidate window [center-w, center+w] overlaps TE window [c-w, c+w]
                # iff |candidate - te_center| < 2*window
                return True
        return False

    candidates = [
        (chrom, (s + e) // 2)
        for chrom, (starts, ends, _) in A4_k9.items()
        for s, e in zip(starts, ends)
    ]

    rows = []
    for idx in rng.permutation(len(candidates)):
        if len(rows) >= n_neg:
            break
        chrom, center = candidates[idx]
        if overlaps_te(chrom, center):
            continue
        A4_sig = extract_K9_window(A4_k9, chrom, center, window, n_bins)
        A7_sig = extract_K9_window(A7_k9, chrom, center, window, n_bins)
        if A4_sig is None or A7_sig is None:
            continue
        rows.append({
            "chrom":       chrom,
            "locus_start": center - window,
            "locus_end":   center + window,
            "center":      center,
            "A4_K9":       A4_sig,
            "A7_K9":       A7_sig,
            "label_A4":    0,
            "label_A7":    0,
        })

    return pd.DataFrame(rows)


def build_full_df(te_df: pd.DataFrame, neg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate te_df and neg_df into a single training DataFrame.
    """
    return pd.concat([te_df, neg_df], ignore_index=True)

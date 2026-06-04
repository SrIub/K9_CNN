import bisect
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, TypedDict

class MappedTE(TypedDict):
    chrom:     str
    src_start: int
    src_end:   int
    tgt_start: int
    tgt_end:   int
    name:      str
    family:    str

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

def load_TE_map(filepath: str) -> List[MappedTE]:
    """
    Load a synteny map file with format:
        chrom  src_start  src_end  name  family  length  flag  tgt_start  tgt_end
        2L     489573     491304   roo   LTR/...  1731   1     589113     589113
    Only rows with flag=1 (syntenic position known) are returned.
    tgt_start/tgt_end are the corresponding breakpoint coords in the other strain.
    """
    records: List[MappedTE] = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if int(parts[6]) != 1:
                continue
            records.append({
                "chrom":     parts[0],
                "src_start": int(parts[1]),
                "src_end":   int(parts[2]),
                "tgt_start": int(parts[7]),
                "tgt_end":   int(parts[8]),
                "name":      parts[3],
                "family":    parts[4],
            })
    return records


def extract_K9_window(
    k9: K9Dict,
    chrom: str,
    center: int,
    window: int = 10_000,
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
            result[b]   += c * (bp_in_bin / bin_size)   # K9 count normalized by bp (bins are almost always the same # of bp)

    return result


def build_te_df(
    A4_k9:    K9Dict,
    A7_k9:    K9Dict,
    A4_to_A7: List[MappedTE],
    A7_to_A4: List[MappedTE],
    flank:    int = 10000,
    n_bins:   int = 400,
) -> pd.DataFrame:
    """
    Build a DataFrame with one row per TE locus using synteny-aware window extraction.

    For A4_to_A7 entries (TE in A4, syntenic breakpoint in A7):
        A4_K9 extracted at A4 coords  → label_A4=1
        A7_K9 extracted at A7 tgt coords → label_A7=0

    For A7_to_A4 entries (TE in A7, syntenic breakpoint in A4):
        A7_K9 extracted at A7 coords  → label_A7=1
        A4_K9 extracted at A4 tgt coords → label_A4=0

    Returns DataFrame: chrom, locus_start, locus_end, center, A4_K9, A7_K9, label_A4, label_A7
    """
    half_flank    = flank  // 2
    bins_per_side = n_bins // 2
    rows = []

    for te in A4_to_A7:
        chrom      = te["chrom"]
        A4_start   = te["src_start"]
        A4_end     = te["src_end"]
        A7_center  = (te["tgt_start"] + te["tgt_end"]) // 2

        A4_left  = extract_K9_window(A4_k9, chrom, A4_start - half_flank, half_flank, bins_per_side)
        A4_right = extract_K9_window(A4_k9, chrom, A4_end   + half_flank, half_flank, bins_per_side)
        A7_left  = extract_K9_window(A7_k9, chrom, A7_center - half_flank, half_flank, bins_per_side)
        A7_right = extract_K9_window(A7_k9, chrom, A7_center + half_flank, half_flank, bins_per_side)

        if any(sig is None for sig in [A4_left, A4_right, A7_left, A7_right]):
            continue

        rows.append({
            "chrom":       chrom,
            "locus_start": A4_start,
            "locus_end":   A4_end,
            "center":      (A4_start + A4_end) // 2,
            "A4_K9":       np.concatenate([A4_left,  A4_right]),
            "A7_K9":       np.concatenate([A7_left,  A7_right]),
            "label_A4":    1,
            "label_A7":    0,
        })

    for te in A7_to_A4:
        chrom      = te["chrom"]
        A7_start   = te["src_start"]
        A7_end     = te["src_end"]
        A4_center  = (te["tgt_start"] + te["tgt_end"]) // 2

        A7_left  = extract_K9_window(A7_k9, chrom, A7_start  - half_flank, half_flank, bins_per_side)
        A7_right = extract_K9_window(A7_k9, chrom, A7_end    + half_flank, half_flank, bins_per_side)
        A4_left  = extract_K9_window(A4_k9, chrom, A4_center - half_flank, half_flank, bins_per_side)
        A4_right = extract_K9_window(A4_k9, chrom, A4_center + half_flank, half_flank, bins_per_side)

        if any(sig is None for sig in [A4_left, A4_right, A7_left, A7_right]):
            continue

        rows.append({
            "chrom":       chrom,
            "locus_start": A7_start,
            "locus_end":   A7_end,
            "center":      (A7_start + A7_end) // 2,
            "A4_K9":       np.concatenate([A4_left,  A4_right]),
            "A7_K9":       np.concatenate([A7_left,  A7_right]),
            "label_A4":    0,
            "label_A7":    1,
        })

    return pd.DataFrame(rows)

def sample_negatives(
    te_df:     pd.DataFrame,
    A4_k9:     K9Dict,
    A7_k9:     K9Dict,
    neg_ratio: float = 0.10,
    flank:     int   = 10_000,
    n_bins:    int   = 400,
    seed:      int   = 42,
) -> pd.DataFrame:
    """
    Sample random negative pairs: one A4 position + one independent A7 position,
    each confirmed absent from their own strain's TE regions.

    A4 exclusion zones are built from label_A4=1 rows in te_df (A4 TE loci).
    A7 exclusion zones are built from label_A7=1 rows in te_df (A7 TE loci).
    Candidates are drawn from K9 interval midpoints in each strain independently.

    A4_K9 uses a virtual TE size (sampled from te_df) for realistic locus columns.
    A7_K9 uses the sampled center as a point, matching the tgt convention in build_te_df.
    Returns DataFrame with same columns as te_df, all labels 0.
    """
    rng           = np.random.default_rng(seed)
    n_neg         = int(len(te_df) * neg_ratio)
    te_sizes      = (te_df["locus_end"] - te_df["locus_start"]).values
    half_flank    = flank  // 2
    bins_per_side = n_bins // 2

    def build_exclusion(mask) -> Dict[str, List[Tuple[int, int]]]:
        regions: Dict[str, List[Tuple[int, int]]] = {}
        for _, row in te_df[mask].iterrows():
            regions.setdefault(row["chrom"], []).append(
                (row["locus_start"] - flank, row["locus_end"] + flank)
            )
        for chrom in regions:
            regions[chrom].sort()
        return regions

    A4_regions = build_exclusion(te_df["label_A4"] == 1)
    A7_regions = build_exclusion(te_df["label_A7"] == 1)

    def overlaps(chrom: str, center: int, regions: Dict[str, List[Tuple[int, int]]]) -> bool:
        if chrom not in regions:
            return False
        idx = bisect.bisect_left(regions[chrom], (center,))
        for r_start, r_end in regions[chrom][max(0, idx - 1): idx + 2]:
            if r_start <= center <= r_end:
                return True
        return False

    A4_candidates = [
        (chrom, (s + e) // 2)
        for chrom, (starts, ends, _) in A4_k9.items()
        for s, e in zip(starts, ends)
        if not overlaps(chrom, (s + e) // 2, A4_regions)
    ]
    A7_candidates = [
        (chrom, (s + e) // 2)
        for chrom, (starts, ends, _) in A7_k9.items()
        for s, e in zip(starts, ends)
        if not overlaps(chrom, (s + e) // 2, A7_regions)
    ]

    A4_order = rng.permutation(len(A4_candidates))
    A7_order = rng.permutation(len(A7_candidates))

    rows = []
    a7_i = 0
    for a4_i in A4_order:
        if len(rows) >= n_neg or a7_i >= len(A7_candidates):
            break

        A4_chrom, A4_center = A4_candidates[int(a4_i)]
        A7_chrom, A7_center = A7_candidates[int(A7_order[a7_i])]
        a7_i += 1

        half_te  = int(rng.choice(te_sizes)) // 2
        te_start = A4_center - half_te
        te_end   = A4_center + half_te

        A4_left  = extract_K9_window(A4_k9, A4_chrom, te_start  - half_flank, half_flank, bins_per_side)
        A4_right = extract_K9_window(A4_k9, A4_chrom, te_end    + half_flank, half_flank, bins_per_side)
        A7_left  = extract_K9_window(A7_k9, A7_chrom, A7_center - half_flank, half_flank, bins_per_side)
        A7_right = extract_K9_window(A7_k9, A7_chrom, A7_center + half_flank, half_flank, bins_per_side)

        if any(sig is None for sig in [A4_left, A4_right, A7_left, A7_right]):
            continue

        rows.append({
            "chrom":       A4_chrom,
            "locus_start": te_start,
            "locus_end":   te_end,
            "center":      A4_center,
            "A4_K9":       np.concatenate([A4_left,  A4_right]),
            "A7_K9":       np.concatenate([A7_left,  A7_right]),
            "label_A4":    0,
            "label_A7":    0,
        })

    return pd.DataFrame(rows)


def build_full_df(te_df: pd.DataFrame, neg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate te_df and neg_df into a single training DataFrame.
    """
    return pd.concat([te_df, neg_df], ignore_index=True)

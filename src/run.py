import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from load_df      import load_K9, load_TE_ann, build_te_df, sample_negatives, build_full_df
from load_tensors import make_splits
from simple_K9_CNN import train

data_dir = Path(__file__).parent.parent / "test_data"

A4_K9_PATH  = data_dir / "A4_K9.bed"
A7_K9_PATH  = data_dir / "A7_K9.bed"
A4_ANN_PATH = data_dir / "A4_TE_annotations.txt"
A7_ANN_PATH = data_dir / "A7_TE_annotations.txt"


print("Loading data...")
A4_k9 = load_K9(str(A4_K9_PATH))
A7_k9 = load_K9(str(A7_K9_PATH))
A4_ann = load_TE_ann(str(A4_ANN_PATH))
A7_ann = load_TE_ann(str(A7_ANN_PATH))

te_df  = build_te_df(A4_k9, A7_k9, A4_ann, A7_ann, n_bins= 800)
neg_df = sample_negatives(te_df, A4_k9, A7_k9, neg_ratio=0.10, n_bins= 800)
full_df = build_full_df(te_df, neg_df)

print(f"Dataset: {len(te_df)} TE loci  |  {len(neg_df)} negatives  |  {len(full_df)} total")
train_df, val_df, test_df = make_splits(full_df, val_fraction=0.15, test_fraction=0.15)
print(f"Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}\n")

model, history, test_metrics = train(
    train_df, val_df, test_df,
    batch_size = 32,
    n_epochs   = 40,
    lr         = 1e-3,
    dropout    = 0.3,
)

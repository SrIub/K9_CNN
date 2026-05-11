import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from load_tensors import make_tensors


class K9_CNN(nn.Module):
    """
    2D CNN predicting TE presence in A4/A7 strains from K9 data.
    Input:  (batch, 1, 2, n_bins)  — 1 channel (K9), 2 strains (A4/A7), n_bins (K9 counts)
    Output: (batch, 2)             — logits for [label_A4, label_A7]
    """

    def __init__(self, dropout=0.3):
        super().__init__()

        # Input: (B, 1, 2, 800)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 7), padding=(0, 3)),  # (B, 16, 2, 800)  within-strain
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),                       # (B, 16, 2, 400)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 7), padding=(0, 3)), # (B, 32, 1, 400)  cross-strain 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),                       # (B, 32, 1, 200)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 5), padding=(0, 2)), # (B, 64, 1, 200)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),                          # (B, 64, 1,   1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),        # (B, 64, 1, 1) → (B, 64)
            nn.Linear(64, 32),   # (B, 64)       → (B, 32)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),    # (B, 32)        → (B,  2)  A4 and A7 logits
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return self.classifier(x)


def evaluate(model, loader, device):
    """
    Evaluate model on a DataLoader.
    Returns dict: loss, bAcc_A4, bAcc_A7, auROC_A4, auROC_A7
    """
    model.eval()
    criterion  = nn.BCEWithLogitsLoss()
    all_logits, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits      = model(X_batch)
            total_loss += criterion(logits, y_batch).item()
            all_logits.append(logits.cpu())
            all_labels.append(y_batch.cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = 1 / (1 + np.exp(-logits))  # sigmoid function for probablities
    preds  = (probs >= 0.5).astype(int)

    metrics = {"loss": total_loss / len(loader)}
    for i, strain in enumerate(["A4", "A7"]):
        metrics[f"bAcc_{strain}"]  = balanced_accuracy_score(labels[:, i], preds[:, i])
        metrics[f"auROC_{strain}"] = (
            roc_auc_score(labels[:, i], probs[:, i])
            if len(np.unique(labels[:, i])) > 1 else float("nan")
        )
    return metrics


def predict(model, loader, device):
    """
    Run model on a DataLoader and return raw output arrays for plotting.
    Returns probs (N, 2) and labels (N, 2) as numpy float arrays.
    """
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            all_logits.append(model(X_batch.to(device)).cpu())
            all_labels.append(y_batch)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    return 1 / (1 + np.exp(-logits)), labels


def train(
    train_df,
    val_df,
    test_df,
    batch_size = 32,
    n_epochs   = 80,
    lr         = 1e-3,
    dropout    = 0.3,
    seed       = 42,
    save_path  = "best_model.pth",
):
    """
    Train K9_CNN, checkpoint the best val-loss model, evaluate on test at the end.
    Returns (model, history, test_metrics).
    history is a list of dicts, one per epoch: epoch, train_loss, val loss/bAcc/auROC.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X_train, y_train = make_tensors(train_df)
    X_val,   y_val   = make_tensors(val_df)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False)

    model     = K9_CNN(dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n{'Epoch':>5}  {'TrainLoss':>10}  {'ValLoss':>8}  "
          f"{'bAcc_A4':>8}  {'bAcc_A7':>8}  {'auROC_A4':>9}  {'auROC_A7':>9}")
    print("-" * 75)

    history       = []
    best_val_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        val_metrics = evaluate(model, val_loader, device)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), save_path)
            print(f"  → saved best model (epoch {epoch})")

        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        print(f"{epoch:>5}  {train_loss:>10.4f}  {val_metrics['loss']:>8.4f}  "
              f"{val_metrics['bAcc_A4']:>8.3f}  {val_metrics['bAcc_A7']:>8.3f}  "
              f"{val_metrics['auROC_A4']:>9.3f}  {val_metrics['auROC_A7']:>9.3f}")

    # Final evaluation on test set using best checkpoint
    model.load_state_dict(torch.load(save_path, weights_only=True))
    X_test, y_test = make_tensors(test_df)
    test_loader    = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    test_metrics   = evaluate(model, test_loader, device)

    print(f"\nTest  auROC_A4={test_metrics['auROC_A4']:.3f}  "
          f"auROC_A7={test_metrics['auROC_A7']:.3f}  "
          f"bAcc_A4={test_metrics['bAcc_A4']:.3f}  "
          f"bAcc_A7={test_metrics['bAcc_A7']:.3f}")

    return model, history, test_metrics

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ============================================================
# DEVICE SETUP & DATA LOADING
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

for file in ["./X.npy", "./y.npy"]:
    if not os.path.exists(file):
        raise FileNotFoundError(
            f"Could not find '{file}'! Make sure you have uploaded the dataset files "
            f"directly to the file icon menu on the left side of your Colab screen."
        )

X_raw = np.load("./X.npy")
y = np.load("./y.npy")

valid_mask = y >= 0
X_raw = X_raw[valid_mask]
y = y[valid_mask]

X_raw = np.transpose(X_raw, (0, 3, 2, 1)).squeeze(-1)

def reshape_tiles(X):
    N = X.shape[0]
    out = np.zeros((N, 15, 4, 9), dtype=np.float32)
    out[:, :, 0, :] = X[:, :, 0:9]    # Manzu
    out[:, :, 1, :] = X[:, :, 9:18]   # Pinzu
    out[:, :, 2, :] = X[:, :, 18:27]  # Souzu
    out[:, :, 3, :7] = X[:, :, 27:34] # Honors + Padding
    return out

X = reshape_tiles(X_raw)

# Splits
np.random.seed(42)
indices = np.random.permutation(len(X))
n_total = len(X)
n_test = int(0.15 * n_total)
n_val = int(0.15 * n_total)

X_train, y_train = X[indices[n_test+n_val:]], y[indices[n_test+n_val:]]
X_val, y_val = X[indices[n_test:n_test+n_val]], y[indices[n_test:n_test+n_val]]
X_test, y_test = X[indices[:n_test]], y[indices[:n_test]]

batch_size = 256
train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)), batch_size=batch_size, shuffle=False)

# ============================================================
# MODEL ARCHITECTURE
# ============================================================
class MahjongResidualBlock(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), bias=True)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

class MahjongWinPredictor(nn.Module):
    def __init__(self, in_channels=15, feature_channels=128):
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, feature_channels, kernel_size=(1, 3), padding=(0, 1), bias=True)
        self.init_bn = nn.BatchNorm2d(feature_channels)
        
        self.res_blocks = nn.Sequential(*[MahjongResidualBlock(feature_channels) for _ in range(5)])
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_channels * 4 * 9, 512),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        x = F.relu(self.init_bn(self.init_conv(x)))
        x = self.res_blocks(x)
        x = x.view(x.size(0), -1) 
        return self.classifier(x)

model = MahjongWinPredictor().to(device)

if torch.cuda.is_available() and hasattr(torch, 'compile'):
    try:
        print("Attempting model compilation for hardware acceleration...")
        model = torch.compile(model)
        print("Model compiled successfully!")
    except Exception as e:
        print(f"Compilation skipped/failed (falling back to standard PyTorch execution): {e}")

# ============================================================
# OPTIMIZER & SCHEDULER SETUP
# ============================================================
pos_weight = torch.tensor([1.5], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# ============================================================
# EVALUATION HELPER
# ============================================================
def evaluate_performance(model, loader, threshold=0.5):
    model.eval()
    probs_all, targets_all = [], []
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch_logits = y_batch.float().unsqueeze(1).to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch_logits)
            
            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            probs_all.extend(probs.cpu().numpy().flatten())
            targets_all.extend(y_batch.numpy())

    avg_loss = total_loss / len(loader)
    probs_all, targets_all = np.array(probs_all), np.array(targets_all)
    preds = (probs_all > threshold).astype(int)

    return {
        "loss": avg_loss,
        "acc": accuracy_score(targets_all, preds),
        "prec": precision_score(targets_all, preds, zero_division=0),
        "rec": recall_score(targets_all, preds, zero_division=0),
        "f1": f1_score(targets_all, preds, zero_division=0),
        "auc": roc_auc_score(targets_all, probs_all)
    }, probs_all, targets_all

# ============================================================
# EXTENDED TRAINING LOOP WITH EARLY STOPPING
# ============================================================
best_val_loss = float('inf')
patience_counter = 0
early_stop_patience = 5  # Stop training if val loss doesn't improve for 5 epochs

print("\nStarting deep optimization training loop...")

for epoch in range(1, 41): # Runway extended to 40
    model.train()
    total_train_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.float().unsqueeze(1).to(device)

        optimizer.zero_grad(set_to_none=True)
        
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    val_metrics, _, _ = evaluate_performance(model, val_loader, threshold=0.5)
    scheduler.step(val_metrics["loss"])

    print(
        f"Epoch {epoch:2d} | "
        f"Train Loss {avg_train_loss:.4f} | "
        f"Val Loss {val_metrics['loss']:.4f} | "
        f"Val AUC {val_metrics['auc']:.4f} | "
        f"Val F1@0.5 {val_metrics['f1']:.4f}"
    )

    if val_metrics["loss"] < best_val_loss:
        best_val_loss = val_metrics["loss"]
        patience_counter = 0
        torch.save(model.state_dict(), "best_deep_mahjong.pth")
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered at epoch {epoch}. Validation loss hasn't improved.")
            break

# ============================================================
# THRESHOLD OPTIMIZATION AND FINAL VERIFICATION
# ============================================================
print("\nLoading best weights for ultimate evaluation...")
model.load_state_dict(torch.load("best_deep_mahjong.pth", map_location=device))

_, val_probs, val_targets = evaluate_performance(model, val_loader, threshold=0.5)

best_thresh = 0.5
best_val_f1 = 0
for thresh in np.arange(0.1, 0.9, 0.02):
    preds = (val_probs > thresh).astype(int)
    current_f1 = f1_score(val_targets, preds, zero_division=0)
    if current_f1 > best_val_f1:
        best_val_f1 = current_f1
        best_thresh = thresh

print(f"Optimized Decision Threshold Found: {best_thresh:.2f} (Val F1: {best_val_f1:.4f})")
test_metrics, _, _ = evaluate_performance(model, test_loader, threshold=best_thresh)

print("\n" + "="*48)
print("FINAL TEST RESULTS (MAX OPTIMIZATION RUN)")
print("="*48)
for k, v in test_metrics.items():
    print(f"{k.upper()}: {v:.4f}")
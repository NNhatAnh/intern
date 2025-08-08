import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ======================
# Config
# ======================
DATA_DIR = "melspect_intrument"
LABEL_FILE = "dataset_intrument.csv"
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1
SEED = 42
torch.manual_seed(SEED)

# ======================
# Dataset
# ======================
class EarthquakeDataset(Dataset):
    def __init__(self, data_dir, label_file):
        self.data_dir = data_dir
        self.labels = pd.read_csv(label_file, sep=";")
        self.filenames = self.labels["filename"].str.replace(".mseed", ".npy")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file = self.filenames.iloc[idx]
        filepath = os.path.join(self.data_dir, file)
        mel = np.load(filepath).astype(np.float32)  # (128, 128)

        if mel.shape != (128, 128):
            raise ValueError(f"Invalid shape {mel.shape} for file {file}")

        mel_tensor = torch.tensor(mel).unsqueeze(0)  # (1, 128, 128)
        label = torch.tensor(self.labels.iloc[idx]["label"], dtype=torch.float32).unsqueeze(0)
        return mel_tensor, label

# ======================
# Model
# ======================
class CutoffRegressorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # (16, 64, 64)
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # (32, 32, 32)
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # (64, 16, 16)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

# ======================
# Train / Evaluate
# ======================
def train_model(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def eval_model(model, loader, criterion):
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item() * x.size(0)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    return total_loss / len(loader.dataset), np.array(y_true), np.array(y_pred)

# ======================
# Main
# ======================
def main():
    # Load dataset
    full_dataset = EarthquakeDataset(DATA_DIR, LABEL_FILE)
    total_len = len(full_dataset)
    val_len = int(VAL_SPLIT * total_len)
    test_len = int(TEST_SPLIT * total_len)
    train_len = total_len - val_len - test_len
    train_set, val_set, test_set = random_split(full_dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    model = CutoffRegressorCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Training loop
    train_losses, val_losses = [], []
    best_mae = float("inf")
    best_r2 = -float("inf")
    best_model_path = "best_cutoff_predictor.pth"

    for epoch in range(EPOCHS):
        train_loss = train_model(model, train_loader, optimizer, criterion)
        val_loss, y_true, y_pred = eval_model(model, val_loader, criterion)

        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"[{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {mae:.4f} | Val R2: {r2:.4f}")

        if mae < best_mae:
            best_mae = mae
            best_r2 = r2
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model (MAE: {mae:.4f}, R2: {r2:.4f}) to {best_model_path}")


    # Evaluate
    test_loss, y_true, y_pred = eval_model(model, test_loader, criterion)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"\nTest MSE Loss: {test_loss:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")

    # Plot loss
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.show()

    print(f"\nBest MAE on validation: {best_mae:.4f}")
    print(f"Corresponding R2: {best_r2:.4f}")


if __name__ == "__main__":
    main()

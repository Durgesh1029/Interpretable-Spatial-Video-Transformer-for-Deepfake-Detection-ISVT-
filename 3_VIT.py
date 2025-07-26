import os
import torch
from torch.utils.data import Dataset, DataLoader
import random
from collections import defaultdict
import timm
import torch.nn as nn
import torch.nn.functional as F

# Check CUDA
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

# === CONFIG ===
real_seq_path = r"C:\Users\VIMS_Lab\Downloads\Project_Deepfake\DeepfakeProject\sequence_tensors\real"
fake_seq_path = r"C:\Users\VIMS_Lab\Downloads\Project_Deepfake\DeepfakeProject\sequence_tensors\fake"
EPOCHS = 5
BATCH_SIZE = 8
LR = 3e-5
SAVE_DIR = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# === GROUPING FUNCTION ===
def group_by_video(directory):
    grouped = defaultdict(list)
    for file in os.listdir(directory):
        if file.endswith(".pt"):
            vid_id = "_".join(file.split("_")[:2])  # e.g., "000_123"
            grouped[vid_id].append(os.path.join(directory, file))
    return grouped

# === COLLECT AND GROUP ===
real_groups = group_by_video(real_seq_path)
fake_groups = group_by_video(fake_seq_path)

all_groups = [(vid, paths, 0) for vid, paths in real_groups.items()] + \
             [(vid, paths, 1) for vid, paths in fake_groups.items()]

random.shuffle(all_groups)
split_idx = int(0.8 * len(all_groups))
train_groups = all_groups[:split_idx]
val_groups = all_groups[split_idx:]

def expand_groups(groups):
    files, labels = [], []
    for _, paths, label in groups:
        files.extend(paths)
        labels.extend([label] * len(paths))
    return files, labels

train_files, train_labels = expand_groups(train_groups)
val_files, val_labels = expand_groups(val_groups)

# === CUSTOM DATASET ===
class DeepfakeSeqDataset(Dataset):
    def __init__(self, files, labels):
        self.data = files
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.load(self.data[idx])  # [6, 3, 224, 224]
        y = self.labels[idx]
        return x, torch.tensor(y, dtype=torch.float32)

# === MODEL ===
class ViTDeepfake(nn.Module):
    def __init__(self):
        super(ViTDeepfake, self).__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.vit.head = nn.Linear(self.vit.head.in_features, 1)

    def forward(self, x):  # x: [B, 6, 3, 224, 224]
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        x = self.vit(x)                # [B*S, 1]
        x = x.view(B, S)
        x = torch.mean(x, dim=1)
        return x

# === LOADERS ===
train_ds = DeepfakeSeqDataset(train_files, train_labels)
val_ds = DeepfakeSeqDataset(val_files, val_labels)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# === TRAINING ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTDeepfake().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = torch.sigmoid(logits) > 0.5
            correct += (pred.int() == y.int()).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"[Epoch {epoch+1}] Validation Accuracy: {acc:.4f}")

    # Save model
    save_path = os.path.join(SAVE_DIR, f"vit_deepfake_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Run inference on first few validation samples
    print("\nSample predictions:")
    with torch.no_grad():
        for i in range(len(val_ds)):  # first 3 samples
            x_sample, y_sample = val_ds[i]
            x_sample = x_sample.unsqueeze(0).to(device)  # [1, 6, 3, 224, 224]
            y_sample = y_sample.to(device)
            prob = torch.sigmoid(model(x_sample)).item()
            print(f"Sample {i+1} - True label: {int(y_sample.item())}, Predicted prob (fake): {prob:.4f}")
    print("-" * 50)

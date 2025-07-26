import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import timm
import torch.nn as nn

# === CONFIG ===
TEST_REAL_PATH = r"C:\Users\VIMS_Lab\Downloads\Project_Deepfake\DeepfakeProject\test\sequence_tensors\real"
TEST_FAKE_PATH = r"C:\Users\VIMS_Lab\Downloads\Project_Deepfake\DeepfakeProject\test\sequence_tensors\fake"
CHECKPOINT_DIR = r"C:\Users\VIMS_Lab\Downloads\Project_Deepfake\checkpoints"
EPOCHS = [1, 2, 3, 4]
BATCH_SIZE = 8

def group_by_video(directory):
    grouped = defaultdict(list)
    for file in os.listdir(directory):
        if file.endswith(".pt"):
            vid_id = "_".join(file.split("_")[:2])
            grouped[vid_id].append(os.path.join(directory, file))
    return grouped

def expand_groups(groups, label):
    files, labels = [], []
    for _, paths in groups.items():
        files.extend(paths)
        labels.extend([label] * len(paths))
    return files, labels

class DeepfakeSeqDataset(Dataset):
    def __init__(self, files, labels):
        self.data = files
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.load(self.data[idx])
        y = self.labels[idx]
        return x, torch.tensor(y, dtype=torch.float32)

class ViTDeepfake(nn.Module):
    def __init__(self):
        super(ViTDeepfake, self).__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.vit.head = nn.Linear(self.vit.head.in_features, 1)

    def forward(self, x):
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        x = self.vit(x)
        x = x.view(B, S)
        return torch.mean(x, dim=1)

# Load test data
real_groups = group_by_video(TEST_REAL_PATH)
fake_groups = group_by_video(TEST_FAKE_PATH)
real_files, real_labels = expand_groups(real_groups, 0)
fake_files, fake_labels = expand_groups(fake_groups, 1)
test_files = real_files + fake_files
test_labels = real_labels + fake_labels
test_ds = DeepfakeSeqDataset(test_files, test_labels)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# Evaluation loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "loss": [], "auc": []}

for epoch in EPOCHS:
    model = ViTDeepfake().to(device)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, f"vit_deepfake_epoch_{epoch}.pth"), map_location=device))
    model.eval()

    all_probs, all_labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(y.cpu().numpy())

    preds = np.array(all_probs) > 0.5
    metrics["accuracy"].append(accuracy_score(all_labels, preds))
    metrics["precision"].append(precision_score(all_labels, preds))
    metrics["recall"].append(recall_score(all_labels, preds))
    metrics["f1"].append(f1_score(all_labels, preds))
    metrics["loss"].append(log_loss(all_labels, all_probs))
    metrics["auc"].append(roc_auc_score(all_labels, all_probs))

# Plotting results
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs = axs.flatten()
metric_names = ["accuracy", "precision", "recall", "f1", "loss", "auc"]

for i, metric in enumerate(metric_names):
    axs[i].plot(EPOCHS, metrics[metric], marker='o', label=metric.upper())
    axs[i].set_title(f"{metric.upper()} over Epochs")
    axs[i].set_xlabel("Epoch")
    axs[i].set_ylabel(metric.upper())
    axs[i].grid(True)
    axs[i].legend()

plt.tight_layout()
plt.show()

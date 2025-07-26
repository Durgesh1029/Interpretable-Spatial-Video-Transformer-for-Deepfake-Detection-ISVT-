import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict
import timm
import numpy as np

# Configuration
real_test_path = r"C:\Users\VIMS_Lab\Downloads\Project_Deepfake\DeepfakeProject\test\sequence_tensors\real"
fake_test_path = r"C:\Users\VIMS_Lab\Downloads\Project_Deepfake\DeepfakeProject\test\sequence_tensors\fake"
checkpoint_path = r"C:\Users\VIMS_Lab\Downloads\Project_Deepfake\checkpoints\vit_deepfake_epoch_4.pth"

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
        x = torch.mean(x, dim=1)
        return x

class DeepfakeSeqDataset(Dataset):
    def __init__(self, grouped_data):
        self.samples = []
        for video_id, items in grouped_data.items():
            for path in items['paths']:
                self.samples.append((path, video_id, items['label']))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, video_id, label = self.samples[idx]
        tensor = torch.load(path)
        return tensor, label, video_id

def group_test_data(real_dir, fake_dir):
    grouped = defaultdict(lambda: {'paths': [], 'label': None})
    for file in os.listdir(real_dir):
        if file.endswith(".pt"):
            video_id = file.split(".")[0].split("_")[0]
            grouped[video_id]['paths'].append(os.path.join(real_dir, file))
            grouped[video_id]['label'] = 0
    for file in os.listdir(fake_dir):
        if file.endswith(".pt"):
            parts = file.split(".")[0].split("_")
            video_id = f"{parts[0]}_{parts[1]}"
            grouped[video_id]['paths'].append(os.path.join(fake_dir, file))
            grouped[video_id]['label'] = 1
    return grouped

# Load data
grouped_data = group_test_data(real_test_path, fake_test_path)
dataset = DeepfakeSeqDataset(grouped_data)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTDeepfake().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Collect predictions
all_preds, all_probs, all_labels, all_vids = [], [], [], []
with torch.no_grad():
    for x_batch, y_batch, video_ids in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(x_batch)
        probs = torch.sigmoid(logits).squeeze()
        preds = (probs > 0.5).int()

        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(y_batch.cpu().tolist())
        all_vids.extend(video_ids)

# Aggregate predictions per video
video_results = defaultdict(lambda: {'probs': [], 'label': None})
for prob, label, vid in zip(all_probs, all_labels, all_vids):
    video_results[vid]['probs'].append(prob)
    video_results[vid]['label'] = label

video_preds, video_labels, video_probs = [], [], []
for vid, result in video_results.items():
    avg_prob = np.mean(result['probs'])
    pred = int(avg_prob > 0.5)
    video_preds.append(pred)
    video_probs.append(avg_prob)
    video_labels.append(result['label'])

# Save predictions
results_df = pd.DataFrame({
    'video_id': list(video_results.keys()),
    'true_label': video_labels,
    'predicted_prob_fake': video_probs,
    'predicted_label': video_preds
})
results_df.to_csv("video_level_predictions.csv", index=False)

# Classification report
report = classification_report(video_labels, video_preds, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("video_level_classification_report.csv")

# Confusion matrix
cm = confusion_matrix(video_labels, video_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.title("Video-Level Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix_video_level.png")
plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(video_labels, video_probs)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("Video-Level ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve_video_level.png")
plt.close()

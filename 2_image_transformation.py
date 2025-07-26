import os
import torch
from PIL import Image
from torchvision import transforms
import tqdm

# Configuration
real_faces_dir = r"C:\Users\VIMS_Lab\Downloads\Project_Deepfake\DeepfakeProject\test\Realfaces_jpgs"
fake_faces_dir = r"c:\Users\VIMS_Lab\Downloads\Project_Deepfake\DeepfakeProject\test\faces_jpgs"
output_dir = r"c:\Users\VIMS_Lab\Downloads\Project_Deepfake\DeepfakeProject\test\sequence_tensors"
seq_len = 6  # number of images per sequence

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def save_sequences(input_dir, subfolder, label):
    save_path = os.path.join(output_dir, subfolder)
    os.makedirs(save_path, exist_ok=True)

    folders = sorted(os.listdir(input_dir))
    for folder in tqdm.tqdm(folders, desc=f"[{label}] Processing"):
        folder_path = os.path.join(input_dir, folder)
        images = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")])

        if len(images) < seq_len:
            continue  # skip too-short sequences

        for i in range(0, len(images) - seq_len + 1, seq_len):  # non-overlapping
            seq = []
            for j in range(i, i + seq_len):
                img_path = os.path.join(folder_path, images[j])
                img = Image.open(img_path).convert("RGB")
                seq.append(transform(img))

            seq_tensor = torch.stack(seq)  # [6, 3, 224, 224]
            save_name = f"{folder}_{i//seq_len}.pt"
            torch.save(seq_tensor, os.path.join(save_path, save_name))

# Process real and fake
save_sequences(real_faces_dir, "real", label="REAL")
save_sequences(fake_faces_dir, "fake", label="FAKE")

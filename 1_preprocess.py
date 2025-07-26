from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import os
import torch
import numpy as np

# Setup device and MTCNN face detector
device = 'cuda' if torch.cuda.is_available() else 'cpu'
detector = MTCNN(image_size=300, margin=20, device=device)

def extract_faces_from_video(video_path, output_path, frame_skip=5, max_frames=270):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return

    os.makedirs(output_path, exist_ok=True)
    count, saved = 0, 0

    while cap.isOpened() and saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_skip == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            face = detector(img)
            if face is not None and isinstance(face, torch.Tensor):
              face = face.permute(1, 2, 0).mul(255).clamp(0, 255).byte().cpu().numpy()
              if face.sum() > 1000:
                face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_path, f"{saved:03d}.jpg"), face_bgr)
                saved += 1



    cap.release()
    print(f"✅ Saved {saved} face frames from {os.path.basename(video_path)}")


def batch_process(deepfake_dir, original_dir, output_root, max_videos=None):
    print("🔁 Starting batch processing...")

    processed_orig_ids = set()
    count = 0

    # Process Deepfakes
    for file in sorted(os.listdir(deepfake_dir)):
        if not file.endswith(".mp4"):
            continue
        if max_videos is not None and count >= max_videos:
            break

        video_path = os.path.join(deepfake_dir, file)
        folder_name = os.path.splitext(file)[0]  # e.g., 123_456
        orig_id = folder_name.split("_")[0]      # e.g., 123
        processed_orig_ids.add(orig_id)

        output_path = os.path.join(output_root, "faces_jpgs", folder_name)
        extract_faces_from_video(video_path, output_path)
        count += 1

    # Process Originals only if corresponding deepfake exists
    count = 0
    for file in sorted(os.listdir(original_dir)):
        if not file.endswith(".mp4"):
            continue
        orig_id = os.path.splitext(file)[0]
        if orig_id in processed_orig_ids:
            if max_videos is not None and count >= max_videos:
                break
            video_path = os.path.join(original_dir, file)
            output_path = os.path.join(output_root, "Realfaces_jpgs", orig_id)
            extract_faces_from_video(video_path, output_path)
            count += 1

# ---------------------- USER CONFIG ----------------------
deepfake_dir = r"c:\Users\VIMS_Lab\Downloads\Project_Deepfake\output_folder\manipulated_sequences\Deepfakes\c23\videos"
original_dir = r"c:\Users\VIMS_Lab\Downloads\Project_Deepfake\output_folder\original_sequences\youtube\c23\videos"
output_root = r"c:\Users\VIMS_Lab\Downloads\Project_Deepfake\DeepfakeProject"

# ---------------------- RUN ----------------------
batch_process(deepfake_dir, original_dir, output_root, max_videos=200)

from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import os
import torch
import numpy as np

# Setup device and MTCNN face detector
device = 'cpu' #cuda' if torch.cuda.is_available() else 'cpu'
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


def batch_process(deepfake_dir, original_dir, output_root, start_idx=201, end_idx=240):
    print(f"🔁 Starting batch processing for videos {start_idx} to {end_idx}...")

    processed_orig_ids = set()

    # Get sorted list of deepfake video files
    deepfake_files = sorted([f for f in os.listdir(deepfake_dir) if f.endswith(".mp4")])
    selected_deepfake_files = deepfake_files[start_idx-1:end_idx]  # 0-based index

    # Process Deepfakes
    for file in selected_deepfake_files:
        video_path = os.path.join(deepfake_dir, file)
        folder_name = os.path.splitext(file)[0]  # e.g., 123_456
        orig_id = folder_name.split("_")[0]      # e.g., 123
        processed_orig_ids.add(orig_id)

        output_path = os.path.join(output_root, "faces_jpgs", folder_name)
        extract_faces_from_video(video_path, output_path)

    # Get sorted list of real/original video files
    original_files = sorted([f for f in os.listdir(original_dir) if f.endswith(".mp4")])

    # Process corresponding Originals
    for file in original_files:
        orig_id = os.path.splitext(file)[0]
        if orig_id in processed_orig_ids:
            video_path = os.path.join(original_dir, file)
            output_path = os.path.join(output_root, "Realfaces_jpgs", orig_id)
            extract_faces_from_video(video_path, output_path)


# ---------------------- USER CONFIG ----------------------
deepfake_dir = r"c:\Users\VIMS_Lab\Downloads\Project_Deepfake\output_folder\manipulated_sequences\Deepfakes\c23\videos"
original_dir = r"c:\Users\VIMS_Lab\Downloads\Project_Deepfake\output_folder\original_sequences\youtube\c23\videos"
output_root = r"c:\Users\VIMS_Lab\Downloads\Project_Deepfake\DeepfakeProject\test"
# ---------------------- RUN ----------------------
batch_process(deepfake_dir, original_dir, output_root, start_idx=201, end_idx=240)


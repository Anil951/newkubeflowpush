"""
datapreprocess.py
=================
Preprocesses the HumanAct12 dataset for the Action-Conditioned Motion Generator.

Pipeline:
  1. Scan HumanAct12/ folder; keep only the 4 target action classes.
  2. Load each .npy  (shape [T, 24, 3]).
  3. Project to 2D front-view  ->  use X (axis 0), flip Y (axis 1) for upright display.
  4. Map 24 SMPL-style joints -> 13 project keypoints (matching pose_extractor.py).
  5. Root-centre every clip  (hip midpoint -> origin) and normalise to [0, 1].
  6. Pad (repeat last frame) or truncate to exactly T = 64 frames.
  7. Save as  iofiles/dataset_preprocessed.npz
         keys: motions      [N, 64, 13, 2]  float32
               labels       [N]             int64
               action_names  array of str

Joint mapping: HumanAct12 (24 joints) -> project 13 keypoints
  Proj idx | Name        | HumanAct12 source joint(s)
  ---------|-------------|----------------------------
  0        | Face        | avg(12, 15)  <- neck(12), head(15)
  1        | L_Shoulder  | 13
  2        | R_Shoulder  | 14
  3        | L_Elbow     | 16
  4        | R_Elbow     | 17
  5        | L_Wrist     | 18
  6        | R_Wrist     | 19
  7        | L_Hip       | 1
  8        | R_Hip       | 2
  9        | L_Knee      | 4
  10       | R_Knee      | 5
  11       | L_Ankle     | 7
  12       | R_Ankle     | 8
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset", "HumanAct12")
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "iofiles", "dataset_preprocessed.npz"
)

# Target action codes and their integer labels (0-indexed)
TARGET_ACTIONS = {
    "0201": 0,   # walk
    "0301": 1,   # run
    "0402": 2,   # jump_vertical
    "1201": 3,   # throw_right_hand
}
ACTION_NAMES = ["walk", "run", "jump_vertical", "throw_right_hand"]
NUM_ACTIONS  = len(ACTION_NAMES)
SEQ_LEN      = 64   # fixed output sequence length
NUM_JOINTS   = 13


# ---------------------------------------------------------------------------
# Joint mapping helpers
# ---------------------------------------------------------------------------

def map_joints_24_to_13(pose_3d):
    """
    Map HumanAct12 24-joint skeleton to the project's 13-joint skeleton.

    Args:
        pose_3d: ndarray [T, 24, 3]
    Returns:
        ndarray [T, 13, 3]
    """
    T = pose_3d.shape[0]
    out = np.zeros((T, 13, 3), dtype=np.float32)
    out[:, 0,  :] = (pose_3d[:, 12, :] + pose_3d[:, 15, :]) / 2.0  # Face
    out[:, 1,  :] = pose_3d[:, 13, :]   # L_Shoulder
    out[:, 2,  :] = pose_3d[:, 14, :]   # R_Shoulder
    out[:, 3,  :] = pose_3d[:, 16, :]   # L_Elbow
    out[:, 4,  :] = pose_3d[:, 17, :]   # R_Elbow
    out[:, 5,  :] = pose_3d[:, 18, :]   # L_Wrist
    out[:, 6,  :] = pose_3d[:, 19, :]   # R_Wrist
    out[:, 7,  :] = pose_3d[:, 1,  :]   # L_Hip
    out[:, 8,  :] = pose_3d[:, 2,  :]   # R_Hip
    out[:, 9,  :] = pose_3d[:, 4,  :]   # L_Knee
    out[:, 10, :] = pose_3d[:, 5,  :]   # R_Knee
    out[:, 11, :] = pose_3d[:, 7,  :]   # L_Ankle
    out[:, 12, :] = pose_3d[:, 8,  :]   # R_Ankle
    return out


def project_to_front_view(joints_13):
    """
    Extract front-view (X, -Y) from 3D joints so the figure stands upright.

    Args:
        joints_13: [T, 13, 3]
    Returns:
        [T, 13, 2]  (x, y_flipped)
    """
    x =  joints_13[:, :, 0]   # horizontal
    y = -joints_13[:, :, 1]   # flip Y so head is at top
    return np.stack([x, y], axis=-1).astype(np.float32)


def normalise_clip(pose_2d):
    """
    Root-centre around the initial hip midpoint, then normalise scale using Torso length.
    This ensures that jumping sequences aren't shrunk compared to walking sequences.

    Args:
        pose_2d: [T, 13, 2]
    Returns:
        [T, 13, 2]  centered at (0,0) and scaled so torso length = 1.0
    """
    hip_mid = (pose_2d[0, 7, :] + pose_2d[0, 8, :]) / 2.0
    pose_2d = pose_2d - hip_mid[np.newaxis, np.newaxis, :]

    # Torso length in the first frame
    shoulder_mid = (pose_2d[0, 1, :] + pose_2d[0, 2, :]) / 2.0
    # Since we already subtracted hip_mid, hip_mid is now (0,0)
    torso_length = np.linalg.norm(shoulder_mid)
    
    if torso_length < 1e-6:
        torso_length = 1.0
        
    pose_2d = pose_2d / torso_length
    return pose_2d.astype(np.float32)


def pad_or_trim(pose_2d, target_len=SEQ_LEN):
    """Trim or pad (repeat last frame) to exactly target_len frames."""
    T = pose_2d.shape[0]
    if T >= target_len:
        return pose_2d[:target_len]
    pad = np.repeat(pose_2d[-1:], target_len - T, axis=0)
    return np.concatenate([pose_2d, pad], axis=0)


def get_action_code(filename):
    """
    Extract 4-digit action code from HumanAct12 filename.
    e.g. 'P01G01R01F0001T0064A0201.npy'  -> '0201'
    """
    base = os.path.splitext(filename)[0]
    a_idx = base.rfind('A')
    if a_idx == -1:
        return None
    code = base[a_idx + 1:]
    return code if len(code) == 4 else None


# ---------------------------------------------------------------------------
# Main preprocessing routine
# ---------------------------------------------------------------------------

def preprocess(dataset_dir=DATASET_DIR, output_path=OUTPUT_PATH):
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"HumanAct12 dataset not found at: {dataset_dir}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    motions_list = []
    labels_list  = []
    class_counts = {n: 0 for n in ACTION_NAMES}
    skipped = 0

    for fname in sorted(os.listdir(dataset_dir)):
        if not fname.endswith(".npy"):
            continue

        code = get_action_code(fname)
        if code not in TARGET_ACTIONS:
            skipped += 1
            continue

        label       = TARGET_ACTIONS[code]
        action_name = ACTION_NAMES[label]

        fpath = os.path.join(dataset_dir, fname)
        try:
            raw = np.load(fpath)   # [T, 24, 3]
        except Exception as e:
            print(f"  [WARN] Could not load {fname}: {e}")
            skipped += 1
            continue

        if raw.ndim != 3 or raw.shape[1] != 24 or raw.shape[2] != 3:
            print(f"  [WARN] Unexpected shape {raw.shape} in {fname}, skipping.")
            skipped += 1
            continue

        joints_13  = map_joints_24_to_13(raw)
        pose_2d    = project_to_front_view(joints_13)
        pose_norm  = normalise_clip(pose_2d)
        pose_fixed = pad_or_trim(pose_norm, SEQ_LEN)

        motions_list.append(pose_fixed)
        labels_list.append(label)
        class_counts[action_name] += 1

    if not motions_list:
        raise RuntimeError("No valid motion clips found. Check the dataset directory.")

    motions_arr = np.stack(motions_list, axis=0).astype(np.float32)  # [N, 64, 13, 2]
    labels_arr  = np.array(labels_list, dtype=np.int64)              # [N]

    np.savez(output_path,
             motions=motions_arr,
             labels=labels_arr,
             action_names=np.array(ACTION_NAMES))

    print("=" * 60)
    print(f"Preprocessing complete -> {output_path}")
    print(f"  Total clips  : {len(motions_list)}")
    print(f"  Skipped      : {skipped}  (other action classes)")
    print(f"  motions shape: {motions_arr.shape}")
    print("  Class counts :")
    for name, cnt in class_counts.items():
        print(f"    {name:<22} : {cnt}")
    print("=" * 60)
    return motions_arr, labels_arr


if __name__ == "__main__":
    preprocess()

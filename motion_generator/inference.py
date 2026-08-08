"""
inference.py
============
Inference pipeline for the Action-Conditioned Motion CVAE-GRU.

Given:
  - A starting keypoint pose  (from pose_extractor.py / normalized_keypoints.json)
  - An action label           (0=walk, 1=run, 2=jump_vertical, 3=throw_right_hand)

Outputs:
  - iofiles/generated_motion.npy       – [T, 13, 2] float32 array (normalised)
  - iofiles/skeleton_animation.mp4     – OpenCV video of the skeleton sequence

Usage:
    python -m motion_generator.inference                 # default: walk
    python -m motion_generator.inference --action run
    python -m motion_generator.inference --action jump_vertical --model iofiles/motion_cvae_best.pt
    python -m motion_generator.inference --action throw_right_hand --num_samples 3

Action label map:
    0 = walk           (A0201)
    1 = run            (A0301)
    2 = jump_vertical  (A0402)
    3 = throw_right_hand (A1201)
"""

import os
import sys
import json
import argparse

import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from motion_generator.datapreprocess import NUM_JOINTS, SEQ_LEN, ACTION_NAMES, NUM_ACTIONS
from motion_generator.dnn import MotionCVAE, P

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "iofiles", "motion_cvae_best.pt")
DEFAULT_KPTS_PATH  = os.path.join(PROJECT_ROOT, "iofiles", "normalized_keypoints.json")
DEFAULT_OUT_NPY    = os.path.join(PROJECT_ROOT, "iofiles", "generated_motion.npy")
DEFAULT_OUT_VID    = os.path.join(PROJECT_ROOT, "iofiles", "skeleton_animation.mp4")

# Matches pose_extractor.py NEW_SKELETON connections
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 2),           # Face to shoulders & shoulder line
    (1, 3), (3, 5), (2, 4), (4, 6),   # Arms
    (1, 7), (2, 8), (7, 8),            # Torso
    (7, 9), (9, 11), (8, 10), (10, 12) # Legs
]

ACTION_TO_IDX = {name: i for i, name in enumerate(ACTION_NAMES)}


# ---------------------------------------------------------------------------
# Load initial pose from JSON (output of pose_extractor.py)
# ---------------------------------------------------------------------------

def load_initial_pose(json_path=DEFAULT_KPTS_PATH):
    """
    Load the 13-keypoint normalised pose from pose_extractor.py output.

    Returns:
        Tensor [P]  (P = 26 = 13 joints * 2 coords, normalised [0,1])
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(
            f"Keypoints file not found: {json_path}\n"
            "Run pose_extractor.py first to extract the initial pose."
        )
    with open(json_path, "r") as f:
        data = json.load(f)

    kpts = data["keypoints"]    # list of [x, y] pairs, 13 points
    if len(kpts) != NUM_JOINTS:
        raise ValueError(
            f"Expected {NUM_JOINTS} keypoints in {json_path}, got {len(kpts)}."
        )

    flat = []
    for (x, y) in kpts:
        flat.extend([x, y])
    return torch.tensor(flat, dtype=torch.float32)    # [P]


# ---------------------------------------------------------------------------
# Load trained model
# ---------------------------------------------------------------------------

def load_model(model_path=DEFAULT_MODEL_PATH, device=None):
    """
    Load a trained MotionCVAE checkpoint.

    Returns:
        model  (MotionCVAE, eval mode, on device)
        config (dict)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}\n"
            "Run  python -m motion_generator.train  first."
        )

    ckpt   = torch.load(model_path, map_location=device)
    config = ckpt.get("config", {})

    model = MotionCVAE(
        latent_dim  = config.get("latent_dim",  128),
        hidden_dim  = config.get("hidden_dim",  256),
        enc_layers  = config.get("enc_layers",  1),
        dec_layers  = config.get("dec_layers",  2),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded model from: {model_path}  (epoch={ckpt.get('epoch', '?')})")
    return model, config, device


# ---------------------------------------------------------------------------
# Skeleton visualizer (OpenCV)
# ---------------------------------------------------------------------------

def draw_skeleton_frame(canvas, keypoints_xy, frame_w, frame_h,
                         joint_color=(0, 0, 255),
                         bone_color=(0, 255, 0),
                         joint_radius=6, bone_thickness=3):
    """
    Draw one skeleton frame onto a canvas.

    Args:
        canvas       : np.ndarray HxWx3 (uint8) – modified in-place
        keypoints_xy : np.ndarray [13, 2] in [0, 1] normalised space
        frame_w, frame_h : pixel dimensions
    """
    # Convert normalised [0,1] coords to pixel coords
    px_pts = []
    for (x, y) in keypoints_xy:
        px = int(x * (frame_w - 80) + 40)
        py = int(y * (frame_h - 80) + 40)
        px_pts.append((px, py))

    # Draw bones
    for (i, j) in SKELETON_CONNECTIONS:
        pt1, pt2 = px_pts[i], px_pts[j]
        cv2.line(canvas, pt1, pt2, bone_color, bone_thickness, lineType=cv2.LINE_AA)

    # Draw joints
    for k, pt in enumerate(px_pts):
        cv2.circle(canvas, pt, joint_radius, joint_color, -1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, str(k), (pt[0] + 7, pt[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    return canvas


def visualize_skeleton_sequence(
    motion_seq,         # np.ndarray [T, 13, 2] or [num_samples, T, 13, 2]
    output_path=DEFAULT_OUT_VID,
    fps=20,
    frame_w=600,
    frame_h=700,
    action_label="",
):
    """
    Write an OpenCV video of the skeleton animation.

    Args:
        motion_seq   : [T, 13, 2] or [S, T, 13, 2] – multiple samples tiled horizontally
        output_path  : .mp4 output path
        fps          : frames per second
        frame_w, frame_h : per-panel pixel size
        action_label : text annotation
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if motion_seq.ndim == 3:
        motion_seq = motion_seq[np.newaxis]   # [1, T, 13, 2]

    S, T_frames, J, _ = motion_seq.shape
    canvas_w = frame_w * S
    canvas_h = frame_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (canvas_w, canvas_h))

    for t in range(T_frames):
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        for s in range(S):
            panel = canvas[:, s * frame_w:(s + 1) * frame_w]
            kpts  = motion_seq[s, t]   # [13, 2]
            draw_skeleton_frame(panel, kpts, frame_w, frame_h)

            # Label overlay
            label_text = f"{action_label}  |  sample {s+1}  |  frame {t+1}/{T_frames}"
            cv2.putText(panel, label_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 200, 255), 1, cv2.LINE_AA)

        writer.write(canvas)

    writer.release()
    print(f"Skeleton animation saved -> {output_path}  ({S} sample(s), {T_frames} frames @ {fps} fps)")


# ---------------------------------------------------------------------------
# Skeletal Retargeting (Forward Kinematics)
# ---------------------------------------------------------------------------

def enforce_bone_lengths(motion_arr, initial_pose_flat):
    """
    Post-processing to strictly enforce initial bone lengths.
    motion_arr: [num_samples, T, 13, 2]
    initial_pose_flat: [26] tensor
    """
    import numpy as np
    S, T, J, _ = motion_arr.shape
    if hasattr(initial_pose_flat, "cpu"):
        init_pose = initial_pose_flat.cpu().numpy().reshape(13, 2)
    else:
        init_pose = np.array(initial_pose_flat).reshape(13, 2)
        
    tree = [
        (0, 1), (0, 2),          
        (1, 3), (3, 5),          
        (2, 4), (4, 6),          
        (1, 7), (7, 9), (9, 11), 
        (2, 8), (8, 10), (10, 12) 
    ]
    
    gt_lengths = {}
    for p, c in tree:
        gt_lengths[(p, c)] = np.linalg.norm(init_pose[c] - init_pose[p])
        
    fixed_arr = np.zeros_like(motion_arr)
    for s in range(S):
        for t in range(T):
            fixed_arr[s, t, 0] = motion_arr[s, t, 0] # Root follows prediction
            for p, c in tree:
                direction = motion_arr[s, t, c] - motion_arr[s, t, p]
                norm = np.linalg.norm(direction)
                if norm > 1e-5:
                    direction = direction / norm
                else:
                    direction = np.array([0.0, 1.0])
                fixed_arr[s, t, c] = fixed_arr[s, t, p] + direction * gt_lengths[(p, c)]
                
    return fixed_arr


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_motion(
    action           = "walk",
    model_path       = DEFAULT_MODEL_PATH,
    kpts_json_path   = DEFAULT_KPTS_PATH,
    out_npy_path     = DEFAULT_OUT_NPY,
    out_vid_path     = DEFAULT_OUT_VID,
    num_samples      = 1,
    temperature      = 1.0,
    fps              = 20,
):
    """
    Full inference pipeline: pose -> model -> keypoint sequence -> video.

    Args:
        action        : str  one of ACTION_NAMES or an int index
        model_path    : path to trained checkpoint
        kpts_json_path: path to normalized_keypoints.json (pose_extractor output)
        out_npy_path  : where to save the generated motion array
        out_vid_path  : where to save the skeleton animation video
        num_samples   : how many diverse sequences to generate
        temperature   : latent noise scale (1.0 = standard, <1 conservative)
        fps           : video frame rate

    Returns:
        motion_arr : np.ndarray [num_samples, T, 13, 2]
    """
    # Resolve action label
    if isinstance(action, str):
        if action not in ACTION_TO_IDX:
            valid = list(ACTION_TO_IDX.keys())
            raise ValueError(f"Unknown action '{action}'. Valid options: {valid}")
        label_idx = ACTION_TO_IDX[action]
        action_name = action
    else:
        label_idx   = int(action)
        action_name = ACTION_NAMES[label_idx]

    print(f"\n{'='*60}")
    print(f"Generating action  : '{action_name}'  (label={label_idx})")
    print(f"Num samples        : {num_samples}")
    print(f"Temperature        : {temperature}")
    print(f"{'='*60}")

    # 1. Load model
    model, cfg, device = load_model(model_path)

    # 2. Load initial pose
    initial_pose = load_initial_pose(kpts_json_path).to(device)   # [P]
    print(f"Initial pose loaded from: {kpts_json_path}")

    # 3. Generate
    with torch.no_grad():
        gen_tensor = model.generate(
            initial_pose_flat = initial_pose,
            label_idx         = label_idx,
            num_samples       = num_samples,
            temperature       = temperature,
        )  # [num_samples, T, 13, 2]

    motion_arr = gen_tensor.cpu().numpy()   # [num_samples, T, 13, 2]
    motion_arr = enforce_bone_lengths(motion_arr, initial_pose)
    print(f"Generated shape: {motion_arr.shape}")

    # 4. Save numpy array
    os.makedirs(os.path.dirname(out_npy_path), exist_ok=True)
    np.save(out_npy_path, motion_arr)
    print(f"Keypoint sequence saved -> {out_npy_path}")

    # 5. Render skeleton video
    visualize_skeleton_sequence(
        motion_arr,
        output_path  = out_vid_path,
        fps          = fps,
        action_label = action_name,
    )

    print(f"\nDone! Outputs:")
    print(f"  Keypoints  : {out_npy_path}")
    print(f"  Animation  : {out_vid_path}")
    print(f"{'='*60}\n")

    return motion_arr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motion Generator Inference")
    parser.add_argument("--action",      type=str, default="walk",
                        help=f"Action label: {ACTION_NAMES}")
    parser.add_argument("--model",       type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to trained checkpoint (.pt)")
    parser.add_argument("--keypoints",   type=str, default=DEFAULT_KPTS_PATH,
                        help="Path to normalized_keypoints.json (from pose_extractor.py)")
    parser.add_argument("--out_npy",     type=str, default=DEFAULT_OUT_NPY,
                        help="Output .npy path for generated sequence")
    parser.add_argument("--out_vid",     type=str, default=DEFAULT_OUT_VID,
                        help="Output .mp4 path for skeleton animation")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of diverse sequences to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Latent noise temperature (default=1.0)")
    parser.add_argument("--fps",         type=int, default=20,
                        help="Video frame rate (default=20)")
    args = parser.parse_args()

    generate_motion(
        action        = args.action,
        model_path    = args.model,
        kpts_json_path= args.keypoints,
        out_npy_path  = args.out_npy,
        out_vid_path  = args.out_vid,
        num_samples   = args.num_samples,
        temperature   = args.temperature,
        fps           = args.fps,
    )

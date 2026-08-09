"""
train.py
========
Training script for the Action-Conditioned Motion CVAE-GRU.

Usage:
    python -m motion_generator.train

Key features:
  - KL annealing warmup (weight 0 -> 1 over first KL_WARMUP_EPOCHS)
  - Best-model checkpoint saved to iofiles/motion_cvae_best.pt
  - Final model saved to iofiles/motion_cvae_final.pt
  - Loss history saved to iofiles/train_loss.json
  - Preprocess dataset automatically if .npz not found
"""

import os
import sys
import json
import time

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Allow running as  python -m motion_generator.train  from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from motion_generator.datapreprocess import preprocess, OUTPUT_PATH as NPZ_PATH
from motion_generator.dataloader import get_dataloaders
from motion_generator.dnn import MotionCVAE, cvae_loss

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

CONFIG = {
    # Model
    "latent_dim" : 128,
    "hidden_dim" : 256,
    "enc_layers" : 1,
    "dec_layers" : 2,

    # Training
    "epochs"         : 500,
    "batch_size"     : 32,
    "lr"             : 1e-3,
    "weight_decay"   : 1e-5,
    "val_split"      : 0.15,
    "seed"           : 42,

    # KL annealing: weight ramps from 0 to KL_MAX over KL_WARMUP_EPOCHS
    "kl_max"          : 0.01,
    "kl_warmup_epochs": 100,

    # Teacher forcing: decays from tf_start -> 0 over tf_decay_epochs
    "tf_start"        : 0.6,
    "tf_decay_epochs" : 300,

    # Paths
    "npz_path"         : NPZ_PATH,
    "checkpoint_dir"   : os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "iofiles"
    ),
    "best_model_name"  : "motion_cvae_best.pt",
    "final_model_name" : "motion_cvae_final.pt",
    "loss_log_name"    : "train_loss.json",

    # Log interval
    "log_every": 10,   # print every N epochs
}


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def kl_weight(epoch, kl_max, warmup_epochs):
    """Linear KL warmup schedule."""
    if warmup_epochs <= 0:
        return kl_max
    return min(kl_max, kl_max * epoch / warmup_epochs)


def tf_ratio_schedule(epoch, tf_start, decay_epochs):
    """Linear teacher forcing decay: tf_start -> 0 over decay_epochs."""
    if decay_epochs <= 0:
        return 0.0
    return max(0.0, tf_start * (1.0 - epoch / decay_epochs))


@torch.no_grad()
def validate(model, loader, device, kl_w):
    """Run one validation pass and return mean total, recon, KL losses."""
    model.eval()
    total_l, recon_l, kl_l = 0.0, 0.0, 0.0
    n = 0
    for motion, label in loader:
        motion = motion.to(device, non_blocking=True)
        label  = label.to(device,  non_blocking=True)
        seq_r, mu, logvar = model(motion, label, tf_ratio=0.0)
        tl, rl, kl = cvae_loss(seq_r, motion, mu, logvar, kl_weight=kl_w)
        total_l += tl.item()
        recon_l += rl.item()
        kl_l    += kl.item()
        n += 1
    return total_l / n, recon_l / n, kl_l / n


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg=None):
    if cfg is None:
        cfg = CONFIG

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*60}")
    print(f"Training on device : {device}")
    if device.type == "cuda":
        print(f"GPU                : {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}")

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    # ---- Dataset ----
    if not os.path.isfile(cfg["npz_path"]):
        print("Preprocessed dataset not found. Running datapreprocess.py ...")
        preprocess()

    train_loader, val_loader, full_ds = get_dataloaders(
        npz_path    = cfg["npz_path"],
        batch_size  = cfg["batch_size"],
        val_split   = cfg["val_split"],
        seed        = cfg["seed"],
    )

    # ---- Model ----
    model = MotionCVAE(
        latent_dim  = cfg["latent_dim"],
        hidden_dim  = cfg["hidden_dim"],
        enc_layers  = cfg["enc_layers"],
        dec_layers  = cfg["dec_layers"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters   : {total_params:,}")

    # ---- Optimiser + Scheduler ----
    optimizer = optim.Adam(model.parameters(),
                           lr           = cfg["lr"],
                           weight_decay = cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-5)

    # ---- Training state ----
    best_val_loss = float("inf")
    history = {"train_total": [], "train_recon": [], "train_kl": [],
               "val_total":   [], "val_recon":   [], "val_kl":   []}

    best_path  = os.path.join(cfg["checkpoint_dir"], cfg["best_model_name"])
    final_path = os.path.join(cfg["checkpoint_dir"], cfg["final_model_name"])
    log_path   = os.path.join(cfg["checkpoint_dir"], cfg["loss_log_name"])

    t0 = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        # KL weight and teacher forcing ratio for this epoch
        kl_w = kl_weight(epoch, cfg["kl_max"], cfg["kl_warmup_epochs"])
        tf_r = tf_ratio_schedule(epoch, cfg["tf_start"], cfg["tf_decay_epochs"])

        # ---- Train ----
        model.train()
        train_total, train_recon, train_kl = 0.0, 0.0, 0.0
        n_batches = 0

        for motion, label in train_loader:
            motion = motion.to(device, non_blocking=True)
            label  = label.to(device,  non_blocking=True)

            optimizer.zero_grad()
            seq_recon, mu, logvar = model(motion, label, tf_ratio=tf_r)
            loss, rl, kl = cvae_loss(seq_recon, motion, mu, logvar, kl_weight=kl_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_total += loss.item()
            train_recon += rl.item()
            train_kl    += kl.item()
            n_batches   += 1

        scheduler.step()

        train_total /= n_batches
        train_recon /= n_batches
        train_kl    /= n_batches

        # ---- Validate ----
        val_total, val_recon, val_kl = validate(model, val_loader, device, kl_w)

        # ---- Log ----
        history["train_total"].append(train_total)
        history["train_recon"].append(train_recon)
        history["train_kl"].append(train_kl)
        history["val_total"].append(val_total)
        history["val_recon"].append(val_recon)
        history["val_kl"].append(val_kl)

        if epoch % cfg["log_every"] == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:4d}/{cfg['epochs']} | "
                f"T={elapsed:7.1f}s | "
                f"KL_w={kl_w:.4f} | TF={tf_r:.3f} | "
                f"Train [tot={train_total:.4f} rec={train_recon:.4f} kl={train_kl:.4f}] | "
                f"Val   [tot={val_total:.4f} rec={val_recon:.4f} kl={val_kl:.4f}]"
            )

        # ---- Checkpoint: save best ----
        if val_total < best_val_loss:
            best_val_loss = val_total
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "val_loss"   : best_val_loss,
                "config"     : cfg,
            }, best_path)

    # ---- Save final model ----
    torch.save({
        "epoch"      : cfg["epochs"],
        "model_state": model.state_dict(),
        "config"     : cfg,
    }, final_path)

    # ---- Save loss history ----
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best val loss : {best_val_loss:.6f}")
    print(f"  Best model    : {best_path}")
    print(f"  Final model   : {final_path}")
    print(f"  Loss log      : {log_path}")
    print(f"{'='*60}")

    return model, history


if __name__ == "__main__":
    train()

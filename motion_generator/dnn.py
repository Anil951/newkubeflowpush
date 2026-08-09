"""
dnn.py
======
Action-Conditioned Motion Generator — CVAE-GRU Architecture.

Design lineage:
  - GenMotion / action2motion  :  GRU-based generator with action one-hot injection
                                  (MotionGenerator, GaussianGRU, DecoderGRU)
  - GenMotion / action_conditioned :  CVAE framing; encoder + decoder conditioning
  - monkey-net / PredictionModule  :  Step-wise GRU rollout seeded by an initial pose

Architecture Overview
---------------------
                 ┌──────────────────────────────┐
  initial_pose   │         PoseEncoder           │ (training only)
  + action_onehot│   Linear -> GRU -> mu/logvar  │ -> z ~ N(mu, sigma)
                 └──────────────────────────────┘
                            │  z
                            ▼
                 ┌──────────────────────────────┐
                 │       MotionDecoder           │
  z + action +   │   per-step GRUCell rollout   │ -> [T, 13, 2] sequence
  initial_pose   │   (T=64 auto-regressive)     │
                 └──────────────────────────────┘

At inference: z ~ N(0, 1)  (no encoder needed)

Shapes
------
  B  = batch size
  T  = 64  (fixed sequence length)
  J  = 13  (number of keypoints)
  A  = 4   (number of action classes)
  P  = J*2 = 26  (flattened initial pose)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from motion_generator.datapreprocess import NUM_ACTIONS, SEQ_LEN, NUM_JOINTS

# Convenience constants
A = NUM_ACTIONS   # 4
T = SEQ_LEN       # 64
J = NUM_JOINTS    # 13
P = J * 2         # 26  (flattened 2D pose)


# ---------------------------------------------------------------------------
# Helper: build a simple MLP
# ---------------------------------------------------------------------------

def _mlp(in_dim, hidden_dims, out_dim, activation=nn.LeakyReLU(0.2)):
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), activation]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# 1.  PoseEncoder  (CVAE encoder – used only during training)
# ---------------------------------------------------------------------------

class PoseEncoder(nn.Module):
    """
    Encodes the initial pose + action label into a latent distribution (mu, logvar).

    Inspired by:
      - GenMotion's GaussianGRU (motion_vae.py)
      - action_conditioned CVAE encoder

    Input  : initial_pose [B, P]  (flattened, normalised to [0,1])
             action_onehot [B, A]
    Output : mu [B, latent_dim], logvar [B, latent_dim]
    """

    def __init__(self, latent_dim=128, hidden_dim=256, n_layers=1):
        super().__init__()
        self.latent_dim = latent_dim

        # Project pose+action -> hidden space
        self.input_proj = nn.Sequential(
            nn.Linear(P + A, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
        )

        # GRU to capture temporal structure of the input
        # (here we treat the single initial pose as a sequence of length 1)
        self.gru = nn.GRU(
            input_size  = hidden_dim,
            hidden_size = hidden_dim,
            num_layers  = n_layers,
            batch_first = True,
        )

        # Gaussian heads
        self.mu_head     = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, motion_seq, action_onehot):
        """
        Args:
            motion_seq    : [B, T, J, 2]  (full sequence)
            action_onehot : [B, A]
        Returns:
            mu     : [B, latent_dim]
            logvar : [B, latent_dim]
        """
        B, T, J, _ = motion_seq.shape
        x = motion_seq.view(B, T, -1)  # [B, T, P]
        
        # Expand action to every frame
        action_expanded = action_onehot.unsqueeze(1).expand(B, T, -1)
        x = torch.cat([x, action_expanded], dim=-1)  # [B, T, P+A]
        
        x = self.input_proj(x)                       # [B, T, hidden]
        _, h = self.gru(x)                           # h: [num_layers, B, hidden]
        
        # Take the final hidden state of the last GRU layer
        h_final = h[-1]                              # [B, hidden]
        
        mu     = self.mu_head(h_final)
        logvar = self.logvar_head(h_final)
        return mu, logvar

    @staticmethod
    def reparameterise(mu, logvar):
        """Reparameterisation trick: z = mu + eps * sigma."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


# ---------------------------------------------------------------------------
# 2.  MotionDecoder  (CVAE decoder / generator)
# ---------------------------------------------------------------------------

class MotionDecoder(nn.Module):
    """
    Auto-regressive GRU decoder that generates a T-frame keypoint sequence.

    Inspired by:
      - monkey-net PredictionModule (step-wise GRU rollout seeded by initial pose)
      - GenMotion DecoderGRU / MotionGenerator (hidden-state rollout with action label)

    At each step t:
        input_t  = [prev_pose | z | action_onehot]
        h_t      = GRUCell(input_t, h_{t-1})
        pose_t   = Linear(h_t)  -> tanh -> [J, 2]

    The very first prev_pose is the initial pose from pose_extractor.py.

    Input:
        z             [B, latent_dim]
        initial_pose  [B, P]
        action_onehot [B, A]
    Output:
        seq           [B, T, J, 2]
    """

    def __init__(self, latent_dim=128, hidden_dim=256, n_layers=2, seq_len=T):
        super().__init__()
        self.seq_len    = seq_len
        self.n_layers   = n_layers
        self.hidden_dim = hidden_dim

        # Dimensionality of one decoder step's input
        self.step_input_dim = P + latent_dim + A  # prev_pose + z + action

        # Project step input to hidden space
        self.input_embed = nn.Sequential(
            nn.Linear(self.step_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
        )

        # Stack of GRUCells (from GenMotion DecoderGRU / motion_vae.py pattern)
        self.gru_cells = nn.ModuleList([
            nn.GRUCell(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])

        # Output head: hidden -> flattened pose [P]
        self.output_head = nn.Linear(hidden_dim, P)

    def _init_hidden(self, batch_size, device):
        """Initialise GRU hidden states with zeros."""
        return [
            torch.zeros(batch_size, self.hidden_dim, device=device)
            for _ in range(self.n_layers)
        ]

    def forward(self, z, initial_pose_flat, action_onehot, target_seq=None, tf_ratio=0.0):
        """
        Args:
            z                 : [B, latent_dim]
            initial_pose_flat : [B, P]
            action_onehot     : [B, A]
            target_seq        : [B, T, J, 2] (optional, for teacher forcing)
            tf_ratio          : float [0, 1] (probability of using ground truth)
        Returns:
            seq : [B, T, J, 2]
        """
        B      = z.shape[0]
        device = z.device
        hidden = self._init_hidden(B, device)

        prev_pose = initial_pose_flat   # [B, P]
        outputs   = []

        for t in range(self.seq_len):
            # Build step input
            step_in = torch.cat([prev_pose, z, action_onehot], dim=-1)  # [B, step_input_dim]
            h_in    = self.input_embed(step_in)                           # [B, hidden]

            # Multi-layer GRUCell forward pass
            h_new = hidden[0] = self.gru_cells[0](h_in, hidden[0])
            for i in range(1, self.n_layers):
                h_new = hidden[i] = self.gru_cells[i](h_new, hidden[i])

            # Output pose (Residual offset)
            pose_delta = self.output_head(h_new)      # [B, P]
            pose_flat  = prev_pose + pose_delta       # auto-regressive addition

            outputs.append(pose_flat.unsqueeze(1))   # [B, 1, P]
            
            # Teacher forcing
            if target_seq is not None and torch.rand(1).item() < tf_ratio:
                # Use ground-truth for the next step's input
                prev_pose = target_seq[:, t, :, :].reshape(B, -1)
            else:
                prev_pose = pose_flat

        seq_flat = torch.cat(outputs, dim=1)                  # [B, T, P]
        seq      = seq_flat.view(B, self.seq_len, J, 2)       # [B, T, J, 2]
        return seq


# ---------------------------------------------------------------------------
# 3.  MotionCVAE  (full model wrapper)
# ---------------------------------------------------------------------------

class MotionCVAE(nn.Module):
    """
    Full Conditional Variational Auto-Encoder for action-conditioned motion synthesis.

    Usage:
        model = MotionCVAE()

        # Training:
        seq_recon, mu, logvar = model(motion_seq, label)

        # Inference (generation):
        gen_seq = model.generate(initial_pose, label)  # -> [1, T, J, 2]
    """

    def __init__(
        self,
        latent_dim  = 128,
        hidden_dim  = 256,
        enc_layers  = 1,
        dec_layers  = 2,
        num_actions = A,
        seq_len     = T,
    ):
        super().__init__()
        self.latent_dim  = latent_dim
        self.num_actions = num_actions
        self.seq_len     = seq_len

        self.encoder = PoseEncoder(
            latent_dim = latent_dim,
            hidden_dim = hidden_dim,
            n_layers   = enc_layers,
        )
        self.decoder = MotionDecoder(
            latent_dim = latent_dim,
            hidden_dim = hidden_dim,
            n_layers   = dec_layers,
            seq_len    = seq_len,
        )

    # ------------------------------------------------------------------
    def _label_to_onehot(self, label):
        """Convert integer label tensor [B] to one-hot [B, A]."""
        B = label.shape[0]
        onehot = torch.zeros(B, self.num_actions, device=label.device)
        onehot.scatter_(1, label.unsqueeze(1), 1.0)
        return onehot

    # ------------------------------------------------------------------
    def forward(self, motion_seq, label, tf_ratio=0.5):
        """
        Training forward pass.

        Args:
            motion_seq : [B, T, J, 2]  – ground-truth sequence
            label      : [B]            – integer class index
            tf_ratio   : float          – teacher forcing ratio

        Returns:
            seq_recon  : [B, T, J, 2]  – reconstructed sequence
            mu         : [B, latent_dim]
            logvar     : [B, latent_dim]
        """
        B = motion_seq.shape[0]
        device = motion_seq.device

        # Use the first frame as the initial pose
        initial_pose_flat = motion_seq[:, 0, :, :].reshape(B, P)  # [B, P]
        action_onehot     = self._label_to_onehot(label)           # [B, A]

        # Encode full sequence -> latent distribution
        mu, logvar = self.encoder(motion_seq, action_onehot)

        # Sample z (reparameterisation trick)
        z = PoseEncoder.reparameterise(mu, logvar)                  # [B, latent_dim]

        # Decode
        seq_recon = self.decoder(z, initial_pose_flat, action_onehot, target_seq=motion_seq, tf_ratio=tf_ratio)

        return seq_recon, mu, logvar

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, initial_pose_flat, label_idx, num_samples=1, temperature=1.0):
        """
        Generate motion sequences at inference time.

        Args:
            initial_pose_flat : Tensor [P] or [1, P] or [B, P]  (normalised [0,1])
            label_idx         : int  – action class index (0=walk, 1=run, 2=jump, 3=throw)
            num_samples       : number of sequences to generate
            temperature       : scale on latent noise (>1 more diverse, <1 more conservative)

        Returns:
            seq : Tensor [num_samples, T, J, 2]
        """
        device = next(self.parameters()).device
        self.eval()

        # Handle varying input shapes
        if initial_pose_flat.dim() == 1:
            initial_pose_flat = initial_pose_flat.unsqueeze(0)  # [1, P]
        if initial_pose_flat.shape[0] == 1 and num_samples > 1:
            initial_pose_flat = initial_pose_flat.expand(num_samples, -1)

        initial_pose_flat = initial_pose_flat.to(device)

        label_t   = torch.tensor([label_idx] * num_samples, dtype=torch.long, device=device)
        action_oh = self._label_to_onehot(label_t)

        # Sample z from standard normal prior (scaled by temperature)
        z = torch.randn(num_samples, self.latent_dim, device=device) * temperature

        seq = self.decoder(z, initial_pose_flat, action_oh)  # [num_samples, T, J, 2]
        return seq


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

# Bone connections for bone-length consistency (matching skeleton topology)
BONE_PAIRS = [
    (0, 1), (0, 2),           # Face to shoulders
    (1, 3), (3, 5),           # Left arm
    (2, 4), (4, 6),           # Right arm
    (1, 7), (2, 8),           # Torso sides
    (7, 8),                    # Hip line
    (7, 9), (9, 11),          # Left leg
    (8, 10), (10, 12),        # Right leg
]


def compute_bone_lengths(seq):
    """Compute bone lengths for each frame.
    
    Args:
        seq: [B, T, J, 2]
    Returns:
        [B, T, num_bones] - length of each bone per frame
    """
    lengths = []
    for (i, j) in BONE_PAIRS:
        bone_vec = seq[:, :, i, :] - seq[:, :, j, :]        # [B, T, 2]
        bone_len = torch.norm(bone_vec, dim=-1)               # [B, T]
        lengths.append(bone_len)
    return torch.stack(lengths, dim=-1)                        # [B, T, num_bones]


def cvae_loss(seq_recon, seq_gt, mu, logvar, kl_weight=1.0, vel_weight=1.0, bone_weight=0.5):
    """
    CVAE loss = MPJPE + Velocity Loss + Bone Consistency Loss + KL divergence.

    Args:
        seq_recon   : [B, T, J, 2]  model output
        seq_gt      : [B, T, J, 2]  ground truth
        mu          : [B, latent_dim]
        logvar      : [B, latent_dim]
        kl_weight   : scalar weight on KL term
        vel_weight  : scalar weight on velocity (temporal smoothness) term
        bone_weight : scalar weight on bone-length consistency term

    Returns:
        total_loss, mpjpe_loss, kl_loss  (all scalar tensors)
    """
    # 1. MPJPE Reconstruction loss (Mean Per Joint Position Error)
    mpjpe_loss = torch.mean(torch.norm(seq_recon - seq_gt, dim=-1))

    # 2. Velocity loss (Temporal smoothness)
    vel_recon = seq_recon[:, 1:, :, :] - seq_recon[:, :-1, :, :]
    vel_gt    = seq_gt[:, 1:, :, :] - seq_gt[:, :-1, :, :]
    vel_loss  = torch.mean(torch.norm(vel_recon - vel_gt, dim=-1))

    # 3. Bone-length consistency loss
    # Penalizes bone lengths in reconstruction that differ from ground truth
    bones_recon = compute_bone_lengths(seq_recon)  # [B, T, num_bones]
    bones_gt    = compute_bone_lengths(seq_gt)      # [B, T, num_bones]
    bone_loss   = torch.mean((bones_recon - bones_gt) ** 2)

    # 4. KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = mpjpe_loss + vel_weight * vel_loss + bone_weight * bone_loss + kl_weight * kl_loss
    return total_loss, mpjpe_loss, kl_loss


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MotionCVAE(latent_dim=128, hidden_dim=256).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    B = 4
    motion_seq = torch.rand(B, T, J, 2, device=device)
    label      = torch.randint(0, A, (B,), device=device)

    # Training forward
    seq_recon, mu, logvar = model(motion_seq, label)
    loss, rl, kl = cvae_loss(seq_recon, motion_seq, mu, logvar, kl_weight=0.01)
    print(f"Train loss: {loss.item():.4f}  (recon={rl.item():.4f}, kl={kl.item():.4f})")

    # Inference
    init_pose = torch.rand(P, device=device)
    gen_seq   = model.generate(init_pose, label_idx=0, num_samples=2)
    print(f"Generated shape: {gen_seq.shape}")  # [2, 64, 13, 2]
    print("Sanity check passed.")

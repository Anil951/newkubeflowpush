"""
warping.py
==========
Action-Conditioned Character Animation via ARAP Mesh Deformation.

Design:
  - Inspired by AnimatedDrawings (Meta) ARAP approach, ported to be NumPy 2.x compatible.
  - Builds a Delaunay triangle mesh constrained to the character's silhouette.
  - Assigns each mesh triangle to its nearest skeleton joint via BFS (body-part ownership).
  - For each animation frame: ARAP solver repositions mesh vertices to match new joint locations.
  - Texture (source image pixels) is remapped using an OpenCV remap grid for speed.

Pipeline:
  1. Load character image (sketch.png) + initial keypoints (normalized_keypoints.json)
  2. Extract silhouette mask (threshold for sketch images)
  3. Build Delaunay mesh over silhouette
  4. BFS-assign each triangle to its nearest joint bone
  5. Init ARAP solver with source joint positions as pins
  6. Load generated_motion.npy  [S, T, 13, 2]
  7. For each frame:
       a. ARAP.solve(new_joint_positions) -> new vertex positions
       b. Build OpenCV remap (map_x, map_y) from old->new triangle transforms
       c. cv2.remap(source_image, ...) -> warped frame
       d. Write to video

Output:
  iofiles/warped_animation.mp4   — animated character video

Usage:
  python warping.py                          # animate with 'walk' (uses generated_motion.npy)
  python warping.py --action run             # specify action (also runs inference if needed)
  python warping.py --motion iofiles/my.npy  # use a pre-generated motion file
"""

import os
import sys
import json
import heapq
import argparse
import logging
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
from scipy.spatial import Delaunay
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from skimage import measure

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT  = os.path.dirname(__file__)
IOFILES       = os.path.join(PROJECT_ROOT, "iofiles")

DEFAULT_IMAGE  = os.path.join(IOFILES, "sketch.png")
DEFAULT_KPTS   = os.path.join(IOFILES, "normalized_keypoints.json")
DEFAULT_MOTION = os.path.join(IOFILES, "generated_motion.npy")
DEFAULT_OUTPUT = os.path.join(IOFILES, "warped_animation.mp4")

# Skeleton connections (same as pose_extractor.py and inference.py)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),              # Face to shoulders
    (1, 2),                      # Shoulder bar
    (1, 3), (3, 5),              # Left arm
    (2, 4), (4, 6),              # Right arm
    (1, 7), (2, 8),              # Torso sides
    (7, 8),                      # Hip bar
    (7, 9), (9, 11),             # Left leg
    (8, 10), (10, 12),           # Right leg
]

KEYPOINT_NAMES = [
    "Face", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
    "L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
]


# ===========================================================================
# Part 1: ARAP Solver (NumPy 2.x compatible port of AnimatedDrawings/arap.py)
# ===========================================================================

class ARAP:
    """
    As-Rigid-As-Possible mesh deformation.

    Based on:
      Igarashi & Igarashi, "Implementing As-Rigid-As-Possible Shape Manipulation"
      J. Graphics, GPU and Game Tools, 2009.

    Ported from AnimatedDrawings (Meta) arap.py to be NumPy 2.x compatible
    (removes deprecated numpy.bool8).

    Args:
        pins_xy   : ndarray [N, 2]  initial pin (control point) positions
        triangles : list of ndarray [3] vertex-id triplets
        vertices  : ndarray [V, 2]  initial vertex positions
        w         : int             weight for pin constraints (default 1000)
    """

    def __init__(self, pins_xy: np.ndarray, triangles: list, vertices: np.ndarray, w: int = 1000):
        self.w = w
        self.vertices = np.copy(vertices).astype(np.float32)

        # Build deduplicated edge list
        edge_set = set()
        for v0, v1, v2 in triangles:
            edge_set.add(tuple(sorted((int(v0), int(v1)))))
            edge_set.add(tuple(sorted((int(v1), int(v2)))))
            edge_set.add(tuple(sorted((int(v2), int(v0)))))
        self.e_v_idxs: List[Tuple[int, int]] = list(edge_set)

        # Edge vectors
        self.edge_vectors = np.array(
            [self.vertices[j] - self.vertices[i] for i, j in self.e_v_idxs],
            dtype=np.float32
        )  # [E, 2]

        # Compute barycentric coordinates of pins inside mesh
        pins_bc, self.pin_mask = self._xy_to_barycentric_coords(
            pins_xy.astype(np.float32), self.vertices, triangles
        )

        # Vertex neighbor map
        v_vnbr: Dict[int, set] = defaultdict(set)
        for v0, v1, v2 in triangles:
            v_vnbr[int(v0)] |= {int(v1), int(v2)}
            v_vnbr[int(v1)] |= {int(v2), int(v0)}
            v_vnbr[int(v2)] |= {int(v0), int(v1)}

        E = len(self.e_v_idxs)
        V = len(self.vertices)
        self.pin_num = int(self.pin_mask.sum())

        # Build A1: [2*(E+pin_num), 2*V]
        self.A1 = np.zeros([2 * (E + self.pin_num), 2 * V], dtype=np.float32)
        G_dense = np.zeros([2 * E, 2 * V], dtype=np.float32)

        for k, (vi, vj) in enumerate(self.e_v_idxs):
            self.A1[2*k:2*k+2, 2*vi:2*vi+2] = -np.eye(2)
            self.A1[2*k:2*k+2, 2*vj:2*vj+2] = np.eye(2)

            e_nbrs = list(v_vnbr[vi] & v_vnbr[vj])
            e_nbrs = [vi, vj] + e_nbrs

            e_verts = np.array([self.vertices[idx] for idx in e_nbrs], dtype=np.float32)

            rows = []
            for v in e_verts[1:]:
                dx = v[0] - e_verts[0][0]
                dy = v[1] - e_verts[0][1]
                rows.append([dx, dy])
                rows.append([dy, -dx])
            Gk = np.array(rows, dtype=np.float32)

            try:
                Gk_star = np.linalg.inv(Gk.T @ Gk) @ Gk.T
            except np.linalg.LinAlgError:
                Gk_star = np.linalg.pinv(Gk)

            ekx, eky = self.edge_vectors[k]
            e_mat = np.array([[ekx, eky], [eky, -ekx]], dtype=np.float32)

            n = len(e_nbrs)
            edge_mat = np.hstack([np.tile(-np.eye(2), (n-1, 1)), np.eye(2*(n-1))])
            g = Gk_star @ edge_mat
            h = e_mat @ g

            for h_off, v_idx in enumerate(e_nbrs):
                self.A1[2*k:2*k+2, 2*v_idx:2*v_idx+2] -= h[:, 2*h_off:2*h_off+2]
                G_dense[2*k:2*k+2, 2*v_idx:2*v_idx+2] = g[:, 2*h_off:2*h_off+2]

        # Pin rows in A1
        valid_pins = [bc for bc, m in zip(pins_bc, self.pin_mask) if m]
        for pi, pin_bc in enumerate(valid_pins):
            for v_idx, v_w in pin_bc:
                self.A1[2*E + 2*pi,   2*v_idx]   = self.w * v_w
                self.A1[2*E + 2*pi+1, 2*v_idx+1] = self.w * v_w

        # Build A2: [E+pin_num, V]
        A2_top = np.zeros([E, V], dtype=np.float32)
        for k, (vi, vj) in enumerate(self.e_v_idxs):
            A2_top[k, vi] = -1
            A2_top[k, vj] = 1
        A2_bot = np.zeros([self.pin_num, V], dtype=np.float32)
        for pi, pin_bc in enumerate(valid_pins):
            for v_idx, v_w in pin_bc:
                A2_bot[pi, v_idx] = self.w * v_w
        self.A2 = np.vstack([A2_top, A2_bot])

        # Cache as sparse
        self.tA1 = sp.csr_matrix(self.A1.T)
        self.tA2 = sp.csr_matrix(self.A2.T)
        self.G   = sp.csr_matrix(G_dense)

        # Cache (tA1 @ A1) and (tA2 @ A2) — perturb if singular
        def _safe_sparse(M):
            if hasattr(M, 'toarray'):
                M = M.toarray()
            M = M.astype(np.float64)
            while np.linalg.det(M) == 0.0:
                M += 1e-8 * np.eye(M.shape[0])
            return sp.csr_matrix(M)

        self.tA1xA1 = _safe_sparse((self.tA1 @ self.A1))
        self.tA2xA2 = _safe_sparse((self.tA2 @ self.A2))

        logging.info(f"ARAP init: V={V}  E={E}  pins={self.pin_num}")

    # ------------------------------------------------------------------
    def solve(self, pins_xy_new: np.ndarray) -> np.ndarray:
        """
        Given new pin positions, return updated vertex positions.

        Args:
            pins_xy_new : ndarray [N_pins, 2]  — must match init order
        Returns:
            ndarray [V, 2]  — new vertex positions
        """
        pins = pins_xy_new[self.pin_mask].astype(np.float64)
        E = len(self.e_v_idxs)

        b1 = np.hstack([np.zeros(2 * E), self.w * pins.ravel()])
        v1 = spla.spsolve(self.tA1xA1, self.tA1 @ b1)

        T1 = self.G @ v1
        b2_top = np.empty([E, 2], dtype=np.float64)
        for idx, e0 in enumerate(self.edge_vectors):
            c = T1[2*idx]; s = T1[2*idx+1]
            scale = 1.0 / max(np.sqrt(c*c + s*s), 1e-12)
            c *= scale; s *= scale
            R = np.array([[c, s], [-s, c]])
            b2_top[idx] = R @ e0

        b2 = np.vstack([b2_top, self.w * pins])
        v2x = spla.spsolve(self.tA2xA2, self.tA2 @ b2[:, 0])
        v2y = spla.spsolve(self.tA2xA2, self.tA2 @ b2[:, 1])
        return np.column_stack([v2x, v2y])

    # ------------------------------------------------------------------
    def _xy_to_barycentric_coords(self, points, vertices, triangles):
        tv = np.array([[vertices[t[0]], vertices[t[1]], vertices[t[2]]] for t in triangles],
                      dtype=np.float32)   # [T, 3, 2]

        v0 = tv[:, 0, :]          # [T, 2]
        v1 = tv[:, 1, :] - v0    # [T, 2]
        v2 = tv[:, 2, :] - v0    # [T, 2]

        bc_list = []
        mask = []

        def det2(u, v):
            return u[:, 0]*v[:, 1] - u[:, 1]*v[:, 0]

        for p in points:
            pv = p[np.newaxis] - v0   # [T, 2]
            denom = det2(v1, v2)
            a = (det2(pv, v2) / (denom + 1e-12))
            b = (-det2(pv, v1) / (denom + 1e-12))

            inside = (a > -1e-6) & (b > -1e-6) & (a + b < 1 + 1e-6)
            idxs = np.where(inside)[0]

            if len(idxs) == 0:
                mask.append(False)
                bc_list.append(None)
                logging.warning(f"Pin {p} outside mesh — will be skipped.")
                continue

            ti = idxs[0]
            verts = [int(triangles[ti][0]), int(triangles[ti][1]), int(triangles[ti][2])]
            a_xy, b_xy, c_xy = vertices[verts[0]], vertices[verts[1]], vertices[verts[2]]
            uvw = self._bary(p, a_xy, b_xy, c_xy)
            bc_list.append(list(zip(verts, uvw)))
            mask.append(True)

        return bc_list, np.array(mask, dtype=bool)

    def _bary(self, p, a, b, c):
        v0 = b - a; v1 = c - a; v2 = p - a
        d00 = v0 @ v0; d01 = v0 @ v1; d11 = v1 @ v1
        d20 = v2 @ v0; d21 = v2 @ v1
        denom = d00*d11 - d01*d01 + 1e-12
        bv = (d11*d20 - d01*d21) / denom
        bw = (d00*d21 - d01*d20) / denom
        bu = 1.0 - bv - bw
        return np.array([bu, bv, bw], dtype=np.float32)


# ===========================================================================
# Part 2: CharacterMesh — silhouette extraction + mesh + BFS joint assignment
# ===========================================================================

class CharacterMesh:
    """
    Builds and manages the deformable triangle mesh for a character image.

    Steps:
      1. Extract silhouette mask (threshold for sketch images)
      2. Find character contour
      3. Build Delaunay mesh constrained to silhouette interior
      4. BFS-assign each triangle to nearest joint bone
      5. Initialise ARAP with source keypoints as pins
    """

    # Joint bones: (parent_idx, child_idx) — defines the bone line segments
    # used for BFS distance computation
    BONES = [
        (0, 1), (0, 2),    # Face -> shoulders
        (1, 3), (3, 5),    # Left arm
        (2, 4), (4, 6),    # Right arm
        (1, 7), (2, 8),    # Torso
        (7, 9), (9, 11),   # Left leg
        (8, 10), (10, 12), # Right leg
    ]

    def __init__(self, image_path: str, keypoints_path: str, mesh_density: int = 30):
        """
        Args:
            image_path    : path to character sketch image
            keypoints_path: path to normalized_keypoints.json
            mesh_density  : number of interior points along each axis for mesh generation
        """
        self.img_path = image_path
        self.mesh_density = mesh_density

        # Load image
        self.src_img = cv2.imread(image_path)
        if self.src_img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        self.H, self.W = self.src_img.shape[:2]
        logging.info(f"Image loaded: {self.W}x{self.H}")

        # Load keypoints  [13, 2]  normalised [0,1]
        with open(keypoints_path, "r") as f:
            kpts_data = json.load(f)
        self.kpts_norm = np.array(kpts_data["keypoints"], dtype=np.float32)  # [13, 2]
        logging.info(f"Keypoints loaded: {len(self.kpts_norm)} joints")

        # Step 1: silhouette mask
        self.mask = self._extract_mask()

        # Step 2 & 3: build Delaunay mesh
        self.vertices, self.triangles = self._build_mesh()
        logging.info(f"Mesh built: {len(self.vertices)} vertices, {len(self.triangles)} triangles")

        # Step 4: BFS joint assignment
        self.tri_to_joint: np.ndarray = self._bfs_assign_triangles()

        # Step 5: init ARAP
        # Convert normalised keypoints to pixel space: x->W, y->H (flip y for image coords)
        pins_px = self._norm_to_px(self.kpts_norm)   # [13, 2]
        # Vertices are also in pixel space
        self.arap = ARAP(pins_px, self.triangles, self.vertices, w=1000)

    # ------------------------------------------------------------------
    def _extract_mask(self) -> np.ndarray:
        """
        Extract binary silhouette mask from sketch image.
        Sketch: dark lines/fill on white background -> invert and threshold.
        Returns binary mask [H, W] uint8 (255 = character, 0 = background).
        """
        gray = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2GRAY)
        # Threshold: pixels darker than 200 belong to the character
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        # Morphological close to fill small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # Keep only the largest connected component
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n_labels > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = ((labels == largest) * 255).astype(np.uint8)
        logging.info(f"Mask: {np.count_nonzero(mask)} foreground pixels")
        return mask

    def _norm_to_px(self, kpts_norm: np.ndarray) -> np.ndarray:
        """
        Convert normalised [0,1] keypoints to pixel coords.
        x -> col = x * W,  y -> row = (1 - y) * H  (flip y: 0=bottom, 1=top in normalised space)
        """
        px = np.zeros_like(kpts_norm)
        px[:, 0] = kpts_norm[:, 0] * self.W          # col
        px[:, 1] = (1.0 - kpts_norm[:, 1]) * self.H  # row (flipped)
        return px

    # ------------------------------------------------------------------
    def _build_mesh(self) -> Tuple[np.ndarray, list]:
        """
        Build Delaunay triangle mesh constrained to the character silhouette.

        Approach (from AnimatedDrawings._generate_mesh):
          1. Find contour boundary points
          2. Add interior grid points that lie inside contour
          3. Delaunay triangulate all points
          4. Keep only triangles whose centroid lies inside the contour

        Returns:
            vertices  : ndarray [V, 2]  pixel coords (col, row)
            triangles : list of ndarray [3]
        """
        # Contour boundary (skimage gives (row, col) — convert to (col, row))
        contours = measure.find_contours(self.mask, 128)
        if not contours:
            raise RuntimeError("No contour found in mask.")
        contours.sort(key=len, reverse=True)
        contour_rc = contours[0]  # [N, 2]  (row, col)

        # Simplify contour to reduce vertex count while preserving shape
        from skimage.measure import approximate_polygon
        contour_rc = approximate_polygon(contour_rc, tolerance=1.5)
        boundary_pts = contour_rc[:, ::-1].astype(np.float32)  # (col, row)

        # Interior points: grid inside the mask
        cols = np.linspace(0, self.W, self.mesh_density)
        rows = np.linspace(0, self.H, self.mesh_density)
        cc, rr = np.meshgrid(cols, rows)
        grid_pts = np.column_stack([cc.ravel(), rr.ravel()]).astype(np.float32)

        # Keep only interior grid points (mask > 0 at that pixel)
        r_idx = np.clip(grid_pts[:, 1].astype(int), 0, self.H - 1)
        c_idx = np.clip(grid_pts[:, 0].astype(int), 0, self.W - 1)
        inside = self.mask[r_idx, c_idx] > 0
        interior_pts = grid_pts[inside]

        # Combine boundary + interior points
        all_pts = np.vstack([boundary_pts, interior_pts])

        # Delaunay triangulation
        tri = Delaunay(all_pts)

        # Build a rough polygon from the contour for centroid containment test
        # Use OpenCV pointPolygonTest (fast, no shapely needed)
        contour_px = (contour_rc[:, ::-1]).astype(np.float32)  # (col, row) for cv2

        def is_inside_contour(cx, cy):
            """Return True if point (cx=col, cy=row) is inside the character contour."""
            pt = (float(cx), float(cy))
            # contour for cv2.pointPolygonTest must be (col, row) formatted
            return cv2.pointPolygonTest(contour_px.astype(np.int32).reshape(-1, 1, 2), pt, False) >= 0

        kept_triangles = []
        for simplex in tri.simplices:
            tri_verts = all_pts[simplex]
            cx, cy = tri_verts.mean(axis=0)
            if is_inside_contour(cx, cy):
                kept_triangles.append(simplex.tolist())

        logging.info(f"Mesh: kept {len(kept_triangles)}/{len(tri.simplices)} triangles inside silhouette")
        return all_pts, kept_triangles

    # ------------------------------------------------------------------
    def _bfs_assign_triangles(self) -> np.ndarray:
        """
        BFS from each joint bone to assign each triangle to its nearest joint.
        Uses pixel-level BFS (as in AnimatedDrawings).

        Returns:
            tri_to_joint : ndarray [T] int — joint index for each triangle
        """
        H, W = self.H, self.W
        INF = 1 << 20

        # Convert normalised keypoints to pixel (col, row)
        kpts_px = self._norm_to_px(self.kpts_norm)   # [13, 2] (col, row)

        # Distance map + closest joint map
        dist_map  = np.full((H, W), INF, dtype=np.int32)
        joint_map = np.full((H, W), -1, dtype=np.int8)

        # Seed: 20 points along each bone segment
        heap = []
        for (ji, jj) in self.BONES:
            p1 = kpts_px[ji]   # (col, row)
            p2 = kpts_px[jj]   # (col, row)
            for alpha in np.linspace(0, 1, 20, endpoint=False):
                col = int(round(p1[0] * (1 - alpha) + p2[0] * alpha))
                row = int(round(p1[1] * (1 - alpha) + p2[1] * alpha))
                if 0 <= col < W and 0 <= row < H:
                    heapq.heappush(heap, (0, ji, row, col))

        # BFS over mask pixels
        directions = [
            (-1, -1, 1.414), (0, -1, 1.0), (1, -1, 1.414),
            (-1,  0, 1.0),                  (1,  0, 1.0),
            (-1,  1, 1.414), (0,  1, 1.0),  (1,  1, 1.414),
        ]
        while heap:
            d, ji, row, col = heapq.heappop(heap)
            if d >= dist_map[row, col]:
                continue
            if self.mask[row, col] == 0:
                continue
            dist_map[row, col] = d
            joint_map[row, col] = ji
            for dc, dr, nd in directions:
                nr, nc = row + dr, col + dc
                new_d = d + nd
                if 0 <= nr < H and 0 <= nc < W and new_d < dist_map[nr, nc]:
                    heapq.heappush(heap, (new_d, ji, nr, nc))

        # Assign each triangle by centroid's nearest joint
        tri_to_joint = np.zeros(len(self.triangles), dtype=np.int32)
        for ti, simplex in enumerate(self.triangles):
            verts = self.vertices[simplex]           # [3, 2] (col, row)
            cx, cy = verts.mean(axis=0)
            col_i = int(np.clip(round(cx), 0, W - 1))
            row_i = int(np.clip(round(cy), 0, H - 1))
            j = int(joint_map[row_i, col_i])
            tri_to_joint[ti] = max(j, 0)

        return tri_to_joint


# ===========================================================================
# Part 3: WarpingEngine — per-frame ARAP solve + OpenCV remap
# ===========================================================================

class WarpingEngine:
    """
    Generates warped animation frames using ARAP mesh deformation.

    For each frame:
      1. ARAP.solve(new_joint_pixel_positions) -> new vertex positions
      2. Build remap grid: for each output pixel (r, c), find which new triangle
         contains it, compute barycentric coords, map back to source UV
      3. cv2.remap(source_image) -> warped frame
    """

    def __init__(self, mesh: CharacterMesh):
        self.mesh = mesh
        self.H = mesh.H
        self.W = mesh.W

        # Precompute source UV for remap (used as identity initially)
        self._src_img_float = mesh.src_img.astype(np.float32)

    def warp_frame(self, new_joints_norm: np.ndarray) -> np.ndarray:
        """
        Generate one warped frame given new joint positions.

        Args:
            new_joints_norm : ndarray [13, 2]  normalised [0,1] joint positions
        Returns:
            warped : ndarray [H, W, 3] uint8
        """
        m = self.mesh

        # Convert new joints to pixel space
        new_pins_px = m._norm_to_px(new_joints_norm)  # [13, 2] (col, row)

        # Solve ARAP
        new_verts = m.arap.solve(new_pins_px).astype(np.float32)  # [V, 2] (col, row)

        # Build remap from new triangulation back to source
        # map_x[r,c] = source column to sample
        # map_y[r,c] = source row to sample
        map_x = np.full((self.H, self.W), -1.0, dtype=np.float32)
        map_y = np.full((self.H, self.W), -1.0, dtype=np.float32)

        for ti, simplex in enumerate(m.triangles):
            # New (deformed) triangle vertices in pixel space
            new_tri = new_verts[simplex]     # [3, 2] (col, row)
            src_tri = m.vertices[simplex]    # [3, 2] (col, row)  -- original

            # Get bounding box of new triangle in output space
            c_min = max(0, int(new_tri[:, 0].min()) - 1)
            c_max = min(self.W - 1, int(new_tri[:, 0].max()) + 1)
            r_min = max(0, int(new_tri[:, 1].min()) - 1)
            r_max = min(self.H - 1, int(new_tri[:, 1].max()) + 1)

            if c_max < c_min or r_max < r_min:
                continue

            # For all pixels in bounding box, compute barycentric in new_tri
            cc, rr = np.meshgrid(
                np.arange(c_min, c_max + 1),
                np.arange(r_min, r_max + 1)
            )  # both shape [rows, cols]

            pts = np.column_stack([cc.ravel(), rr.ravel()]).astype(np.float32)  # [N, 2]

            # Barycentric coords in new_tri (col, row convention)
            A = new_tri[0]; B = new_tri[1]; C = new_tri[2]
            T = np.array([[B[0]-A[0], C[0]-A[0]],
                          [B[1]-A[1], C[1]-A[1]]], dtype=np.float32)
            try:
                T_inv = np.linalg.inv(T)
            except np.linalg.LinAlgError:
                continue

            diff = (pts - A).T     # [2, N]
            bc = T_inv @ diff      # [2, N]  (lambda1, lambda2)
            lam1, lam2 = bc[0], bc[1]
            lam0 = 1.0 - lam1 - lam2

            inside = (lam0 >= -1e-4) & (lam1 >= -1e-4) & (lam2 >= -1e-4)
            if not inside.any():
                continue

            # Map to source using same barycentric coords in src_tri
            src_pts = (lam0[:, None] * src_tri[0] +
                       lam1[:, None] * src_tri[1] +
                       lam2[:, None] * src_tri[2])   # [N, 2] (col, row)

            out_cols = cc.ravel()[inside].astype(int)
            out_rows = rr.ravel()[inside].astype(int)

            valid_cols = np.clip(out_cols, 0, self.W - 1)
            valid_rows = np.clip(out_rows, 0, self.H - 1)

            map_x[valid_rows, valid_cols] = src_pts[inside, 0]  # source col
            map_y[valid_rows, valid_cols] = src_pts[inside, 1]  # source row

        # Fill pixels with no triangle (outside character) with white
        no_cover = map_x < 0
        map_x = np.where(no_cover, 0.0, map_x)
        map_y = np.where(no_cover, 0.0, map_y)

        warped = cv2.remap(self.mesh.src_img, map_x, map_y,
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(255, 255, 255))

        # White-out pixels that had no triangle coverage
        warped[no_cover] = [255, 255, 255]

        return warped


# ===========================================================================
# Part 4: animate() — full pipeline
# ===========================================================================

def draw_skeleton_overlay(frame: np.ndarray, kpts_norm: np.ndarray,
                           H: int, W: int) -> np.ndarray:
    """Draw skeleton overlay on a frame."""
    out = frame.copy()
    kpts_px = []
    for (x, y) in kpts_norm:
        col = int(x * W)
        row = int((1.0 - y) * H)
        kpts_px.append((col, row))

    for (i, j) in SKELETON_CONNECTIONS:
        cv2.line(out, kpts_px[i], kpts_px[j], (0, 200, 0), 2, cv2.LINE_AA)
    for k, pt in enumerate(kpts_px):
        cv2.circle(out, pt, 5, (0, 0, 255), -1, cv2.LINE_AA)
    return out


def animate(
    image_path   : str = DEFAULT_IMAGE,
    keypoints_path: str = DEFAULT_KPTS,
    motion_path  : str = DEFAULT_MOTION,
    output_path  : str = DEFAULT_OUTPUT,
    sample_idx   : int = 0,
    fps          : int = 20,
    draw_skeleton: bool = True,
    mesh_density : int = 30,
):
    """
    Full animation pipeline.

    Args:
        image_path     : character sketch image
        keypoints_path : normalized_keypoints.json (from pose_extractor.py)
        motion_path    : generated_motion.npy [S, T, 13, 2]
        output_path    : output video path
        sample_idx     : which generated sample to animate (0-indexed)
        fps            : output video frame rate
        draw_skeleton  : whether to overlay skeleton on output frames
        mesh_density   : grid density for mesh interior points (higher = denser mesh)
    """
    logging.info("=" * 60)
    logging.info("Character Animation Pipeline")
    logging.info("=" * 60)

    # Load motion
    if not os.path.isfile(motion_path):
        raise FileNotFoundError(
            f"Motion file not found: {motion_path}\n"
            "Run  python -m motion_generator.inference  first."
        )
    motion = np.load(motion_path)   # [S, T, 13, 2] or [T, 13, 2]
    if motion.ndim == 3:
        motion = motion[np.newaxis]   # [1, T, 13, 2]
    S, T, J, _ = motion.shape
    logging.info(f"Motion loaded: {S} sample(s), {T} frames, {J} joints")

    if sample_idx >= S:
        logging.warning(f"sample_idx {sample_idx} >= {S} samples; using 0.")
        sample_idx = 0
    seq = motion[sample_idx]   # [T, 13, 2]

    # Build character mesh
    logging.info("Building character mesh...")
    mesh = CharacterMesh(image_path, keypoints_path, mesh_density=mesh_density)

    # Warping engine
    engine = WarpingEngine(mesh)

    # Output video
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (mesh.W, mesh.H))

    logging.info(f"Rendering {T} frames at {fps}fps -> {output_path}")

    for t in range(T):
        kpts_t = seq[t]   # [13, 2]  normalised

        frame = engine.warp_frame(kpts_t)

        if draw_skeleton:
            frame = draw_skeleton_overlay(frame, kpts_t, mesh.H, mesh.W)

        # Frame counter
        cv2.putText(frame, f"frame {t+1}/{T}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 200), 1, cv2.LINE_AA)

        writer.write(frame)

        if (t + 1) % 10 == 0:
            logging.info(f"  Frame {t+1}/{T}")

    writer.release()
    logging.info(f"Animation complete -> {output_path}")
    return output_path


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARAP Character Animation")
    parser.add_argument("--image",    type=str, default=DEFAULT_IMAGE,
                        help="Input character image path")
    parser.add_argument("--keypoints",type=str, default=DEFAULT_KPTS,
                        help="Normalized keypoints JSON path")
    parser.add_argument("--motion",   type=str, default=DEFAULT_MOTION,
                        help="Generated motion .npy path  [S, T, 13, 2]")
    parser.add_argument("--output",   type=str, default=DEFAULT_OUTPUT,
                        help="Output video path (.mp4)")
    parser.add_argument("--sample",   type=int, default=0,
                        help="Which generated sample to use (0-indexed)")
    parser.add_argument("--fps",      type=int, default=20,
                        help="Output video frame rate")
    parser.add_argument("--no_skeleton", action="store_true",
                        help="Disable skeleton overlay")
    parser.add_argument("--mesh_density", type=int, default=30,
                        help="Mesh interior grid density (higher = denser mesh)")
    args = parser.parse_args()

    animate(
        image_path    = args.image,
        keypoints_path= args.keypoints,
        motion_path   = args.motion,
        output_path   = args.output,
        sample_idx    = args.sample,
        fps           = args.fps,
        draw_skeleton = not args.no_skeleton,
        mesh_density  = args.mesh_density,
    )

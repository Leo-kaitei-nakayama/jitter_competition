"""
Data bridge between the starter competition's data layout and the ASDN model.

Train data:  <root>/shapenet/<synset>/<model_id>/models/model_normalized.obj
             listed (as 'shapenet/<synset>/<model_id>') in datalist/train.txt
Predict data:<root>/shapenet/<synset>/<model_id>/noisy.npy   (N,3) float
             listed in datalist/test.txt

The training pipeline mirrors the starter's transforms:
  sample surface -> normalize to unit sphere -> add Laplacian noise ->
  cut ONE patch of `patch_size` points around a random seed.

That patch (noisy + clean + seed) is exactly what
DenoiseNetCD.get_supervised_loss consumes.
"""

import os
import numpy as np
import math
import trimesh
from scipy.spatial import cKDTree
from jittor.dataset import Dataset


def _read_datalist(path):
    with open(path, 'r') as f:
        return [ln.strip() for ln in f if ln.strip()]


def normalize_unit_sphere(pc):
    p_max = pc.max(axis=0)
    p_min = pc.min(axis=0)
    center = (p_max + p_min) / 2
    pc = pc - center
    scale = np.sqrt((pc ** 2).sum(axis=1).max())
    return (pc / scale).astype(np.float32), center.astype(np.float32), np.float32(scale)


def sample_mesh_surface(path, num_samples, return_normals=False):
    """Sample points on the mesh surface. With return_normals, also return the
    true face normal at each sample -- sample_surface already hands back the
    face index, so this costs one gather and no estimation. That matters:
    an estimated normal is unreliable exactly at sharp features, while the
    face normal is exact everywhere."""
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    pts, face_idx = trimesh.sample.sample_surface(mesh, num_samples)
    pts = np.asarray(pts, dtype=np.float32)
    if not return_normals:
        return pts
    nrm = np.asarray(mesh.face_normals[face_idx], dtype=np.float32)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    return pts, nrm


def sample_noise(shape, noise_std, dist='laplace', rng=None):
    """
    Draw additive noise whose STANDARD DEVIATION is `noise_std`.

    The competition specifies its test noise by standard deviation
    (0.005 ~ 0.020 after unit-sphere normalization), and the diffusion schedule
    is indexed by sigma_bar, also a standard deviation. numpy's Laplace is
    parameterized by its *scale* b, and Laplace(0, b) has std b*sqrt(2) -- so
    passing noise_std straight in as the scale produced noise 1.41x stronger
    than both the label attached to it and the competition's own range.
    Dividing by sqrt(2) makes `noise_std` mean what it says.

    Args:
        shape: output shape
        noise_std: target standard deviation
        dist: 'laplace' (heavier tails, what this pipeline has always used) or
              'gaussian' (what the diffusion schedule assumes)
    """
    rng = np.random if rng is None else rng
    if dist == 'gaussian':
        return rng.normal(0.0, noise_std, size=shape).astype(np.float32)
    if dist == 'laplace':
        return rng.laplace(0.0, noise_std / math.sqrt(2.0), size=shape).astype(np.float32)
    raise ValueError(f'unknown noise distribution: {dist!r}')


def _random_euler_rotation_matrix(x_range, y_range, z_range):
    """Same convention as the starter's AugmentLinear: independent random
    Euler angles per axis, combined as Rz @ Ry @ Rx (radians)."""
    ax = math.radians(np.random.uniform(*x_range))
    ay = math.radians(np.random.uniform(*y_range))
    az = math.radians(np.random.uniform(*z_range))

    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)

    return (Rz @ Ry @ Rx).astype(np.float32)


def random_linear_augment(pc, scale_range=(0.8, 1.2),
                           rotate_x_range=(-180, 180),
                           rotate_y_range=(-180, 180),
                           rotate_z_range=(-180, 180),
                           scale_p=0.5, rotate_p=0.5, normals=None):
    """Same augmentation as the starter's AugmentLinear: optional random
    rotation, optional random uniform scale. Applied to a (N,3) point cloud.

    With `normals`, the same transform is carried to them -- but only the
    rotation touches a normal: a uniform scale (like the unit-sphere
    normalisation elsewhere) leaves directions unchanged, and a translation
    cannot affect them at all."""
    if np.random.rand() < rotate_p:
        R = _random_euler_rotation_matrix(rotate_x_range, rotate_y_range, rotate_z_range)
        pc = pc @ R.T
        if normals is not None:
            normals = normals @ R.T
    if np.random.rand() < scale_p:
        s = np.random.uniform(scale_range[0], scale_range[1])
        pc = pc * np.float32(s)
    pc = pc.astype(np.float32)
    if normals is None:
        return pc
    return pc, normals.astype(np.float32)


class ShapeNetPatchTrainDataset(Dataset):
    """Yields one (noisy, clean, seed) patch per __getitem__, ASDN-loss ready."""

    def __init__(self, root, datalist, num_samples=32768, patch_size=1000,
                 noise_min=0.005, noise_max=0.02, batch_size=8, shuffle=True,
                 num_workers=4, mesh_name='models/model_normalized.obj',
                 noise_dist='laplace', with_normals=False):
        super().__init__()
        self.root = root
        self.entries = _read_datalist(datalist)
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.noise_dist = noise_dist
        self.mesh_name = mesh_name
        self.with_normals = with_normals
        self.set_attrs(total_len=len(self.entries), batch_size=batch_size,
                       shuffle=shuffle, num_workers=num_workers)


    def __getitem__(self, idx):
        rel = self.entries[idx % len(self.entries)]
        mesh_path = os.path.join(self.root, rel, self.mesh_name)

        if self.with_normals:
            pc, nrm = sample_mesh_surface(mesh_path, self.num_samples,
                                          return_normals=True)
            pc, _, _ = normalize_unit_sphere(pc)   # translate + uniform scale: normals unaffected
            pc, nrm = random_linear_augment(pc, normals=nrm)
        else:
            nrm = None
            pc = sample_mesh_surface(mesh_path, self.num_samples)
            pc, _, _ = normalize_unit_sphere(pc)
            pc = random_linear_augment(pc)  # rotation + scale augmentation

        noise_std = np.random.uniform(self.noise_min, self.noise_max)
        noise = sample_noise(pc.shape, noise_std, self.noise_dist)
        pc_noisy = pc + noise

        # one patch around a random seed
        N = pc_noisy.shape[0]
        seed_i = np.random.randint(N)
        seed = pc_noisy[seed_i]
        tree = cKDTree(pc_noisy)
        _, nn_idx = tree.query(seed[None, :], k=self.patch_size)
        nn_idx = nn_idx[0]

        pat_noisy = pc_noisy[nn_idx].astype(np.float32)   # (M, 3)
        pat_clean = pc[nn_idx].astype(np.float32)         # (M, 3)

        out = {
            'pcl_noisy': pat_noisy,
            'pcl_clean': pat_clean,
            'seed_pnts': seed[None, :].astype(np.float32),  # (1, 3)
            'pcl_std': np.array([noise_std], dtype=np.float32),
        }
        if nrm is not None:
            out['pcl_normals'] = nrm[nn_idx].astype(np.float32)   # (M, 3), true face normals
        return out


class ShapeNetNoisyPredictDataset(Dataset):
    """Yields the full noisy cloud + its relative path, for submission writing."""

    def __init__(self, root, datalist, data_name='noisy.npy',
                 batch_size=1, num_workers=4):
        super().__init__()
        self.root = root
        self.entries = _read_datalist(datalist)
        self.data_name = data_name
        self.set_attrs(total_len=len(self.entries), batch_size=batch_size,
                       shuffle=False, num_workers=num_workers)


    def __getitem__(self, idx):
        rel = self.entries[idx]
        npy_path = os.path.join(self.root, rel, self.data_name)
        pc_noisy = np.load(npy_path).astype(np.float32)
        return {
            'pc_noisy': pc_noisy,   # (N, 3)
            'rel': rel,             # 'shapenet/<synset>/<model_id>'
        }

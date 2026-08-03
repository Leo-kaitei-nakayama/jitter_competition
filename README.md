# ASDN — Jittor port

Full Jittor conversion of the original PyTorch ASDN project
(*"You Should Learn to Stop Denoising on Point Clouds in Advance"*).
Same file layout, same math, no PyTorch / PyTorch-Lightning /
pytorch3d / torch_geometric / CUDA-extension dependencies at runtime.

## What replaced what

| Original dependency | Replacement here |
|---|---|
| `pytorch_lightning` (Trainer, LightningModule) | Plain Jittor training loops in `train_ASDN.py` / `train_classifier.py`; models are plain `jt.nn.Module`s. Lightning-only features (DDP multi-GPU strategy, 16-bit mixed precision, TensorBoard logger, LR monitor callback) are **not** replicated — training is single-process with file/console logging. |
| `pytorch3d.ops.knn_points`, `torch_cluster.fps` | Exact brute-force KNN and exact iterative FPS in `models/pointops_jt.py` |
| `pointops` CUDA extension (`furthestsampling`, `queryandgroup`, `interpolation`) | Pure-Jittor equivalents in `models/pointops_jt.py` (same offset-format API) |
| `Chamfer3D` CUDA extension (`chamfer_3DDist`) | KNN-based `chamfer_dist` in `models/InfoCD.py`, same `(dist1, dist2, idx1, idx2)` return |
| `pytorch3d.loss.chamfer_distance` | `models/utils.py: chamfer_distance` (same `(loss, None)` convention) |
| `pytorch3d` point-mesh distance (Evaluate) | Exact numpy point-triangle distance in `models/utils.py`; reverse (face→point) term approximated by face centroids |
| `torch_geometric` `MessagePassing` (EdgeConv) | Dense `(B,N,k,C)` neighbor gather + `jt.max` in `models/dynamic_edge_conv.py` (mathematically identical for the regular KNN graphs this code builds) |
| `torch.utils.data.DataLoader` + `Dataset` | `jittor.dataset.Dataset` (the dataset IS the loader; batching via `set_attrs`) |
| `torchvision.transforms.Compose` | Tiny local `Compose` in `utils/transforms.py`; transforms now operate on numpy arrays |
| `ReduceLROnPlateau` | Minimal reimplementation inside the train scripts |
| `multiprocessing.Pool` in `test_ASDN.py` | Sequential loop (forked CUDA/Jittor contexts are not safe) |

## Install

```bash
pip install jittor numpy scipy pandas tqdm
# optional, only for Evaluate.py mesh loading (a plain OFF parser is the fallback):
pip install point_cloud_utils
# optional, ONLY needed to load the original torch .ckpt files:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

No CUDA extension build steps (`Chamfer3D/setup.py`, `pointops/setup.py`) are needed anymore.

## Run

```bash
# training the denoiser
python train_ASDN.py --dataset_root ./data --dataset PUNet

# training the classifier
python train_classifier.py --dataset_root ./data --dataset PUNet

# testing / evaluation (accepts a Jittor .pkl or the original torch .ckpt)
python test_ASDN.py --ckpt pretrained/ASDN.ckpt
```

## Loading the original pretrained checkpoints

`DenoiseNetCD.load_from_checkpoint(...)` and `Classify.load_from_checkpoint(...)`
read the original PyTorch-Lightning `.ckpt` files directly (torch is imported
only to deserialize; tensors are copied into Jittor parameters **by name**).
Any unmatched keys are printed at load time — check that output the first time
you load: if many keys are skipped, the module naming needs adjusting.
`models/feature.py` still auto-loads `pretrained/classify.ckpt` at
construction, as the original did.

## Known caveats — read before relying on results

1. **Tested on CPU only, with random weights.** The following were executed
   and verified on CPU (Jittor 1.3.11): all `pointops_jt` ops, ScaleNet,
   every encoder/decoder block, the full `FeatureExtraction` forward pass,
   `DenoiseNetCD.patch_based_denoise` (5k-point cloud), the InfoCD training
   loss with a backward/optimizer step, `Classify`'s loss +
   `patch_based_shang` + optimizer step, transforms/patch-making, and the
   CD/HD/entropy/p2f metrics. **Not** yet verified: GPU execution
   (`jt.flags.use_cuda=1`), loading the real torch `.ckpt` files (needs
   torch installed; watch the unmatched-key warnings on first load), the
   `jittor.dataset` batch collation in a real epoch, and numerical parity
   against the PyTorch implementation's outputs.
2. **Speed.** The CUDA kernels were replaced with exact brute-force Jittor
   tensor math. Results should match, but KNN/FPS on large clouds will be
   significantly slower than the original extensions.
3. **Self-loop removal** in `classifyNet.get_edge_index` assumes each point's
   nearest neighbor is itself (true unless two points coincide exactly).
4. **Evaluate.py p2f**: the face→point half of the bidirectional point-mesh
   distance uses face centroids rather than pytorch3d's dense sampling, so
   absolute p2f numbers can differ slightly from the paper's pipeline (rank
   ordering between methods is preserved in practice).
5. **BatchNorm semantics** (`BatchNorm1d` over flattened (N_total, C)
   activations) follow the original exactly, but Jittor and PyTorch differ in
   minor default settings (momentum definition) — expect small numeric drift
   when fine-tuning from converted checkpoints.

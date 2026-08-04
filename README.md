# ASDN + score-based diffusion — Jittor point cloud denoising

Entry for the Jittor point cloud denoising competition (track 2). A Jittor port
of ASDN (*"You Should Learn to Stop Denoising on Point Clouds in Advance"*)
with the feature-fusion / gradient-fusion head and adaptive iterative sampling
from *"Adaptive and Iterative Point Cloud Denoising with Score-Based Diffusion
Model"* (Wang et al., arXiv 2509.14560) layered on top.

No PyTorch, PyTorch-Lightning, pytorch3d, torch_geometric or CUDA extensions at
runtime. Trained only on the competition's own ShapeNet data.

## Results

Local eval set of 100 shapes built by `make_eval_set.py` (σ ∈ 0.005–0.020,
matching the competition spec), split 60 for tuning and 40 held out. Scoring is
the organizers' `evaluate.py`: `0.5 × CD_score + 0.5 × P2S_score`.

| configuration | CD | P2S | **final** |
|---|---|---|---|
| `asdn-epoch039`, plain ASDN, single pass | 60.19 | 85.33 | **72.76** |
| single model + diffusion + repulsion | 64.36 | 90.15 | 77.26 |
| **ensemble + repulsion — holdout, 40 unseen** | **64.58** | **92.06** | **78.32** |

The holdout scored above the tune set (78.32 vs 77.96), so the tuning
generalizes. Local scores are only comparable against each other; the eval set
is not the competition's test set.

### Best known configuration

```bash
# two models, each at its own calibrated sigma
predict_on_starter.py --ckpt experiments/asdn_diffusion/asdn-epoch049.pkl \
    --use_fusion --static_depth --use_diffusion --diffusion_L 3 --sigma_scale 1.5
predict_on_starter.py --ckpt experiments/asdn_posfeat/asdn-epoch044.pkl \
    --use_fusion --static_depth --fusion_gate posfeat --fusion_k 32 \
    --use_diffusion --diffusion_L 3 --sigma_scale 1.2

ensemble.py --weights 0.6 0.4        # blend, order matches --pred_roots
postprocess.py --strength 0.3 --iters 2 --k 16    # repulsion; projection off
```

Note the posfeat model scores ~1 point *worse* than the baseline on its own, yet
adds ~0.7 to the ensemble — its errors point in different directions.

### What was tested and rejected

Kept in the repo behind default-off flags so they are not retried blind:

| idea | result |
|---|---|
| `--exact_score` (exact displacement target) | −9.6 at matched epoch |
| jet/MLS projection (`--project_strength`) | −0.9 to −3.3, monotonic in strength |
| `--fusion_gate posfeat --fusion_k 32` alone | −1.0 even at its own best sigma |
| per-cloud adaptive `sigma_scale` | +0.16, not worth the mechanism |
| more diffusion steps (L > 3) | monotonically worse to L=12 |

Jet projection failing is informative: the model's output is already smoother
than a local polynomial fit of itself, so any filter that only re-smooths the
denoised points — bilateral, MLS, Laplacian, WLOP's projection half — should be
expected to fail the same way. Repulsion worked because spacing is a property
of the point distribution, not of the surface estimate.

## Pipeline

```
train_classifier_on_starter.py   (optional) ScaleNet for ASDN's depth selection
        │
train_on_starter.py              denoiser: two-stage diffusion loss (Alg. 1)
        │  experiments/<run>/asdn-epochNNN.pkl
        ▼
predict_on_starter.py            adaptive iterative sampling (Alg. 2)
        │  results/<run>/shapenet/<synset>/<model_id>/denoised.npy
        ▼
check_submission.py              point-count / NaN / magnitude checks
```

Local evaluation loop:

```
make_eval_set.py   →  eval_gt/ eval_noisy/ eval_mesh_normalized/ eval_meta.csv
evaluate.py        →  per-sample CD / P2S scores  (organizers' script)
analysis.py        →  breakdown by category and noise level
visualize_errors.py→  interactive per-point error HTML
measure_cd_floor.py→  how much CD headroom is reachable
bench_fps.py       →  correctness + speed check for the vectorized FPS
```

`bridge/data_bridge.py` adapts the competition's data layout to the ASDN loss:
mesh → surface sample → unit sphere → augment → noise → one KNN patch.

## Install

```bash
conda create -n jittor python=3.9 -y && conda activate jittor
conda install -c conda-forge gcc=10 gxx=10 libgomp -y
pip install jittor numpy scipy pandas tqdm trimesh point-cloud-utils plotly matplotlib
```

## Train

```bash
mpirun -np 6 python -u train_on_starter.py \
    --data_root ./dataset_train --datalist ./datalist/train.txt \
    --use_fusion --static_depth \
    --epochs 100 --batch_size 72 --num_workers 2 --lr 1e-3 \
    --save_dir experiments/asdn_diffusion
```

`--batch_size` is **global** in Jittor and is divided across ranks; keep it
divisible by the rank count. `--static_depth` is **required** for multi-GPU —
see the notes below. Scale `--lr` roughly with √(ranks).

## Predict

```bash
python -u predict_on_starter.py \
    --ckpt experiments/asdn_diffusion/asdn-epoch049.pkl \
    --use_fusion --static_depth --use_diffusion \
    --data_root ./dataset_test_noisy --datalist ./datalist/test.txt \
    --out_root ./results/submission

python check_submission.py --datalist ./datalist/test.txt \
    --pred_root ./results/submission --noisy_root ./dataset_test_noisy

cd results/submission && zip -r ../../result.zip shapenet/
```

Inference flags must match training: `--use_fusion`, `--static_depth`,
`--t_norm`, `--fusion_k`, `--fusion_gate`.

## Key parameters

| flag | default | notes |
|---|---|---|
| `--sigma_scale` | 1.5 | Corrects a bias in the Eq. 15 noise estimator, which reads low and causes under-denoising. Measured optimum; **re-sweep after any retrain.** |
| `--diffusion_L` | 5 | reverse sampling steps |
| `--static_depth` | off | pin all samples to 4 layers; required under `mpirun` |
| `--t_norm` | `T` | `t/T`. The paper's `tau` makes the relative timestep identical for every cloud, which cancels the adaptive schedule — keep `T`. |
| `--mask_size` | 256 | K_p in Algorithm 1: only the innermost patch points carry the loss |
| `--fusion_k` | 16 | paper uses 32 |
| `--fusion_gate` | `pos` | `posfeat` matches the paper; changes parameter shapes |

## Notes and known issues

**Multi-GPU requires `--static_depth`.** Jittor's BatchNorm becomes SyncBN under
MPI, issuing an all-reduce inside every forward pass. ASDN picks its
encoder/decoder depth from the data, so ranks execute different numbers of
blocks and therefore different numbers of collectives, and NCCL deadlocks —
visible as 100% GPU utilization at idle power draw. Pinning the depth makes the
graph rank-invariant. A narrower fix would be disabling BN synchronization
instead, which would preserve adaptive depth.

**ASDN's depth selection is currently inert.** `Classify`/`ScaleNet` is never
trained (`--classify_ckpt` defaults to `None`) and receives no gradient, because
ρ is converted to a Python float in `assign_n_layer_based_on_rho`. Making the
mechanism real means training the classifier first via
`train_classifier_on_starter.py`.

**Noise is specified by standard deviation.** `numpy.random.laplace(0, b)` has
std `b·√2`; passing the target std as the scale produces noise 1.41× too strong.
`bridge/data_bridge.py:sample_noise` and `make_eval_set.py` derive the scale from
the target std. Eval sets built before this fix have σ ∈ 0.0074–0.0283 and are
not comparable.

**Where the remaining score is.** CD (62.56) lags P2S (91.32) at every noise
level. Low-noise clouds cap around 67.6 final score regardless of denoising
strength — that quarter of the samples is what holds the overall score down. A
per-cloud `sigma_scale` was measured and is worth only ~0.2 points, so a global
value is fine. An untested idea: `compute_gt_score` approximates the training
target with a nearest-neighbour lookup, but `bridge/data_bridge.py` produces
point-corresponded `(noisy, clean)` pairs, so the exact displacement is already
known.

**The legacy ASDN scripts do not run.** `train_ASDN.py`, `train_classifier.py`
and `test_ASDN.py` import `datasets.pcl`, `datasets.patch` and `Evaluate`, none
of which are in this repository. Use the `*_on_starter.py` scripts.

## Replaced dependencies

| original | replacement |
|---|---|
| `pytorch_lightning` | plain Jittor loops in `train_*_on_starter.py` |
| `pytorch3d.ops.knn_points`, `torch_cluster.fps` | `models/pointops_jt.py` (batched, sync-free FPS) |
| `pointops` CUDA extension | pure-Jittor equivalents, same offset-format API |
| `Chamfer3D` CUDA extension | KNN-based `chamfer_dist` in `models/InfoCD.py` |
| `pytorch3d.loss.chamfer_distance` | `models/utils.py:chamfer_distance` |
| `torch_geometric` EdgeConv | dense neighbour gather in `models/dynamic_edge_conv.py` |
| `torch.utils.data.DataLoader` | `jittor.dataset.Dataset` |
| `torchvision.transforms.Compose` | `utils/transforms.py:Compose` |

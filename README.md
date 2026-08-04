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

All figures below are the 40-shape holdout unless noted.

| configuration | CD | P2S | **final** |
|---|---|---|---|
| `asdn-epoch039`, plain ASDN, single pass | 60.19 | 85.33 | **72.76** |
| two-model ensemble + repulsion | 64.58 | 92.06 | 78.32 |
| two REFINED models + repulsion | 65.09 | 92.69 | 78.89 |
| **one refined model + repulsion** | **64.95** | **93.03** | **78.99** |

A single refined model beats the two-model ensemble, so ensembling is dropped:
same score, half the inference, one checkpoint. That follows from what the
refinement head does -- it removes each model's systematic error, after which
both sit near the same irreducible floor and their residuals look alike.
Averaging only pays when errors differ.

Local scores are only comparable against each other; the eval set is not the
competition's test set.

### Best known configuration

```bash
predict_on_starter.py --ckpt experiments/asdn_diffusion/asdn-epoch049.pkl \
    --use_fusion --static_depth \
    --use_diffusion --diffusion_L 3 --sigma_scale 1.5 \
    --refine_ckpt experiments/refine/refine-epoch009.pkl

postprocess.py --strength 0.3 --iters 2 --k 16    # repulsion; projection off
```

### The refinement stage

The largest single gain after the noise calibration, and the one that made
ensembling redundant. `models/refine.py` is a small EdgeConv head reading the
frozen model's output position and its encoder feature at that position, and
predicting a corrective displacement; the backbone stays frozen under
`jt.no_grad()` with every parameter `stop_grad`'d. Worth +1.33 on the tune set
and +0.67 on the holdout over the same model unrefined.

It was built only after `check_learnable.py` measured the residual error's
spatial autocorrelation at +0.35, implying roughly 12% of the error variance is
predictable from local context. The head converged to a 13% reduction in squared
error — and a head with twice the width and an extra layer reached the same 13%,
so that is the signal available rather than a capacity limit.

Train it with `train_refine.py`; its `--sigma_scale`, `--diffusion_L` and fusion
flags must match how the frozen model will be run at inference, since the head
learns to correct whatever that configuration actually produces.

### What was tested and rejected

Kept in the repo behind default-off flags so they are not retried blind:

| idea | result |
|---|---|
| `--exact_score` (exact displacement target) | −9.6 at matched epoch |
| jet/MLS projection (`--project_strength`) | −0.9 to −3.3, monotonic in strength |
| `--fusion_gate posfeat --fusion_k 32` alone | −1.0 even at its own best sigma |
| per-cloud adaptive `sigma_scale` | +0.16, not worth the mechanism |
| more diffusion steps (L > 3) | monotonically worse to L=12 |
| four-member ensemble (earlier epochs) | −0.86 |
| blending the output back toward the noisy input | monotonically worse |
| a bigger refinement head | same 13%, no gain |
| ensembling two REFINED models | −0.10 vs one refined model |

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
check_spacing.py   →  is the output clumped? (justified the repulsion filter)
check_learnable.py →  is the residual predictable? (justified the refine head)
make_stress_set.py →  synthetic probes: sharp edges, thin plates, noise beyond
                      the training range
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
| `--diffusion_L` | 3 | reverse sampling steps; swept 1-12, monotonic above 3 |
| `--refine_ckpt` | none | trained RefineHead from `train_refine.py`, applied per patch |
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

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
| entropy-ratio adaptive depth (wire the trained classifier) | measured before building (`measure_classifier.py`): on the round-A noise range the supervision target itself barely tracks noise — corr(target, σ) = +0.13 with per-bin means flat at 0.98–1.01 — so even a perfect classifier routes depth ~randomly. Backbone retrain skipped. |
| per-point early stop (`--stop_frac`) | measurement said the overshoot is real (36% of points worsen at step 3, oracle bound 9.2%, corr(score norm, harm) up to −0.98) yet every threshold lost on real tune data: −0.09 / −0.87 / −2.58 at 0.2 / 0.35 / 0.5. The rule freezes slowly-improving points along with the overshooters (64% still improve at step 3), and the refine head + repulsion downstream were fit to unfrozen outputs. Oracle ≠ reachable policy. |
| straight-path steps (`--straight`) | best CD of the sweep (+0.21) but double the P2S cost (−0.36): clamping later steps to the step-1 direction also clamps the normal-direction corrections. Net −0.06, no reason to keep. |
| weighted-mean stitching (`--stitch mean`) | −3.60 at α=1 (75.10 vs 78.70), CD *and* P2S both down. The ~6x patch overlap is not an independent ensemble: averaging positions across patches is a smoothing operation, and it fails for the same reason jet projection did — the output is already smoother than a local average of itself. CD dropping is the tell: averaging pulls points toward each other, worsening exactly the clumping it was meant to fix. Larger α cannot rescue it either — the weights are exp(−α·d), so α→∞ *is* the winner-take-all default, meaning mean stitching can only approach 78.70 from below. |
| tangential-displacement penalty (`--tangent_weight`, E_disp of Xu et al. 2024) | CD −1.35 / P2S +1.11 at its own best σ, at both λ=1.0 and λ=0.3 — the trade is the *reverse* of the intent, and λ-insensitive (0.12 CD across a 3.3× weight change). Under isotropic noise ~2/3 of the error is tangential, so the "slide" this term forbids is mostly legitimate correction. Full autopsy in the distribution-terms section. |

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
measure_classifier.py → is the depth classifier worth wiring in?
                      (measured no on round-A noise; re-run on round B's range)
bench_stitch.py    →  linear-memory stitching == old dense stitching, plus a
                      memory table for large-N clouds
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

## Distribution terms in the loss: three attempts, measured

All three add a distribution signal the score loss cannot see. Compared bare
(no refine head, fixed 0.3/2 filter) on 20 tune clouds, against the original
λ=0 backbone measured identically — **CD 66.39 / P2S 89.94 / final 78.18**.

| term | training behaviour | bare tune result |
|---|---|---|
| `--uniformity_weight 1e-3` | spacing variance 0.45 → 0.11 | CD 66.96 (**+0.57**) / P2S 88.17 (**−1.77**) / 77.58 |
| `--uniformity_weight 1e-4` | variance *rose* to 0.55 (5.8% of loss, too weak) | not evaluated |
| `--coverage_weight 10` | coverage 1.7e-4 → 1.0e-4, score loss below the λ=0 run | **CD 49.84 / P2S 76.62 / 63.23** at matched epoch 049 |
| `--tangent_weight 1.0` | ‖n×s‖² 1.53e-4 → 8.88e-6 over 60 epochs | CD 65.04 (**−1.35**) / P2S 91.05 (**+1.11**) / 78.04 at σ=1.1 |
| `--tangent_weight 0.3` | same term, 3.3× weaker | CD 64.90 / P2S 90.94 / 77.92 at σ=1.1 |

**Uniformity is rejected.** It does raise CD — the hypothesis that a
distribution term in the loss improves CD is confirmed — but it pays 3.1 P2S
points per CD point, a worse exchange rate than the hand-written repulsion
filter's 1:1. The term is satisfiable by moving points off the surface, so
that is what the optimizer does.

**Coverage is rejected too, and it is the most expensive lesson here.** The
argument for it was that it has no escape hatch — the only way to put a
predicted point near a clean point is to put it *at* the clean point, which is
surface accuracy — and its training curves agreed: coverage 1.69e-4 → 1.04e-4
with the score loss falling *below* the λ=0 run at the same epoch. At matched
epoch 049 it scores **CD 49.84 / P2S 76.62 / final 63.23** against the λ=0
control's 66.39 / 89.94 / 78.18. Sixteen CD points worse, having optimised a
term that is literally half of CD.

Why: the term is measured over the central `mask_size` rows only, but
inference runs every point of every patch. Chasing coverage inside the mask
inflates the displacements — mean τ̂ rose 538 → 945 on the same clouds — and
that inflated score field is what the whole cloud then gets. A σ sweep does
not rescue it: 1.5 / 1.0 / 0.7 gave final 64.6 / 66.8 / 67.1, with P2S pinned
near 78 throughout, so this is not a calibration error.

**The tangential penalty is rejected, and it failed in the most informative
way: backwards.** `--tangent_weight` adds ‖n×s‖² = |s|² − (n·s)², the part of
the displacement that slides along the surface instead of onto it, using the
mesh's *exact* face normal at the clean point each displacement targets
(`sample_surface` already returns the face index, so this costs one gather and
is reliable even at sharp edges, where an estimated normal is not). It was
chosen precisely because it cannot lose the way uniformity lost: pure
normal-direction motion reads 8.4e-21 against it, so it can never charge for
moving a point onto the surface. Trained from scratch, 60 epochs, cosine
1e-3 → 1e-5, at two weights.

It worked as specified and lost anyway. Every setting shows the same trade:
**CD −1.35, P2S +1.11.** Suppressing tangential motion improved projection and
degraded distribution — the exact opposite of the intent.

The premise was wrong. The noise is isotropic Laplace in 3D, so **roughly two
thirds of its energy is tangential**: a noisy point is not only lifted off the
surface, it is also displaced *along* it. Correcting that lateral error is
what the score's tangential component is for. P2S cannot see it and CD can, so
forbidding it buys P2S and sells CD. The paper's E_disp is designed to
*preserve* an already-good sampling during mesh fitting; under isotropic noise
the tangential component is not a distribution to protect, it is error to
remove.

The dose-response settles the attribution without another run: **λ = 1.0 and
λ = 0.3 differ by 0.12 CD** (65.04 vs 64.90) despite the trained tangential
residual differing 2.2×. A term whose strength can move 3.3× while the score
does not move is not the thing steering the result — so a λ=0 from-scratch
control, ~10 GPU-hours, would not change the decision.

`--sigma_scale` was re-swept, as it must be: the optimum moved 1.5 → **1.1**,
and the peak is unambiguous (0.8 / 0.9 / 1.0 / 1.1 / 1.2 → 75.42 / 76.90 /
77.76 / **78.04** / 77.87). The numbers above are best-vs-best.

**The transferable lesson: a training metric improving is not evidence that
the deployed pipeline improves.** All three rejected terms had textbook
training curves. None survived contact with the eval set. Prefer measurements
taken through the actual inference path, on held-out clouds, over anything
read off a loss.

**And the second lesson, from three losses in a row: the ceiling is not in the
regulariser.** Uniformity, coverage and the tangential penalty all add a term
to the same loss around the same non-injective target, and all three lose. The
next attempt should change the target or the architecture, not add a fourth
term.

**Changing the loss changes the score scale, so `--sigma_scale` must be
re-swept.** The coverage backbone emits displacements ~1.8× larger, so the
Eq. 15 estimator reads the same cloud as far noisier: mean τ̂ went 538 → 945
with clouds pinned at the schedule top. Scored at the old σ=1.5 it looks
catastrophic (final 64.6) purely from over-denoising. This is the same trap as
the epoch-059 comparison: **a new backbone is not comparable until its σ is
recalibrated.**

## Why CD trails P2S, and where the fix belongs

The training target is Eq. 14's `S(x) = NN(x, x_clean) - x`, and **it is not
injective**: several noisy points can share one nearest clean point, and the
per-point squared loss is *fully satisfied* when all of them land on it. The
loss cannot see collapse. CD's second term — `mean_{b∈clean} min_{a∈pred}` —
punishes exactly that, while P2S only measures point-to-surface and so agrees
with the loss. Hence 93 vs 65.

Everything downstream (repulsion filter, refine head) tidies up after the
scramble. `--uniformity_weight` puts the missing signal **in the backbone's own
loss**, on `x + ŝ` — where the network is sending each point — so the gradient
reaches the stage that creates the problem:

```bash
train_on_starter.py ... --uniformity_weight 1e-4    # sweep 1e-5 .. 1e-3
```

`train_refine.py` has the same flag for the head-only variant. The metric is
the relative variance of nearest-neighbour spacing (scale-free, zero for an
even distribution); a check on synthetic clouds gives 0.00 for a regular grid,
0.36 for random scatter, and 1.00 when half the points are collapsed onto three
spots — the ordering the term needs.

## Noise-adaptive repulsion (sim-calibrated, verify on tune before trusting)

A simulation against the organizers' exact CD metric (ideal surface projection
plus calibrated tangential scramble, unit sphere, 50k points) showed the
optimal repulsion strength flips sign with noise level: at σ≈0.008 the model
already sits at the projection bound and the production 0.3/2 setting *costs*
~1 CD point versus a gentle 0.1/1, while at σ≥0.014 it leaves ~3 CD points on
the table versus 0.5–0.6 strength at 4–5 iters. The filter can now read the
per-cloud σ̂ the model already estimates:

```bash
predict_on_starter.py ... --save_tau results/<run>/taus.csv
postprocess.py --pred_root ... --out_root ... --tau_csv results/<run>/taus.csv
```

`--adaptive_sched` holds the σ→(strength, iters) bands; the default is the
simulation's optimum. Without `--tau_csv` nothing changes. The calibration is
synthetic — sweep the bands on the tune split before submitting with them.

## Running on a different dataset

Nothing about the pipeline is tied to this ShapeNet layout, but four things are
calibrated to *this* data and must be revisited.

**1. Point the scripts at the new data.** `--data_root`, `--datalist`, and
`--mesh_name` if the meshes sit somewhere other than
`models/model_normalized.obj` inside each entry. A datalist line is just a
relative path joined to the root, so any directory shape works.

**2. Set the noise range, and check the schedule can hold it.** `--noise_min`,
`--noise_max`, `--noise_dist`. Both bounds are STANDARD DEVIATIONS.

The diffusion schedule can only represent sigma up to **0.0316** by default.
Above that, `find_t_for_sigma` returns the last timestep for everything, so
every loud cloud is told the same thing and the timestep stops carrying
information. For noisier data raise it:

```bash
--max_sigma 0.06        # on train_on_starter.py, train_refine.py AND predict_on_starter.py
```

It must match across all three -- the timestep-to-sigma mapping is what the
relative timestep means to the network. `predict_on_starter.py` warns if the
estimated tau hits the top of the schedule.

**3. Re-sweep `--sigma_scale`.** It is a calibration constant, not a universal
one: 1.5 corrects a bias measured on this data, and it shifted to 1.2 for a
model that differed only in its fusion gate. Sweep 0.8-2.0 against a local eval
set before trusting any score.

**4. Retrain the refinement head.** It learns one specific frozen model's
systematic error, so a new backbone needs a new head. Its `--sigma_scale`,
`--diffusion_L` and fusion flags must match how that backbone will be run at
inference, since it corrects whatever that configuration actually produces.

Then rebuild the local eval set from the new training data and re-measure --
`make_eval_set.py`, split into tune and holdout, and treat the previous scores
as belonging to the old dataset only.

`make_stress_set.py` is worth running early on new data: it generates its own
geometry, so it works without any dataset at all, and it answers whether the
model survives noise beyond its training range before you find out from a
leaderboard.

## Notes and known issues

**Multi-GPU requires `--static_depth`.** Jittor's BatchNorm becomes SyncBN under
MPI, issuing an all-reduce inside every forward pass. ASDN picks its
encoder/decoder depth from the data, so ranks execute different numbers of
blocks and therefore different numbers of collectives, and NCCL deadlocks —
visible as 100% GPU utilization at idle power draw. Pinning the depth makes the
graph rank-invariant. The narrower fix — `--no_bn_sync` — disables BN
synchronization instead, which preserves adaptive depth under `mpirun`.

**ASDN's depth selection is inert in the current best model.** The classifier
was trained (`train_classifier_on_starter.py` → `experiments/classify/`,
backed up in `checkpoints/`), but the 78.99 backbone was trained and is run
with `--static_depth`, which bypasses it entirely — and a backbone trained at
a fixed depth has never exercised its shallow exit paths, so flipping adaptive
depth on at inference without retraining produces garbage for any sample
routed shallow. `measure_classifier.py` measured whether wiring it in is worth
that retrain, and the answer was no — see the rejected-ideas table. Measured
at σ ∈ 0.005–0.05 (round-B-like) the entropy ratio does wake up —
corr(target, σ) = +0.386, per-bin means monotonic 0.99 → 1.15 — but 60% of
targets then sit above the tanh output cap, so the classifier saturates at
1.0 and would need an activation fix plus retraining. Even then the
score-norm σ̂ estimate that drives the adaptive schedule is the stronger,
already-working noise signal; if depth adaptivity is ever revisited, drive it
from σ̂, not from the entropy classifier. Re-run `measure_classifier.py` with
round B's actual noise range before reopening any of this. The mechanics for
the retrain exist if it is ever justified: train without `--static_depth`,
passing `--classify_ckpt`, `--no_bn_sync`, and `--init_ckpt` to warm-start.
The classifier receives no gradient during denoiser training regardless (ρ is
converted to a Python float in `assign_n_layer_based_on_rho`), so it keeps
exactly the weights it was given.

**Large clouds: stitching and seed-KNN are linear-memory.** The patch
stitching used to build three dense `(num_patches, N)` arrays — O(0.006·N²)
bytes, 240 MB at N=50k but 24 GB at N=500k — and `knn_points` materialized the
full seeds×cloud distance matrix. Both are now bounded:
`DenoiseNetCD.select_stitch_source` walks only the actual patch-point covering
pairs (fuzz-tested identical to the dense argmax, ties included), and
`knn_points` processes query rows in ~800 MB chunks with per-row results
unchanged. `bench_stitch.py` re-verifies both claims end-to-end on synthetic
clouds of any size — run it after touching either code path, and once on any
machine that will serve round B.

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

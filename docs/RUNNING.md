# Running experiments

## Configurations

| Directory / prefix | Model | Optimizer |
|---|---|---|
| `configs/main/llama1b_adamw_` | Llama-3.2-1B-Instruct | AdamW |
| `configs/main/llama8b_adamw_` | Llama-3.1-8B-Instruct | AdamW |
| `configs/main/vitb32_adam_` | CLIP ViT-B/32 | Adam |
| `configs/main/vitl14_adam_` | CLIP ViT-L/14 | Adam |
| `configs/muon/llama1b_muon_` | Llama-3.2-1B-Instruct | Muon + auxiliary AdamW |

Each prefix has `ft.json` and `smat.json`. `configs/ablations/` contains examples
that omit one SMAT operator. Merge coefficients are set in each configuration.

## Parameters

- `seed`: training seed for this run. To run another seed, copy the configuration,
  change `seed` and use a separate `OUTPUT_ROOT`.
- `train`: optimizer, learning rate, batch size, precision-related options and
  SMAT settings.
- `evaluation.repeats`: number of evaluation repetitions per checkpoint.
- `merging`: coefficients and sparsity settings for each merger.
- `MODEL_ROOT`, `DATA_ROOT`, `OUTPUT_ROOT`: local model, prepared-data and output
  directories. Undefined environment variables are errors.

Language training uses BF16, batch 8 and maximum sequence length 512. Task epoch
counts are defined in `smat/experiments/trace_llm.py`. Llama-8B splits one expert
across two GPUs. Vision trains the FP32 image encoder with batch 128 for 4,000
steps per task; the text encoder, projection and classification heads are frozen.
The prepared training split already excludes dev examples.

The fast backend requires Triton. Pinned-base offload additionally needs host
memory for initialization weights (about 2.3 GiB for Llama-1B and 15 GiB for
Llama-8B), plus data, model-loading and process memory.

## Commands and outputs

The README shows the full pipeline. Use `python -m smat --help` or a command's
`--help` for arguments.

- `train CONFIG TASK`: one expert.
- `train-suite CONFIG --tasks TASK ...`: selected experts, or all configured tasks.
- `merge-suite CONFIG OUTPUT --methods wa ta ties dare della`: selected mergers.
- `eval-suite CONFIG OUTPUT --methods expert wa ta ties dare della`: evaluation.
- `scripts/summarize.py OUTPUT/summary.json`: percentage scores and training costs.

TRACE evaluates each task's `eval` split; NumGLUE-cm contains 41 examples and
never appends `test`. Vision evaluates the configured split. Evaluation uses
SGLang for language and frozen text classifiers for vision. Available GPUs run
independent evaluation jobs. `summary.json` contains task scores and macro means;
`Avg` averages the five merger scores and excludes Expert.

Training records synchronized loop wall time and peak allocated GPU memory.
The reporting script sums expert loop times and takes the maximum per-device
memory peak; these are distinct from parallel elapsed time.

Training refuses to overwrite an expert. Failed runs may leave `.partial`
directories. Use a new output root when changing a configuration or checkpoint.
Merge/evaluation can resume matching completed work; optimizer-state training
resume is not implemented. Outputs can include resolved local paths and optional
predictions, so review them before sharing.

## Subspace-constrained training

`train.method = "subspace"` is a third expert-training method alongside `ft`
and `smat`. Where SMAT changes the update rule, this changes the hypothesis
space: every 2-D weight of the vision tower is reparametrised as
`W = W_0 + U C V^T`, with `U` and `V` frozen bases fixed before training and
`C` the only trainable tensor. For ViT-B/32 that is 73 matrices and about
203k trainable parameters, 0.24% of the tower.

The bases come from gradient statistics at the pre-trained weights. For each
matrix, the Grams `A = E[G^T G]` and `S = E[G G^T]` are accumulated on the
task's own data and again on a generic reference set `D_gen`, and the bases are
the top-k solutions of the generalised eigenproblem `A_task v = lambda C v`
with `C = A_gen + delta * (tr A_gen / n) * I`. The directions kept are the ones
the task wants relative to those the pre-trained model already uses
generically, which are the directions the other experts are least likely to
claim.

`D_gen` must not be one of the tasks being merged. Supply it as a directory of
images, searched recursively; no labels are needed because the generic
statistics use a label-free probe.

```bash
export CUDA_VISIBLE_DEVICES=0
python -m smat train-suite configs/main/vitb32_adam_subspace.json
python -m smat merge-suite configs/main/vitb32_adam_subspace.json "$OUTPUT_ROOT/vitb32_subspace"
python -m smat eval-suite  configs/main/vitb32_adam_subspace.json "$OUTPUT_ROOT/vitb32_subspace"
```

Set `train.subspace.generic_dir` before training, and note that the subspace
method uses a larger learning rate than `ft`: AdamW moves each coordinate by
about the learning rate regardless of gradient scale, so `||Delta||` grows like
`lr * sqrt(#parameters)` and matching the delta scale of a full fine-tune needs
roughly twenty times the rate. The supplied configuration uses `1e-4` against
`1e-5` for `ft`, which is still the conservative side of that ratio.

| Option | Default | Meaning |
| --- | --- | --- |
| `k` | 64 | Subspace dimension per side, clamped to each tensor's short side |
| `delta` | 0.01 | Ridge on the generic Gram, as a fraction of its mean eigenvalue |
| `batches` | 30 | Batches of gradients for the task statistics |
| `generic_batches` | `batches` | Batches for the generic statistics |
| `generic_dir` | — | Directory of images for `D_gen`; required unless `no_reference` |
| `generic_size` | 10000 | Cap on images read from `generic_dir` |
| `one_sided` | false | Restrict only the input side, so the delta is `B V^T` |
| `no_reference` | false | Ablation: replace the generic Grams with the identity |
| `train_rest` | false | Also train the tensors the subspace does not cover |
| `cache` | `<work_root>/bases/<name>` | Where computed bases are stored and reused |
| `rebuild` | false | Recompute the bases even when a cached file exists |

Bases are cached per task and reused by a resubmitted run, so the statistics
stage is paid once. The training log prints `||C||`, which equals `||Delta||_F`
because `U` and `V` are orthonormal, and the run refuses to start if the
reparametrisation does not reproduce `W_0` exactly. The delta is folded into
the weights before the checkpoint is written, so an expert is an ordinary
image-encoder state dict and every merger and evaluation path consumes it
unchanged.

`configs/ablations/vitb32_subspace_no_reference.json` removes `D_gen`: with the
identity as the denominator the generalised problem collapses to the ordinary
one and the bases become the top-k eigenvectors of the task Grams alone. That
isolates what the generic reference contributes, as opposed to the dominant
gradient directions of the task.

## open_clip backbone

`"experiment": "clip8_openclip"` runs the same eight tasks, splits, prompts
and protocol as `clip8_vit` on an open_clip backbone. It is a separate
experiment rather than a flag because the two implementations of CLIP do not
expose the same tensors, and for any method that acts per matrix the
difference is not cosmetic:

| | `clip8_vit` (HF) | `clip8_openclip` |
| --- | --- | --- |
| attention projections | `q_proj`, `k_proj`, `v_proj`, 768x768 each | one fused `in_proj_weight`, 2304x768 |
| output projection | `visual_projection`, outside the tower, frozen | `proj`, inside `visual`, trainable |
| 2-D tensors on ViT-B/32 | 73 | 50 |
| subspace cores at k=64 | 297,412 | 203,204 |

The gradient of the fused matrix is the vertical stack of the three, so
`A_fused = A_q + A_k + A_v`: the fused layout gives q, k and v one shared
input basis and lets the output basis mix their rows, while the split layout
gives each its own. Expert checkpoints and merged results from the two
backbones are therefore not interchangeable.

```bash
python -m pip install -r requirements/openclip.txt
export CUDA_VISIBLE_DEVICES=0
python -m smat train-suite configs/main/vitb32_openclip_subspace.json
python -m smat merge-suite configs/main/vitb32_openclip_subspace.json "$OUTPUT_ROOT/vitb32_oc_subspace"
python -m smat eval-suite  configs/main/vitb32_openclip_subspace.json "$OUTPUT_ROOT/vitb32_oc_subspace"
```

`configs/main/vitb32_openclip_ft.json` is the plain fine-tuning baseline on the
same backbone, so the two differ in one variable only.

Weights resolve in two unambiguous forms, because open_clip has two loading
paths and guessing between them is how an offline run becomes a download:

* `base_model` is a weight file (`open_clip_pytorch_model.bin`,
  `open_clip_model.safetensors`, `pytorch_model.bin`) or a directory holding
  one: it is passed to open_clip as the state dict;
* `base_model` is any other directory: it is the open_clip cache directory and
  `open_clip_pretrained` (default `openai`) is the tag. This is the route for
  the OpenAI weights, which ship as a TorchScript archive that the state-dict
  path cannot read.

`open_clip_arch` defaults to `ViT-B-32-quickgelu`. The backend refuses to load
the OpenAI weights under an architecture without `quickgelu` in its name:
those weights were trained with QuickGELU, open_clip decides the activation
from the architecture name, and the mismatch loads the same tensors and trains
without error while scoring several points lower.

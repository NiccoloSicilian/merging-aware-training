"""Merge-aware training by confining each expert update to a selected subspace.

Where SMAT changes the *update rule* -- simulating a merger's transform of the
parameter update -- this changes the *hypothesis space*: every eligible matrix
is reparametrised as

    W = W_0 + U C V^T,

with U and V frozen bases fixed before training and C the only trainable
tensor. The delta is therefore rank <= k with its row and column spaces pinned,
by construction rather than by penalty. Experts still train independently.

Choosing the bases
------------------
For each matrix we accumulate two gradient Grams on the task's own data,

    A_j = E[G^T G]   (n x n, input side)     S_j = E[G G^T]   (m x m, output side)

and the same two on a generic reference set D_gen that is not one of the tasks
being merged, giving A_g and S_g. The bases are the top-k solutions of the
generalised symmetric-definite eigenproblem

    A_j v = lambda C v,     C = A_g + delta * (tr A_g / n) * I,

i.e. the directions the task wants to move along *relative to* the directions
the pre-trained model already uses generically. Those are the ones the other
experts are least likely to claim, which is what makes the resulting task
vectors composable. The same is solved with (S_j, S_g) for the output side.

The ridge is scaled by the mean eigenvalue of the denominator, so it is
scale-free. It is not cosmetic: where the generic energy is ~0 the ratio
diverges and the top eigenvectors would be pure estimation noise.

Setting ``no_reference`` replaces A_g and S_g with the identity, which collapses
the generalised problem into the ordinary one: the bases become the top-k
eigenvectors of the task Grams alone, with no selectivity. That ablation
therefore runs through this same code rather than a parallel path.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils import parametrize


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

_ALLOWED = {
    "k",
    "delta",
    "batches",
    "generic_batches",
    "generic_dir",
    "generic_size",
    "one_sided",
    "no_reference",
    "train_rest",
    "cache",
    "rebuild",
}


def validate(settings: Mapping[str, Any]) -> None:
    """Reject misspelled subspace options before any model is loaded."""

    values = settings.get("subspace")
    if not isinstance(values, Mapping):
        raise ValueError("train.subspace must be an object")
    unknown = set(values) - _ALLOWED
    if unknown:
        raise ValueError(f"unsupported train.subspace options: {sorted(unknown)}")
    k = int(values.get("k", 64))
    if k < 1:
        raise ValueError("train.subspace.k must be positive")
    delta = float(values.get("delta", 1e-2))
    if not math.isfinite(delta) or delta < 0:
        raise ValueError("train.subspace.delta must be finite and nonnegative")
    for key in ("batches", "generic_batches", "generic_size"):
        if key in values and int(values[key]) < 0:
            raise ValueError(f"train.subspace.{key} must not be negative")
    if not values.get("no_reference", False) and not values.get("generic_dir"):
        raise ValueError(
            "train.subspace needs generic_dir (a directory of images for D_gen), "
            "or no_reference=true to run the ablation without a reference set"
        )


# ── which tensors ──────────────────────────────────────────────────────────

def eligible(
    root: nn.Module, named_parameters: Sequence[tuple[str, nn.Parameter]]
) -> list[dict[str, Any]]:
    """Every 2-D trainable parameter, resolved to the module that holds it.

    ``parametrize`` needs the owning module, while the runner hands out names.
    Those names are relative to whatever submodule the experiment adapter chose
    -- the CLIP adapter returns ``model.encoder.named_parameters()`` -- so
    splitting a name and calling ``get_submodule`` on the wrong root would
    fail. Resolving by tensor identity instead is independent of which root the
    names came from.

    For HF CLIP ViT-B/32 this selects 73 tensors: 12 layers x
    {q,k,v,out,fc1,fc2} plus the position embedding. The patch embedding is 4-D
    and the LayerNorms and biases are 1-D, so none of them are here -- the same
    set the SVD-based mergers act on.
    """

    owners: dict[int, tuple[nn.Module, str]] = {}
    for module in root.modules():
        for attribute, parameter in module.named_parameters(recurse=False):
            owners[id(parameter)] = (module, attribute)

    records = []
    for name, parameter in named_parameters:
        if parameter.dim() != 2 or not parameter.requires_grad:
            continue
        found = owners.get(id(parameter))
        if found is None:
            raise RuntimeError(
                f"{name} is not a parameter of the model handed to the subspace; "
                f"the adapter's named_parameters and its model disagree"
            )
        module, attribute = found
        records.append(
            {
                "name": name,
                "module": module,
                "attribute": attribute,
                "shape": tuple(parameter.shape),
                "parameter": parameter,
            }
        )
    if not records:
        raise ValueError("no 2-D parameter is eligible for the subspace")
    return records


def rank_for(shape: tuple[int, int], k: int, one_sided: bool) -> int:
    """k clamped to the tensor, so a short side cannot ask for more directions
    than it has. The CLIP position embedding is 50x768 and lands on 50."""

    rows, columns = shape
    return min(k, columns) if one_sided else min(k, rows, columns)


# ── the reparametrisation ──────────────────────────────────────────────────

class SubspaceDelta(nn.Module):
    """W = W_0 + U C V^T with U, V frozen buffers and C the only parameter.

    C starts at zero, so the model begins exactly at W_0: the subspace changes
    the hypothesis space, not the starting point. U and V have orthonormal
    columns, so ||Delta||_F = ||C||_F, which is a free self-check at any step.
    """

    def __init__(
        self, basis: torch.Tensor, output_basis: torch.Tensor | None, rows: int
    ):
        super().__init__()
        self.register_buffer("basis", basis)
        self.two_sided = output_basis is not None
        if self.two_sided:
            self.register_buffer("output_basis", output_basis)
            self.core = nn.Parameter(
                torch.zeros(output_basis.shape[1], basis.shape[1])
            )
        else:
            self.register_buffer("output_basis", None)
            self.core = nn.Parameter(torch.zeros(rows, basis.shape[1]))

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        # fp32 whatever the surrounding autocast does: the delta is a small
        # correction to a large weight and bf16 would quantise it away.
        with torch.autocast(device_type=original.device.type, enabled=False):
            core = self.core.float()
            basis = self.basis.float()
            delta = (
                self.output_basis.float() @ core @ basis.T
                if self.two_sided
                else core @ basis.T
            )
        return original + delta.to(original.dtype)

    def right_inverse(self, weight: torch.Tensor) -> torch.Tensor:
        # Only reached if something assigns to the parametrised attribute. The
        # map is the identity while C is zero, which is what registration needs.
        return weight


class SubspaceTraining:
    """Owns the parametrisations for one expert, from install to materialise."""

    def __init__(self, records, bases, device, train_rest: bool):
        self.records = records
        self.initial = {
            record["name"]: record["parameter"].detach().clone() for record in records
        }
        self.cores: list[tuple[str, nn.Parameter]] = []
        for record in records:
            entry = bases[record["name"]]
            basis = entry["basis"].to(device)
            output_basis = entry.get("output_basis")
            output_basis = None if output_basis is None else output_basis.to(device)
            delta = SubspaceDelta(basis, output_basis, record["shape"][0]).to(device)
            parametrize.register_parametrization(
                record["module"], record["attribute"], delta, unsafe=True
            )
            holder = record["module"].parametrizations[record["attribute"]]
            holder.original.requires_grad_(False)
            record["delta"] = delta
            self.cores.append((f"{record['name']}::core", delta.core))
        self.train_rest = train_rest

    def trainable_named_parameters(self, named_parameters):
        """What the optimizer should see: the cores, plus the uncovered tensors
        only when train_rest is set.

        Off by default. conv/patch embedding, the LayerNorms and the biases are
        shared by every expert, so leaving them at the pre-trained values is
        what lets the matrices alone carry the task -- and it is the arm of the
        experiment that tests whether confining the matrices is enough.
        """

        covered = {record["name"] for record in self.records}
        rest = [
            (name, parameter)
            for name, parameter in named_parameters
            if name not in covered
        ]
        if not self.train_rest:
            for _, parameter in rest:
                parameter.requires_grad_(False)
            return list(self.cores)
        return list(self.cores) + rest

    def drift(self) -> float:
        """max |W - W_0| across the covered matrices; must be 0 before step 1."""

        with torch.no_grad():
            return max(
                float(
                    (
                        getattr(record["module"], record["attribute"]).detach()
                        - self.initial[record["name"]].to(
                            getattr(record["module"], record["attribute"]).device
                        )
                    )
                    .abs()
                    .max()
                )
                for record in self.records
            )

    def core_norm(self) -> float:
        total = 0.0
        for record in self.records:
            delta = record.get("delta")
            if delta is not None:
                total += float(delta.core.detach().float().pow(2).sum())
        return math.sqrt(total)

    def materialise(self) -> None:
        """Fold U C V^T into plain weights and drop the parametrisations.

        Must run before the checkpoint is written. An expert here is a plain
        image-encoder state dict and the merge kernels diff those tensors by
        name, so a saved parametrisation would rename every covered matrix and
        the task vector would quietly lose all of them.
        """

        for record in self.records:
            if parametrize.is_parametrized(record["module"], record["attribute"]):
                parametrize.remove_parametrizations(
                    record["module"], record["attribute"], leave_parametrized=True
                )
            record.pop("delta", None)
        self.cores = []

    def metadata(self) -> dict[str, Any]:
        return {
            "matrices": len(self.records),
            "core_parameters": sum(
                record["delta"].core.numel()
                for record in self.records
                if "delta" in record
            ),
            "train_rest": self.train_rest,
        }


# ── gradient statistics ────────────────────────────────────────────────────

def _flat_image_dataset(directory: str, transform, limit: int):
    """Every image under a directory, recursively, with a dummy label.

    The generic statistics use a label-free probe, so D_gen needs no annotation
    and its layout does not matter. torchvision's ImageFolder would demand one
    subdirectory per class and reject a flat dump.
    """

    from PIL import Image

    paths: list[str] = []
    for current, _, names in os.walk(os.path.expanduser(directory)):
        for name in sorted(names):
            if name.lower().endswith(IMAGE_SUFFIXES):
                paths.append(os.path.join(current, name))
                if limit and len(paths) >= limit:
                    break
        if limit and len(paths) >= limit:
            break
    if not paths:
        raise ValueError(
            f"no images under {directory}; D_gen must be a directory of images "
            f"(any layout, searched recursively)"
        )
    paths.sort()

    class _Flat(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(paths)

        def __getitem__(self, index: int):
            image = Image.open(paths[index]).convert("RGB")
            return {"images": transform(image), "labels": 0, "task": "__generic__"}

    return _Flat(), len(paths)


def _generic_loader(values: Mapping[str, Any], model, batch_size: int, seed: int):
    from torch.utils.data import DataLoader, default_collate

    transform = getattr(model, "train_preprocess", None)
    if transform is None:
        raise RuntimeError(
            "the experiment model exposes no train_preprocess, so D_gen cannot "
            "be preprocessed the same way as the task data"
        )

    def collate(examples):
        images = default_collate([example["images"] for example in examples])
        return {"images": images}

    dataset, count = _flat_image_dataset(
        str(values["generic_dir"]), transform, int(values.get("generic_size", 10000))
    )
    print(f"[subspace] D_gen: {count} images from {values['generic_dir']}", flush=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        generator=torch.Generator().manual_seed(seed + 1),
        collate_fn=collate,
    )


def accumulate_grams(
    model,
    records,
    loader,
    batches: int,
    kind: str,
    device,
    loss_function=None,
    generator=None,
):
    """E[G^T G] and E[G G^T] over ``batches`` aggregated-batch gradients.

    Per-sample Grams would be sharper and cost one backward per sample; the
    batch gradient is dominated by the mean direction, which is the one the
    optimizer actually follows.

    kind='task'    : the task's own loss, through the frozen zero-shot head.
    kind='generic' : r^T f(x) with a FRESH random r per batch. Label-free, so
                     D_gen needs no annotation, and averaging over r probes the
                     whole embedding Jacobian instead of a single direction.
    """

    inputs = {
        record["name"]: torch.zeros(
            record["shape"][1], record["shape"][1], device=device, dtype=torch.float32
        )
        for record in records
    }
    outputs = {
        record["name"]: torch.zeros(
            record["shape"][0], record["shape"][0], device=device, dtype=torch.float32
        )
        for record in records
    }
    owner = getattr(model, "module", model)
    was_training = owner.training
    owner.eval()
    iterator = iter(loader)
    seen = 0
    for index in range(batches):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        model.zero_grad(set_to_none=True)
        if kind == "task":
            loss = loss_function(model, batch, device)
        else:
            images = batch["images"].to(device, non_blocking=True)
            features = owner(images)
            if isinstance(features, tuple):
                features = features[0]
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
            probe = torch.randn(
                features.shape[-1],
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            loss = (features.float() @ probe).sum()
        loss.backward()
        with torch.no_grad():
            for record in records:
                gradient = record["parameter"].grad
                if gradient is None:
                    continue
                gradient = gradient.float()
                inputs[record["name"]] += gradient.T @ gradient
                outputs[record["name"]] += gradient @ gradient.T
        seen += 1
        if (index + 1) % 10 == 0:
            print(f"[subspace]   {kind}: {index + 1}/{batches} batches", flush=True)
    model.zero_grad(set_to_none=True)
    owner.train(was_training)
    for name in inputs:
        inputs[name] /= max(seen, 1)
        outputs[name] /= max(seen, 1)
    return inputs, outputs, seen


# ── the eigenproblem ───────────────────────────────────────────────────────

def top_directions(
    numerator: torch.Tensor, denominator: torch.Tensor, k: int, delta: float
):
    """Top-k of the generalised eigenproblem P v = lambda C v, C = Q + ridge.

    Reduced to a symmetric problem by whitening: with C = R R^T and v = R^-T w,
        P R^-T w = lambda R w  =>  (R^-1 P R^-T) w = lambda w.
    The generalised eigenvectors are C-orthogonal rather than orthonormal, so a
    QR follows. It changes the basis but not the span, and the parametrisation
    depends only on the span because C is free.
    """

    size = numerator.shape[0]
    P = numerator.double().cpu()
    Q = denominator.double().cpu()
    C = Q + delta * (torch.diagonal(Q).sum() / size) * torch.eye(
        size, dtype=torch.float64
    )
    # Grams are symmetric in exact arithmetic, but accumulation leaves them
    # asymmetric at roundoff level and both factorisations assume otherwise.
    C = 0.5 * (C + C.T)
    P = 0.5 * (P + P.T)
    try:
        R = torch.linalg.cholesky(C)
    except Exception:
        # Not positive definite at this ridge: floor the eigenvalues instead,
        # which regularises where it is needed rather than uniformly.
        values, vectors = torch.linalg.eigh(C)
        floor = max(float(values.max()) * 1e-10, 1e-30)
        C = vectors @ torch.diag(values.clamp_min(floor)) @ vectors.T
        R = torch.linalg.cholesky(0.5 * (C + C.T))
    inverse = torch.linalg.solve_triangular(
        R, torch.eye(size, dtype=torch.float64), upper=False
    )
    reduced = inverse @ P @ inverse.T
    reduced = 0.5 * (reduced + reduced.T)
    values, vectors = torch.linalg.eigh(reduced)
    order = torch.argsort(values, descending=True)[:k]
    basis = torch.linalg.qr(inverse.T @ vectors[:, order])[0]
    return basis.float(), values[order].float()


def energy_ratio(basis: torch.Tensor, numerator: torch.Tensor, denominator: torch.Tensor):
    """(tr B^T P B / tr P) / (tr B^T Q B / tr Q): captured task energy over
    captured generic energy. About 1.0 for a basis that knows nothing."""

    B = basis.double().cpu()
    P = numerator.double().cpu()
    Q = denominator.double().cpu()
    own = float(torch.trace(B.T @ P @ B) / torch.trace(P))
    generic = float(torch.trace(B.T @ Q @ B) / torch.trace(Q))
    return own, generic, own / max(generic, 1e-30)


def solve_bases(
    records,
    task_inputs,
    task_outputs,
    generic_inputs,
    generic_outputs,
    k: int,
    delta: float,
    one_sided: bool,
    seed: int,
    no_reference: bool,
):
    """One eigenproblem per matrix per side, with the diagnostics that say
    whether the result is selective at all."""

    bases: dict[str, dict[str, Any]] = {}
    report: dict[str, dict[str, float]] = {}
    random = torch.Generator().manual_seed(seed + 2)
    for index, record in enumerate(records, 1):
        name = record["name"]
        rows, columns = record["shape"]
        rank = rank_for(record["shape"], k, one_sided)
        basis, _ = top_directions(
            task_inputs[name], generic_inputs[name], rank, delta
        )
        row: dict[str, float] = {"rank": rank}
        row["own_in"], row["generic_in"], row["sel_in"] = energy_ratio(
            basis, task_inputs[name], generic_inputs[name]
        )
        reference = torch.linalg.qr(torch.randn(columns, rank, generator=random))[0]
        row["own_in_random"], _, row["sel_in_random"] = energy_ratio(
            reference, task_inputs[name], generic_inputs[name]
        )
        output_basis = None
        if not one_sided:
            output_basis, _ = top_directions(
                task_outputs[name], generic_outputs[name], rank, delta
            )
            row["own_out"], row["generic_out"], row["sel_out"] = energy_ratio(
                output_basis, task_outputs[name], generic_outputs[name]
            )
            reference = torch.linalg.qr(torch.randn(rows, rank, generator=random))[0]
            row["own_out_random"], _, row["sel_out_random"] = energy_ratio(
                reference, task_outputs[name], generic_outputs[name]
            )
        bases[name] = {
            "basis": basis.cpu(),
            "output_basis": None if output_basis is None else output_basis.cpu(),
        }
        report[name] = row
        if index % 16 == 0 or index == len(records):
            print(f"[subspace]   {index}/{len(records)} eigenproblems", flush=True)

    def median(key: str) -> float:
        values = sorted(
            row[key] for row in report.values() if key in row and not math.isnan(row[key])
        )
        return values[len(values) // 2] if values else float("nan")

    print("=" * 78, flush=True)
    if no_reference:
        # The generic energy of an orthonormal basis against the identity is
        # rank/size, so 'sel' here reads as concentration of task energy
        # relative to a random subspace -- not selectivity towards a reference.
        print(
            "no-reference ablation: the denominator is the identity, so 'sel' "
            "reads as\n  concentration of task energy against a random basis, "
            "not selectivity."
        )
    print(
        f"Basis report -- k={k}"
        f"{' (one-sided)' if one_sided else ' (both sides)'}, delta={delta:g}, "
        f"medians over {len(records)} matrices"
    )
    print(
        f"  {'side':8s}{'own':>10s}{'generic':>12s}{'sel':>10s}"
        f"   | {'own':>9s}{'sel':>8s}   (right: random basis)"
    )
    sides = [("input", "_in")] + ([] if one_sided else [("output", "_out")])
    for label, suffix in sides:
        print(
            f"  {label:8s}{median('own' + suffix):10.4f}"
            f"{median('generic' + suffix):12.2e}{median('sel' + suffix):10.2f}"
            f"   | {median('own' + suffix + '_random'):9.4f}"
            f"{median('sel' + suffix + '_random'):8.2f}"
        )
    selectivity = median("sel_in")
    output_selectivity = float("nan") if one_sided else median("sel_out")
    if no_reference:
        print(
            f"\n  ABLATION: median concentration {selectivity:.2f}x on the input "
            f"side"
            + ("" if one_sided else f", {output_selectivity:.2f}x on the output side")
            + ".\n  These bases are not selective by construction."
        )
    elif selectivity <= 1.5 and (one_sided or output_selectivity <= 1.5):
        print(
            "\n  WARNING: the bases are not selective (sel ~ 1, no better than "
            "random).\n  Raise delta, change D_gen, or use more batches before "
            "spending GPU hours here."
        )
    else:
        print(
            f"\n  selective: median sel={selectivity:.2f}"
            + ("" if one_sided else f" (input), {output_selectivity:.2f} (output)")
            + ", against about 1.0 for a random basis."
        )
    print("=" * 78, flush=True)
    return bases, report


# ── cache and entry point ──────────────────────────────────────────────────

def cache_path(values: Mapping[str, Any], root: Path, model_name: str, task: str) -> Path:
    """Deterministic name holding every knob that changes the bases, so two
    settings can never share a file."""

    k = int(values.get("k", 64))
    delta = float(values.get("delta", 1e-2))
    parts = [f"k{k}", f"d{delta:g}"]
    if values.get("one_sided", False):
        parts.append("1s")
    if values.get("no_reference", False):
        parts.append("noref")
    return root / f"{model_name}_{task}_{'_'.join(parts)}.pt"


def prepare(
    settings: Mapping[str, Any],
    config: Mapping[str, Any],
    task: str,
    model,
    named_parameters,
    loader,
    loss_function,
    device,
    seed: int,
    cache_root: Path,
) -> SubspaceTraining | None:
    """Build or load the bases, install the parametrisations, return the handle.

    Returns ``None`` for any method other than 'subspace', so the common
    trainer can call this unconditionally.
    """

    if settings.get("method") != "subspace":
        return None
    validate(settings)
    values = dict(settings["subspace"])
    k = int(values.get("k", 64))
    delta = float(values.get("delta", 1e-2))
    one_sided = bool(values.get("one_sided", False))
    no_reference = bool(values.get("no_reference", False))
    batches = int(values.get("batches", 30))

    owner = getattr(model, "module", model)
    records = eligible(owner, named_parameters)
    shapes: dict[tuple[int, int], int] = {}
    for record in records:
        shapes[record["shape"]] = shapes.get(record["shape"], 0) + 1
    print(
        f"[subspace] {len(records)} eligible matrices: "
        + ", ".join(f"{count}x({s[0]}x{s[1]})" for s, count in sorted(shapes.items())),
        flush=True,
    )

    root = Path(values.get("cache") or cache_root)
    path = cache_path(values, root, str(config.get("name", "run")), task)
    bases = None
    if path.exists() and not values.get("rebuild", False):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        missing = [r["name"] for r in records if r["name"] not in blob["bases"]]
        if missing:
            raise ValueError(
                f"{path} covers {len(blob['bases'])} matrices but this model has "
                f"{len(records)}; first missing: {missing[0]}. Delete the file or "
                f"set train.subspace.rebuild=true."
            )
        bases = blob["bases"]
        print(f"[subspace] bases loaded from {path}", flush=True)

    if bases is None:
        for _, parameter in named_parameters:
            parameter.requires_grad_(False)
        for record in records:
            record["parameter"].requires_grad_(True)
        print(f"[subspace] task gradients: {batches} batches", flush=True)
        task_inputs, task_outputs, seen = accumulate_grams(
            model, records, loader, batches, "task", device,
            loss_function=loss_function,
        )
        dead = [
            record["name"]
            for record in records
            if float(torch.diagonal(task_inputs[record["name"]]).sum()) <= 0
        ]
        if dead:
            # The eigenvectors of a zero matrix are arbitrary, so the basis
            # would be meaningless and training would silently run in a random
            # subspace. Better to stop here.
            raise RuntimeError(
                f"{len(dead)} tensor(s) received no gradient from the task loss: "
                f"{dead[:8]}"
            )
        if no_reference:
            print("[subspace] no_reference: the denominator is the identity", flush=True)
            generic_inputs = {
                r["name"]: torch.eye(r["shape"][1], device=device) for r in records
            }
            generic_outputs = {
                r["name"]: torch.eye(r["shape"][0], device=device) for r in records
            }
            generic_batches = 0
        else:
            generic_batches = int(values.get("generic_batches", 0)) or batches
            generic = _generic_loader(
                values, owner, int(settings.get("batch_size", 128)), seed
            )
            probe = next(iter(generic))
            print(
                f"[subspace] D_gen check: batch {tuple(probe['images'].shape)}",
                flush=True,
            )
            print(f"[subspace] generic gradients: {generic_batches} batches", flush=True)
            generator = torch.Generator(device=device).manual_seed(seed + 1)
            generic_inputs, generic_outputs, _ = accumulate_grams(
                model, records, generic, generic_batches, "generic", device,
                generator=generator,
            )
        print("[subspace] solving the eigenproblems (fp64, on CPU)", flush=True)
        bases, report = solve_bases(
            records, task_inputs, task_outputs, generic_inputs, generic_outputs,
            k, delta, one_sided, seed, no_reference,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "bases": bases,
                "report": report,
                "meta": {
                    "task": task,
                    "name": config.get("name"),
                    "base_model": config.get("base_model"),
                    "k": k,
                    "delta": delta,
                    "one_sided": one_sided,
                    "no_reference": no_reference,
                    "task_batches": seen,
                    "generic_batches": generic_batches,
                    "batch_size": int(settings.get("batch_size", 128)),
                    "seed": seed,
                    "generic_dir": None if no_reference else values.get("generic_dir"),
                    "matrices": [record["name"] for record in records],
                },
            },
            path,
        )
        print(f"[subspace] bases saved to {path}", flush=True)
        del task_inputs, task_outputs, generic_inputs, generic_outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    handle = SubspaceTraining(
        records, bases, device, bool(values.get("train_rest", False))
    )
    drift = handle.drift()
    print(
        f"[subspace] {handle.metadata()['core_parameters']} trainable core "
        f"parameters; max |W - W_0| at init = {drift:.3e} "
        f"{'[ok]' if drift < 1e-5 else '[FAIL]'}",
        flush=True,
    )
    if drift >= 1e-5:
        raise RuntimeError(
            "the reparametrisation does not start at the pre-trained weights; "
            "refusing to train"
        )
    return handle

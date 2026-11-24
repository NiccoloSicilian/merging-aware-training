"""open_clip adapter for the eight-expert CLIP vision benchmark.

Same tasks, splits, prompts and protocol as ``clip8_vit``; the only difference
is the backbone implementation. It exists because the two implementations of
CLIP do not expose the same tensors:

* open_clip keeps one fused ``in_proj_weight`` of shape 2304x768 per block,
  while HF splits it into ``q_proj``, ``k_proj`` and ``v_proj`` of 768x768;
* open_clip keeps the output projection ``proj`` inside ``visual``, while HF
  holds ``visual_projection`` outside the vision tower and FusionBench freezes
  it.

For a method that constrains each matrix to a subspace this is not cosmetic.
The gradient of the fused matrix is the vertical stack of the three, so
A_fused = A_q + A_k + A_v: the fused layout forces one shared input basis
across q, k and v and lets the output basis mix their rows, whereas the split
layout gives each its own. At k=64 that is 203,204 trainable parameters here
against 297,412 under HF, and ``proj`` is trainable here and absent there.
Results from the two backbones are therefore not interchangeable, which is why
this is a separate experiment rather than a flag.

Prompt templates, the frozen-head module and the checkpoint store are imported
from ``clip8_vit`` so the two backbones cannot drift apart on anything except
the backbone.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, default_collate

from ..clip_data import (
    CLIP8_TASKS,
    VisionTaskData,
    canonical_task,
    load_fusionbench_task,
)
from ..contracts import CheckpointRef
from ..runtime import atomic_write_json
from .clip8_vit import _PROMPTS, FrozenZeroShotHead, _TorchStateStore


TASKS = CLIP8_TASKS
# The runner builds the model before the loader so the exact open_clip
# transform can be reused instead of constructing a second model for it.
MODEL_BEFORE_LOADER = True

DEFAULT_ARCH = "ViT-B-32-quickgelu"
_WEIGHT_NAMES = (
    "open_clip_pytorch_model.bin",
    "open_clip_model.safetensors",
    "pytorch_model.bin",
)


def _data_root(config: Mapping[str, Any]) -> Path:
    return Path(config["data_root"])


def task_names(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Canonicalize a config task list, preserving the eight-task order."""

    names = tuple(canonical_task(task) for task in config["tasks"])
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate CLIP8 task in config: {names}")
    return names


def _device(config: Mapping[str, Any]) -> torch.device:
    requested = config.get("device")
    if requested is not None:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _arch(config: Mapping[str, Any]) -> str:
    return str(config.get("open_clip_arch", DEFAULT_ARCH))


def _resolve_pretrained(config: Mapping[str, Any]) -> tuple[str, str | None]:
    """Return (pretrained, cache_dir) for ``create_model_and_transforms``.

    Two unambiguous forms, because open_clip resolves weights in two different
    ways and guessing between them is how an offline run silently becomes a
    download:

    * ``base_model`` is a weight file -> it is passed as ``pretrained``
      directly, which is the open_clip state-dict path;
    * ``base_model`` is a directory -> it is the open_clip cache directory and
      ``open_clip_pretrained`` (default ``openai``) is the tag. open_clip then
      finds the cached archive instead of fetching it, which is how the OpenAI
      weights are meant to be loaded: they ship as a TorchScript archive that
      the state-dict path cannot read.
    """

    source = str(config["base_model"])
    path = Path(source).expanduser()
    tag = str(config.get("open_clip_pretrained", "openai"))
    if path.is_file():
        return str(path), None
    if path.is_dir():
        for name in _WEIGHT_NAMES:
            candidate = path / name
            if candidate.exists():
                return str(candidate), None
        return tag, str(path)
    if path.suffix in {".pt", ".bin", ".safetensors"}:
        raise FileNotFoundError(f"open_clip weight file not found: {path}")
    # Neither a file nor an existing directory: treat base_model as the cache
    # location so the first run populates it, rather than failing late.
    return tag, str(path)


def _require_quick_gelu(arch: str, pretrained: str) -> None:
    """Refuse the OpenAI weights under a plain-GELU architecture.

    The OpenAI CLIP weights were trained with QuickGELU. open_clip decides the
    activation from the architecture name, and ``ViT-B-32`` with
    ``pretrained='openai'`` silently uses plain GELU: it loads the same
    tensors, trains without error, and simply scores several points lower.
    Older open_clip releases applied QuickGELU implicitly for these weights, so
    a recipe written against one of those is wrong on a current install without
    anything saying so.
    """

    looks_openai = "openai" in pretrained.lower() or pretrained.lower() == "openai"
    if looks_openai and "quickgelu" not in arch.lower():
        raise ValueError(
            f"open_clip_arch={arch!r} with pretrained={pretrained!r} would run "
            f"the OpenAI weights under plain GELU, which costs several accuracy "
            f"points without raising an error. Use "
            f"open_clip_arch='{arch}-quickgelu'."
        )


@dataclass(frozen=True)
class _OpenClipParts:
    model: nn.Module
    tokenizer: Any
    train_preprocess: Any
    eval_preprocess: Any


_PREPROCESS_CACHE: dict[str, tuple[Any, Any]] = {}


def _build_open_clip(config: Mapping[str, Any]) -> _OpenClipParts:
    try:
        import open_clip
    except ImportError as error:  # pragma: no cover - depends on vision env
        raise ImportError(
            "the open_clip CLIP8 backend requires open-clip-torch"
        ) from error
    arch = _arch(config)
    pretrained, cache_dir = _resolve_pretrained(config)
    _require_quick_gelu(arch, pretrained)
    model, train_preprocess, eval_preprocess = open_clip.create_model_and_transforms(
        arch, pretrained=pretrained, cache_dir=cache_dir
    )
    try:
        tokenizer = open_clip.get_tokenizer(arch)
    except AttributeError:  # pragma: no cover - older open_clip
        tokenizer = open_clip.tokenize
    # Only ``visual`` trains; everything else exists to build the frozen heads.
    model.requires_grad_(False)
    model.visual.requires_grad_(True)
    return _OpenClipParts(
        model=model,
        tokenizer=tokenizer,
        train_preprocess=train_preprocess,
        eval_preprocess=eval_preprocess,
    )


def _text_features(
    model: nn.Module, tokenizer: Any, prompts: Sequence[str], device: torch.device
) -> torch.Tensor:
    tokens = tokenizer(list(prompts))
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.to(device)
    return model.encode_text(tokens)


def _zero_shot_class_weights(
    model: nn.Module,
    tokenizer: Any,
    classnames: Sequence[str],
    templates: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    """One prompt-encoding pass per class, as in the HF backend."""

    weights = []
    for classname in classnames:
        prompts = [template.format(classname=classname) for template in templates]
        embeddings = F.normalize(
            _text_features(model, tokenizer, prompts, device), dim=-1
        ).mean(dim=0)
        weights.append(F.normalize(embeddings, dim=-1))
    return torch.stack(weights)


class OpenClipVisionBundle(nn.Module):
    """The trainable visual tower plus lazily built frozen text heads.

    ``_text_model`` is deliberately unregistered, so the only serializable
    component is ``visual``: text-tower parameters can never enter a task
    vector. It is released as soon as the one required head is built, which is
    the independent-expert workflow the benchmark assumes.
    """

    def __init__(self, *, parts: _OpenClipParts, device: torch.device, data_root: Path):
        super().__init__()
        # In open_clip the output projection lives inside ``visual``, so it is
        # part of the expert and of every task vector. The HF backend holds it
        # outside the tower and keeps it frozen; this is one of the reasons the
        # two backbones are not interchangeable.
        self.visual = parts.model.visual
        self.heads = nn.ModuleDict()
        self.train_preprocess = parts.train_preprocess
        self.eval_preprocess = parts.eval_preprocess
        self._task_info: dict[str, tuple[str, ...]] = {}
        self._head_device = device
        self._data_root = data_root
        self.__dict__["_text_model"] = parts.model
        self.__dict__["_tokenizer"] = parts.tokenizer

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.visual(images)
        return features[0] if isinstance(features, tuple) else features

    def register_task(self, task: str, classnames: Sequence[str]) -> None:
        canonical = canonical_task(task)
        names = tuple(str(name) for name in classnames)
        if not names:
            raise ValueError(f"{canonical} has no class names")
        prior = self._task_info.get(canonical)
        if prior is not None and prior != names:
            raise ValueError(f"inconsistent class names registered for {canonical}")
        self._task_info[canonical] = names

    @torch.no_grad()
    def _build_head(self, task: str) -> FrozenZeroShotHead:
        canonical = canonical_task(task)
        names = self._task_info.get(canonical)
        if names is None:
            raise RuntimeError(
                f"register a loader for {canonical} before computing its CLIP loss"
            )
        text_model = self.__dict__.get("_text_model")
        if text_model is None:
            raise RuntimeError(
                "this bundle already released its text tower after building "
                "another task head; use one bundle per independent expert"
            )
        visual_training = self.visual.training
        text_model.to(self._head_device).eval()
        try:
            weights = (
                _zero_shot_class_weights(
                    text_model,
                    self.__dict__["_tokenizer"],
                    names,
                    _PROMPTS[canonical],
                    self._head_device,
                ).float()
                * text_model.logit_scale.exp().float()
            )
        finally:
            # ``text_model.eval()`` recurses into the shared visual tower, so
            # restore the caller's mode or head construction would silently
            # disable training.
            self.visual.train(visual_training)
        head = FrozenZeroShotHead(weights).to(self._head_device)
        head.requires_grad_(False)
        self.heads[canonical] = head
        self.__dict__["_text_model"] = None
        return head

    def logits(self, task: str, images: torch.Tensor) -> torch.Tensor:
        canonical = canonical_task(task)
        head = (
            self.heads[canonical]
            if canonical in self.heads
            else self._build_head(canonical)
        )
        return head(self(images))

    def ensure_task(self, task: str) -> None:
        canonical = canonical_task(task)
        if canonical not in self._task_info:
            metadata = load_fusionbench_task(
                canonical, "full_train", root=self._data_root, max_samples=0
            )
            self.register_task(canonical, metadata.classnames)


def _fallback_preprocess(config: Mapping[str, Any], training: bool):
    key = f"{_arch(config)}|{config['base_model']}"
    if key in _PREPROCESS_CACHE:
        return _PREPROCESS_CACHE[key][0 if training else 1]
    raise RuntimeError(
        "build_model must run before build_loader for the open_clip backend"
    )


def _collate(examples: list[tuple[torch.Tensor, int]], *, task: str) -> dict[str, Any]:
    images, labels = default_collate(examples)
    return {"images": images, "labels": labels, "task": task}


def build_loader(
    config: Mapping[str, Any],
    task: str,
    split: str = "train",
    max_samples: int | None = None,
    sample_seed: int = 0,
) -> DataLoader:
    """Return an exact local TA8 loader; batches include their task identity."""

    if split == "train" and not config["data"]["train_full_split"]:
        raise ValueError("CLIP8 protocol requires data.train_full_split=true")
    source_split = "full_train" if split == "train" else split
    loaded: VisionTaskData = load_fusionbench_task(
        task,
        source_split,
        root=_data_root(config),
        max_samples=max_samples,
        sample_seed=sample_seed,
    )
    transform = _fallback_preprocess(config, training=split == "train")
    dataset = loaded.dataset.with_transform(transform)
    train = split == "train"
    phase_config = config["train" if train else "evaluation"]
    workers = int(phase_config["num_workers"])
    generator = torch.Generator().manual_seed(int(config["seed"]))
    loader = DataLoader(
        dataset,
        batch_size=int(phase_config["batch_size"]),
        shuffle=train,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=generator if train else None,
        collate_fn=partial(_collate, task=loaded.task),
    )
    loader.smat_task = loaded.task  # type: ignore[attr-defined]
    loader.smat_classnames = loaded.classnames  # type: ignore[attr-defined]
    loader.smat_sample_class_weights = loaded.sample_class_weights  # type: ignore[attr-defined]
    loader.smat_sampling = loaded.sampling  # type: ignore[attr-defined]
    return loader


def build_model(
    config: Mapping[str, Any],
    checkpoint: str | Path | None = None,
    training: bool = True,
) -> OpenClipVisionBundle:
    """Load one open_clip backbone, retaining only its trainable visual tower."""

    smoothing = (
        float(config.get("train", {}).get("label_smoothing", 0.0)) if training else 0.0
    )
    if not 0.0 <= smoothing <= 1.0:
        raise ValueError("label_smoothing must be between zero and one")
    device = _device(config)
    parts = _build_open_clip(config)
    bundle = OpenClipVisionBundle(
        parts=parts, device=device, data_root=_data_root(config)
    ).to(device)
    bundle.label_smoothing = smoothing
    _PREPROCESS_CACHE[f"{_arch(config)}|{config['base_model']}"] = (
        parts.train_preprocess,
        parts.eval_preprocess,
    )
    if checkpoint is not None:
        bundle.visual.load_state_dict(_image_encoder_state(checkpoint), strict=True)
    bundle.train(training)
    return bundle


def _checkpoint_file(checkpoint: str | Path) -> Path:
    """Experts and merges are directories; their tensors are one file."""

    path = Path(checkpoint)
    if path.is_dir():
        for name in ("encoder.pt", "model.safetensors", "pytorch_model.bin"):
            candidate = path / name
            if candidate.exists():
                return candidate
        return path / "encoder.pt"
    if path.suffix in {".pt", ".bin", ".safetensors"}:
        return path
    return path / "encoder.pt"


def checkpoint_path(config: Mapping[str, Any], directory: str | Path) -> Path:
    del config
    return _checkpoint_file(directory)


def _torch_load(path: Path) -> Any:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def _image_encoder_state(checkpoint: str | Path) -> dict[str, torch.Tensor]:
    """Read an expert state, or select the visual tower out of a full CLIP."""

    path = _checkpoint_file(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"image encoder checkpoint not found: {path}")
    raw = _torch_load(path)
    if isinstance(raw, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = raw.get(key)
            if isinstance(candidate, Mapping):
                raw = candidate
                break
    if not isinstance(raw, Mapping):
        raise ValueError(f"unsupported image-encoder checkpoint: {path}")
    tensors = {
        str(key): value for key, value in raw.items() if isinstance(value, torch.Tensor)
    }
    for prefix in ("visual.", "module.visual."):
        visual = {
            key.removeprefix(prefix): value
            for key, value in tensors.items()
            if key.startswith(prefix)
        }
        if visual:
            return visual
    return tensors


class _OpenClipCheckpointAdapter:
    """Adapter projecting full CLIP bases to visual-tower tensors only."""

    @contextmanager
    def read(self, checkpoint: CheckpointRef):
        yield _TorchStateStore(_image_encoder_state(checkpoint))

    @contextmanager
    def write(self, output: CheckpointRef, *, template: CheckpointRef):
        del template
        store = _TorchStateStore()
        try:
            yield store
        except BaseException:
            raise
        else:
            path = _checkpoint_file(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(store.tensors, path)


def checkpoint_adapter(config: Mapping[str, Any]) -> _OpenClipCheckpointAdapter:
    del config
    return _OpenClipCheckpointAdapter()


def register_loader(model: OpenClipVisionBundle, loader: DataLoader) -> None:
    model.register_task(loader.smat_task, loader.smat_classnames)  # type: ignore[attr-defined]


def named_parameters(
    config: Mapping[str, Any], model: OpenClipVisionBundle
) -> list[tuple[str, torch.nn.Parameter]]:
    """Optimizer and perturbations see the visual tower only.

    This is where the two backbones diverge for a subspace method: the fused
    ``transformer.resblocks.N.attn.in_proj_weight`` (2304x768) appears as one
    tensor, and ``proj`` (768x512) is here rather than frozen outside.
    """

    del config
    return list(model.visual.named_parameters())


def loss(
    model: OpenClipVisionBundle, batch: Mapping[str, Any], device: str | torch.device
) -> torch.Tensor:
    """Cross entropy over the frozen task-specific zero-shot CLIP head."""

    # The head belongs to the bundle, but features must go through
    # DDP.forward when the runner wraps it, or the reducer is bypassed and the
    # visual gradients stay local.
    owner = getattr(model, "module", model)
    owner._head_device = torch.device(device)
    task = canonical_task(str(batch["task"]))
    owner.ensure_task(task)
    head = owner.heads[task] if task in owner.heads else owner._build_head(task)
    images = batch["images"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    return F.cross_entropy(
        head(model(images)),
        labels,
        label_smoothing=getattr(owner, "label_smoothing", 0.0),
    )


def save_model(model: OpenClipVisionBundle, output: str | Path) -> Path:
    """Persist precisely the visual tower, never a text head or tower."""

    output = _checkpoint_file(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.visual.state_dict(), output)
    return output


class _OpenClipTaskEvaluator:
    """One evaluator reused across checkpoints for one task."""

    def __init__(self, config: Mapping[str, Any], task: str, max_samples: int | None):
        self.config = config
        self.task = canonical_task(task)
        self.split = str(config["evaluation"]["split"])
        self.device = _device(config)
        self.model = build_model(config, training=False)
        self.loader = build_loader(
            config, self.task, split=self.split, max_samples=max_samples
        )
        register_loader(self.model, self.loader)
        self.model.eval()
        self.model._build_head(self.task)

    def evaluate(self, checkpoint: str | Path, output: str | Path) -> float:
        state = _image_encoder_state(checkpoint)
        self.model.visual.load_state_dict(state, strict=True)
        del state
        correct = total = 0
        for batch in self.loader:
            labels = batch["labels"].to(self.device, non_blocking=True)
            predictions = self.model.logits(
                batch["task"], batch["images"].to(self.device, non_blocking=True)
            ).argmax(dim=-1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
        accuracy = correct / total if total else 0.0
        atomic_write_json(
            Path(output),
            {
                "protocol": "clip8",
                "provider": "open_clip",
                "arch": _arch(self.config),
                "task": self.task,
                "split": self.split,
                "checkpoint": str(Path(checkpoint)),
                "top1": accuracy,
                "correct": correct,
                "num_samples": total,
            },
        )
        return accuracy


@torch.inference_mode()
def evaluate_many(
    config: Mapping[str, Any],
    task: str,
    requests: Sequence[tuple[str | Path, str | Path]],
    max_samples: int | None = None,
) -> list[float]:
    """Evaluate several checkpoints on one task through one session."""

    if not requests:
        return []
    evaluator = _OpenClipTaskEvaluator(config, task, max_samples)
    return [evaluator.evaluate(checkpoint, output) for checkpoint, output in requests]


def evaluate(
    config: Mapping[str, Any],
    checkpoint: str | Path,
    task: str,
    output: str | Path,
    max_samples: int | None = None,
) -> float:
    return evaluate_many(config, task, ((checkpoint, output),), max_samples)[0]

"""Numerical and structural checks for subspace-constrained expert training.

The eigenproblem, the reparametrisation and the eligibility rule are checked
without transformers or a downloaded CLIP: a small module reproduces the HF
CLIP vision parameter layout, which is what the rule has to match.
"""

import math
import unittest

import torch
from torch import nn

from smat.train.subspace import (
    SubspaceTraining,
    eligible,
    energy_ratio,
    rank_for,
    top_directions,
    validate,
)


WIDTH, HEADS, MLP, LAYERS, POSITIONS = 768, 12, 3072, 12, 50


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("k_proj", "v_proj", "q_proj", "out_proj"):
            setattr(self, name, nn.Linear(WIDTH, WIDTH))


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(WIDTH, MLP)
        self.fc2 = nn.Linear(MLP, WIDTH)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()
        self.layer_norm1 = nn.LayerNorm(WIDTH)
        self.mlp = _Mlp()
        self.layer_norm2 = nn.LayerNorm(WIDTH)


class _Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.class_embedding = nn.Parameter(torch.randn(WIDTH))
        self.patch_embedding = nn.Conv2d(3, WIDTH, 32, 32, bias=False)
        self.position_embedding = nn.Embedding(POSITIONS, WIDTH)


class VisionTowerStub(nn.Module):
    """The HF CLIPVisionTransformer parameter layout for ViT-B/32."""

    def __init__(self):
        super().__init__()
        self.embeddings = _Embeddings()
        self.pre_layrnorm = nn.LayerNorm(WIDTH)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(_Layer() for _ in range(LAYERS))
        self.post_layernorm = nn.LayerNorm(WIDTH)


class EligibilityTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.tower = VisionTowerStub()
        self.named = list(self.tower.named_parameters())

    def test_selects_every_two_dimensional_weight(self):
        records = eligible(self.tower, self.named)
        self.assertEqual(len(records), LAYERS * 6 + 1)
        shapes = sorted(
            (shape, sum(1 for r in records if r["shape"] == shape))
            for shape in {r["shape"] for r in records}
        )
        self.assertEqual(
            shapes,
            [
                ((POSITIONS, WIDTH), 1),
                ((WIDTH, WIDTH), LAYERS * 4),
                ((WIDTH, MLP), LAYERS),
                ((MLP, WIDTH), LAYERS),
            ],
        )

    def test_excludes_conv_layernorms_and_biases(self):
        names = {record["name"] for record in eligible(self.tower, self.named)}
        for excluded in (
            "embeddings.patch_embedding.weight",
            "embeddings.class_embedding",
            "pre_layrnorm.weight",
            "post_layernorm.bias",
            "encoder.layers.0.self_attn.q_proj.bias",
            "encoder.layers.0.layer_norm1.weight",
        ):
            self.assertNotIn(excluded, names)
        self.assertIn("encoder.layers.11.mlp.fc2.weight", names)
        self.assertIn("embeddings.position_embedding.weight", names)

    def test_names_resolve_relative_to_a_submodule(self):
        # The CLIP adapter hands out names relative to its own encoder, so
        # resolution must not depend on which root they came from.
        named = list(self.tower.encoder.named_parameters())
        records = eligible(self.tower, named)
        self.assertEqual(len(records), LAYERS * 6)
        for record in records:
            self.assertIs(
                getattr(record["module"], record["attribute"]), record["parameter"]
            )

    def test_unknown_parameter_is_reported(self):
        stray = [("stray", nn.Parameter(torch.randn(4, 4)))]
        with self.assertRaisesRegex(RuntimeError, "not a parameter"):
            eligible(self.tower, stray)

    def test_rank_is_clamped_to_the_short_side(self):
        self.assertEqual(rank_for((MLP, WIDTH), 64, False), 64)
        self.assertEqual(rank_for((POSITIONS, WIDTH), 64, False), POSITIONS)
        self.assertEqual(rank_for((POSITIONS, WIDTH), 64, True), 64)


class EigenproblemTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.size, self.rank = 40, 6
        samples = torch.randn(200, self.size)
        self.gram = samples.T @ samples / 200

    def _overlap(self, left, right):
        return (left.T @ right).pow(2).sum().item() / left.shape[1]

    def test_identity_denominator_gives_the_ordinary_eigenvectors(self):
        basis, _ = top_directions(
            self.gram, torch.eye(self.size), self.rank, 0.0
        )
        values, vectors = torch.linalg.eigh(self.gram.double())
        order = torch.argsort(values, descending=True)[: self.rank]
        self.assertAlmostEqual(
            self._overlap(basis, vectors[:, order].float()), 1.0, places=4
        )

    def test_columns_are_orthonormal(self):
        basis, _ = top_directions(self.gram, torch.eye(self.size), self.rank, 1e-2)
        self.assertTrue(
            torch.allclose(basis.T @ basis, torch.eye(self.rank), atol=1e-5)
        )

    def test_ridge_is_scale_free(self):
        plain, _ = top_directions(self.gram, torch.eye(self.size), self.rank, 1e-2)
        scaled, _ = top_directions(
            self.gram * 1000, torch.eye(self.size) * 1000, self.rank, 1e-2
        )
        self.assertGreater(self._overlap(plain, scaled), 0.999)

    def test_a_singular_denominator_still_yields_a_basis(self):
        # Rank-deficient generic Gram: the Cholesky fails and the eigenvalue
        # floor has to take over instead of propagating the exception.
        denominator = torch.zeros(self.size, self.size)
        denominator[0, 0] = 1.0
        basis, _ = top_directions(self.gram, denominator, self.rank, 0.0)
        self.assertEqual(tuple(basis.shape), (self.size, self.rank))
        self.assertTrue(torch.isfinite(basis).all())

    def test_the_generic_direction_is_avoided(self):
        denominator = torch.zeros(self.size, self.size)
        denominator[0, 0] = 100.0
        numerator = self.gram.clone()
        numerator[0, 0] += 100.0
        selective, _ = top_directions(numerator, denominator, self.rank, 1e-2)
        plain, _ = top_directions(
            numerator, torch.eye(self.size), self.rank, 1e-2
        )
        self.assertLess(
            selective[0].pow(2).sum().item(), plain[0].pow(2).sum().item()
        )
        self.assertGreater(
            energy_ratio(selective, numerator, denominator)[2],
            energy_ratio(plain, numerator, denominator)[2],
        )


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 8)
        self.loose = nn.Parameter(torch.randn(8, 16))

    def forward(self, values):
        return self.linear(values) @ self.loose


class ReparametrisationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.module = _Tiny()
        self.named = list(self.module.named_parameters())
        self.records = eligible(self.module, self.named)
        self.initial = {
            record["name"]: record["parameter"].detach().clone()
            for record in self.records
        }
        self.bases = {
            record["name"]: {
                "basis": torch.linalg.qr(torch.randn(record["shape"][1], 4))[0],
                "output_basis": torch.linalg.qr(torch.randn(record["shape"][0], 4))[0],
            }
            for record in self.records
        }

    def _install(self, train_rest=False):
        handle = SubspaceTraining(self.records, self.bases, "cpu", train_rest)
        self.named = handle.trainable_named_parameters(self.named)
        return handle

    def test_training_starts_at_the_pretrained_weights(self):
        handle = self._install()
        self.assertLess(handle.drift(), 1e-6)
        self.assertEqual(handle.core_norm(), 0.0)

    def test_only_the_cores_are_trainable_by_default(self):
        self._install()
        trainable = sorted(
            name for name, parameter in self.module.named_parameters()
            if parameter.requires_grad
        )
        self.assertEqual(
            trainable,
            [
                "linear.parametrizations.weight.0.core",
                "parametrizations.loose.0.core",
            ],
        )

    def test_train_rest_adds_the_uncovered_tensors(self):
        handle = self._install(train_rest=True)
        names = [name for name, _ in handle.trainable_named_parameters(
            list(self.module.named_parameters())
        )]
        self.assertTrue(any(name.endswith("::core") for name in names))
        self.assertTrue(any("bias" in name for name in names))

    def test_gradients_reach_the_cores(self):
        self._install()
        self.module(torch.randn(5, 16)).pow(2).sum().backward()
        for record in self.records:
            gradient = record["delta"].core.grad
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_the_delta_stays_inside_the_subspace(self):
        handle = self._install()
        with torch.no_grad():
            for record in self.records:
                record["delta"].core.add_(
                    torch.randn_like(record["delta"].core) * 0.05
                )
        for record in self.records:
            delta = (
                getattr(record["module"], record["attribute"]).detach()
                - self.initial[record["name"]]
            )
            entry = self.bases[record["name"]]
            projected = entry["output_basis"].T @ delta @ entry["basis"]
            total = float(delta.pow(2).sum())
            self.assertAlmostEqual(
                float(projected.pow(2).sum()) / total, 1.0, places=4
            )
            self.assertLessEqual(
                int(torch.linalg.matrix_rank(delta, tol=1e-5)), 4
            )

    def test_core_norm_equals_the_delta_norm(self):
        handle = self._install()
        with torch.no_grad():
            for record in self.records:
                record["delta"].core.add_(
                    torch.randn_like(record["delta"].core) * 0.05
                )
        measured = math.sqrt(
            sum(
                float(
                    (
                        getattr(record["module"], record["attribute"]).detach()
                        - self.initial[record["name"]]
                    )
                    .pow(2)
                    .sum()
                )
                for record in self.records
            )
        )
        self.assertAlmostEqual(measured, handle.core_norm(), places=4)

    def test_materialise_restores_a_plain_state_dict(self):
        handle = self._install()
        with torch.no_grad():
            for record in self.records:
                record["delta"].core.add_(
                    torch.randn_like(record["delta"].core) * 0.05
                )
        before = handle.core_norm()
        self.assertTrue(
            any("parametrizations" in key for key in self.module.state_dict())
        )
        handle.materialise()
        self.assertEqual(
            set(self.module.state_dict()),
            {"linear.weight", "linear.bias", "loose"},
        )
        carried = math.sqrt(
            sum(
                float((self.module.state_dict()[name] - tensor).pow(2).sum())
                for name, tensor in self.initial.items()
            )
        )
        self.assertAlmostEqual(carried, before, places=4)
        handle.materialise()  # idempotent


class ValidationTest(unittest.TestCase):
    def test_a_reference_set_or_the_ablation_is_required(self):
        with self.assertRaisesRegex(ValueError, "generic_dir"):
            validate({"method": "subspace", "subspace": {"k": 64}})
        validate({"method": "subspace", "subspace": {"no_reference": True}})
        validate({"method": "subspace", "subspace": {"generic_dir": "/data/x"}})

    def test_misspelled_options_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            validate(
                {"method": "subspace", "subspace": {"no_reference": True, "rnak": 8}}
            )

    def test_the_block_must_be_an_object(self):
        with self.assertRaisesRegex(ValueError, "must be an object"):
            validate({"method": "subspace"})

    def test_out_of_range_values_are_rejected(self):
        for values in ({"k": 0}, {"delta": -1.0}, {"batches": -3}):
            with self.assertRaises(ValueError):
                validate({"method": "subspace", "subspace": {**values, "no_reference": True}})


class StepperDispatchTest(unittest.TestCase):
    def test_subspace_uses_the_ordinary_update(self):
        from smat.train.updates import FTStepper, build_stepper, validate_settings

        settings = {
            "method": "subspace",
            "optimizer": "adam",
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "schedule": "cosine",
            "max_steps": 10,
            "subspace": {"k": 8, "no_reference": True},
        }
        validate_settings(settings)
        core = nn.Parameter(torch.zeros(4, 4))
        optimizer = torch.optim.Adam([core], lr=1e-4)
        stepper = build_stepper(settings, [("c::core", core)], optimizer, 42)
        self.assertIsInstance(stepper, FTStepper)

    def test_smat_options_are_refused_for_the_subspace_method(self):
        from smat.train.updates import validate_settings

        with self.assertRaisesRegex(ValueError, "unsupported training options"):
            validate_settings(
                {
                    "method": "subspace",
                    "schedule": "cosine",
                    "subspace": {"no_reference": True},
                    "perturb": {"rms": 0.001},
                }
            )


if __name__ == "__main__":
    unittest.main()


class QuickGeluGuardTest(unittest.TestCase):
    """The OpenAI weights need QuickGELU, and a wrong config is silent."""

    def _config(self, activation):
        vision = type("VisionConfig", (), {"hidden_act": activation})()
        return type("ClipConfig", (), {"vision_config": vision})()

    def test_quick_gelu_is_accepted(self):
        from smat.experiments.clip8_vit import _require_quick_gelu

        _require_quick_gelu(self._config("quick_gelu"), "openai/clip-vit-base-patch32")

    def test_plain_gelu_is_refused(self):
        from smat.experiments.clip8_vit import _require_quick_gelu

        with self.assertRaisesRegex(ValueError, "trained with QuickGELU"):
            _require_quick_gelu(self._config("gelu"), "some/clip")

    def test_a_missing_activation_is_refused(self):
        from smat.experiments.clip8_vit import _require_quick_gelu

        with self.assertRaisesRegex(ValueError, "no vision_config.hidden_act"):
            _require_quick_gelu(type("Empty", (), {})(), "some/clip")

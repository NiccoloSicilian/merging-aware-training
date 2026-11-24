"""Checks for the open_clip CLIP8 backbone.

The point of this backbone is the tensor set it exposes, so that is what is
tested: a module reproducing the open_clip visual-tower layout must yield the
fused attention projection and the output projection, and the subspace built
over it must match the published parameter count for ViT-B/32 at k=64.

open_clip itself is not imported: these are structural and path checks.
"""

import unittest
from collections import OrderedDict

import torch
from torch import nn

from smat.experiments.clip8_openclip import (
    _image_encoder_state,
    _require_quick_gelu,
    _resolve_pretrained,
)
from smat.train.subspace import SubspaceTraining, eligible, rank_for


WIDTH, HEADS, MLP, LAYERS, POSITIONS, EMBED = 768, 12, 3072, 12, 50, 512


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(WIDTH)
        # One fused 2304x768 projection, which is the whole reason this
        # backbone exists alongside the HF one.
        self.attn = nn.MultiheadAttention(WIDTH, HEADS)
        self.ln_2 = nn.LayerNorm(WIDTH)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(WIDTH, MLP)),
                    ("gelu", nn.GELU()),
                    ("c_proj", nn.Linear(MLP, WIDTH)),
                ]
            )
        )


class VisualTowerStub(nn.Module):
    """The open_clip VisionTransformer parameter layout for ViT-B/32."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, WIDTH, 32, 32, bias=False)
        self.class_embedding = nn.Parameter(torch.randn(WIDTH))
        self.positional_embedding = nn.Parameter(torch.randn(POSITIONS, WIDTH))
        self.ln_pre = nn.LayerNorm(WIDTH)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList(_Block() for _ in range(LAYERS))
        self.ln_post = nn.LayerNorm(WIDTH)
        # In open_clip the output projection is inside the tower, so it is part
        # of the expert and of every task vector.
        self.proj = nn.Parameter(torch.randn(WIDTH, EMBED))


class TensorSetTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.tower = VisualTowerStub()
        self.records = eligible(self.tower, list(self.tower.named_parameters()))

    def test_fifty_matrices_for_vitb32(self):
        self.assertEqual(len(self.records), LAYERS * 4 + 2)

    def test_attention_is_one_fused_matrix(self):
        fused = [r for r in self.records if r["shape"] == (3 * WIDTH, WIDTH)]
        self.assertEqual(len(fused), LAYERS)
        self.assertTrue(
            all(r["name"].endswith("attn.in_proj_weight") for r in fused)
        )
        # No split projections exist on this backbone.
        self.assertFalse(
            any("q_proj" in r["name"] or "k_proj" in r["name"] for r in self.records)
        )

    def test_the_output_projection_is_trainable(self):
        names = {r["name"] for r in self.records}
        self.assertIn("proj", names)
        self.assertIn("positional_embedding", names)

    def test_conv_layernorms_and_biases_are_excluded(self):
        names = {r["name"] for r in self.records}
        for excluded in (
            "conv1.weight",
            "class_embedding",
            "ln_pre.weight",
            "ln_post.bias",
            "transformer.resblocks.0.attn.in_proj_bias",
            "transformer.resblocks.0.ln_1.weight",
        ):
            self.assertNotIn(excluded, names)

    def test_core_count_matches_the_published_configuration(self):
        # 203,204 trainable parameters at k=64 is the figure this method was
        # measured with. A backbone that changed it would make the recorded
        # results incomparable, so it is pinned here.
        bases = {}
        for record in self.records:
            rank = rank_for(record["shape"], 64, False)
            bases[record["name"]] = {
                "basis": torch.linalg.qr(
                    torch.randn(record["shape"][1], rank)
                )[0],
                "output_basis": torch.linalg.qr(
                    torch.randn(record["shape"][0], rank)
                )[0],
            }
        for parameter in self.tower.parameters():
            parameter.requires_grad_(False)
        handle = SubspaceTraining(self.records, bases, "cpu", False)
        self.assertEqual(handle.metadata()["core_parameters"], 203_204)
        self.assertLess(handle.drift(), 1e-6)

    def test_the_fused_projection_shares_one_core(self):
        # The fused layout gives q, k and v one input basis and one core; the
        # split layout would give three. This is the difference that makes the
        # two backbones' results incomparable.
        fused = next(r for r in self.records if r["shape"] == (3 * WIDTH, WIDTH))
        self.assertEqual(rank_for(fused["shape"], 64, False), 64)
        self.assertEqual(64 * 64, 4096)


class QuickGeluGuardTest(unittest.TestCase):
    def test_openai_weights_require_a_quickgelu_architecture(self):
        with self.assertRaisesRegex(ValueError, "plain GELU"):
            _require_quick_gelu("ViT-B-32", "openai")

    def test_quickgelu_architecture_is_accepted(self):
        _require_quick_gelu("ViT-B-32-quickgelu", "openai")

    def test_other_pretrained_tags_are_not_constrained(self):
        _require_quick_gelu("ViT-B-32", "laion2b_s34b_b79k")


class PretrainedResolutionTest(unittest.TestCase):
    def test_a_weight_file_is_passed_through(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "open_clip_pytorch_model.bin"
            weights.write_bytes(b"")
            pretrained, cache = _resolve_pretrained({"base_model": str(weights)})
            self.assertEqual(pretrained, str(weights))
            self.assertIsNone(cache)

    def test_a_directory_holding_open_clip_weights_is_used_directly(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "open_clip_pytorch_model.bin"
            weights.write_bytes(b"")
            pretrained, cache = _resolve_pretrained({"base_model": directory})
            self.assertEqual(pretrained, str(weights))
            self.assertIsNone(cache)

    def test_a_plain_directory_becomes_the_cache_for_a_tag(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            pretrained, cache = _resolve_pretrained({"base_model": directory})
            self.assertEqual(pretrained, "openai")
            self.assertEqual(cache, directory)

    def test_the_tag_is_configurable(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            pretrained, _ = _resolve_pretrained(
                {"base_model": directory, "open_clip_pretrained": "laion2b_s34b_b79k"}
            )
            self.assertEqual(pretrained, "laion2b_s34b_b79k")

    def test_a_missing_weight_file_is_reported(self):
        with self.assertRaisesRegex(FileNotFoundError, "not found"):
            _resolve_pretrained({"base_model": "/nonexistent/weights.pt"})


class CheckpointStateTest(unittest.TestCase):
    def test_a_visual_prefix_is_stripped(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encoder.pt"
            torch.save(
                {
                    "visual.proj": torch.zeros(2, 2),
                    "transformer.resblocks.0.attn.in_proj_weight": torch.zeros(3, 1),
                    "token_embedding.weight": torch.zeros(4, 4),
                },
                path,
            )
            state = _image_encoder_state(path)
            self.assertEqual(set(state), {"proj"})

    def test_a_bare_visual_state_is_returned_unchanged(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encoder.pt"
            torch.save({"proj": torch.zeros(2, 2), "conv1.weight": torch.zeros(1)}, path)
            self.assertEqual(
                set(_image_encoder_state(path)), {"proj", "conv1.weight"}
            )


if __name__ == "__main__":
    unittest.main()
